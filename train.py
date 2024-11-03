import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import time

subjects = 15
segment_length = 6

# Define PyTorch Dataset
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]  # Shape: (6, 8, 9, 4)
        y = self.labels[idx]  # Shape: (3,)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# Define the Base Convolutional Network
class ConvNet(nn.Module):
    def __init__(self, input_dim):
        super(ConvNet, self).__init__()
        # Changed input_dim[2] to input_dim[0] to get the correct number of channels
        # The input to the convolutional layer should have 4 channels, not 8.
        self.conv1 = nn.Conv2d(input_dim[2], 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, padding=1)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        # Calculate the correct input size for dense1 dynamically
        self.dense1 = nn.Linear(self._get_conv_output_size(input_dim), 512)

    def _get_conv_output_size(self, input_dim):
        # Create a dummy input tensor with the correct shape (channels, height, width)
        # Here you should change to the correct order (batch_size, channels, height, width)
        dummy_input = torch.zeros(1, input_dim[2], input_dim[0], input_dim[1])  # Change the order to (1, 4, 8, 9)
        # Pass the dummy input through the convolutional layers
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.flatten(x)
        # Return the size of the flattened output
        return x.size(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.dense1(x))
        x = x.unsqueeze(1)  # Reshape to (batch_size, 1, 512)
        return x


# Define the Combined LSTM Model
class EEGNet(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(EEGNet, self).__init__()
        self.base_network = ConvNet(input_dim)
        self.lstm = nn.LSTM(512, 128, batch_first=True)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        # x is a list of 6 input tensors
        x = torch.cat([self.base_network(inp) for inp in x], dim=1)
        _, (h_n, _) = self.lstm(x)  # h_n is the hidden state from LSTM
        out = self.out(h_n[-1])  # Use the last hidden state
        return out


if __name__ == '__main__':
    # Hyperparameters and configurations
    num_classes = 4
    batch_size = 128
    img_rows, img_cols, num_chan = 8, 9, 4
    seed = 7
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Check GPU availability
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if cuda_available:
        device = torch.device("cuda:0")
    elif mps_available:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Data loading
    falx = np.load("./features/segmented_x_89.npy")
    y = np.load("./features/segmented_y_89.npy")

    num_segments = falx.shape[1]
    one_y_1 = np.array([y[:num_segments]] * 3).reshape((-1,))
    one_y_1 = one_y_1.astype(int)  # Convert to integers
    one_y_1 = np.eye(num_classes)[one_y_1]  # Convert to one-hot encoded format

    # Cross-Validation
    acc_list = []
    std_list = []

    # Process each subject independently
    for nb in range(15):
        start = time.time()
        one_falx_1 = falx[nb * 3:nb * 3 + 3]
        one_falx_1 = one_falx_1.reshape((-1, segment_length, img_rows, img_cols, 5))
        one_falx = one_falx_1[:, :, :, :, 1:5]  # Only use four frequency bands
        one_y = one_y_1

        dataset = EEGDataset(one_falx, one_y)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_accuracies = []

        for train_idx, test_idx in kfold.split(one_falx, one_y.argmax(1)):
            # Create DataLoader for training and testing
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4,
                                      pin_memory=True)
            test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=4,
                                     pin_memory=True)

            # Initialize model, loss function, and optimizer
            model = EEGNet((img_rows, img_cols, 4)).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters())

            # Mixed precision training
            scaler = torch.amp.GradScaler()

            # Training Loop
            model.train()
            for epoch in range(100):
                epoch_start = time.time()
                running_loss = 0.0
                for batch_x, batch_y in train_loader:
                    # Change the permutation order to (0, 2, 3, 1) to match the expected channel dimension
                    inputs = [batch_x[:, i].permute(0, 3, 1, 2).to(device, non_blocking=True) for i in range(6)]
                    labels = batch_y.argmax(dim=1).to(device, non_blocking=True)

                    optimizer.zero_grad()

                    # Mixed precision training
                    with torch.amp.autocast("cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    running_loss += loss.item()

                epoch_end = time.time()
                # Print the average loss for the epoch and the time taken
                print(
                    f"Epoch [{epoch + 1}/100], Loss: {running_loss / len(train_loader):.4f}, Time: {epoch_end - epoch_start:.2f} seconds")

            # Evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    # Change the permutation order here as well
                    inputs = [batch_x[:, i].permute(0, 3, 1, 2).to(device, non_blocking=True) for i in range(6)]
                    labels = batch_y.argmax(dim=1).to(device, non_blocking=True)

                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            fold_accuracies.append(accuracy)
            print(f"Fold Accuracy: {accuracy:.2f}%")

        print(f"Subject {nb + 1} Mean Accuracy: {np.mean(fold_accuracies):.2f}%")
        acc_list.append(np.mean(fold_accuracies))
        std_list.append(np.std(fold_accuracies))
        end = time.time()
        print(f"Execution Time: {end - start:.2f} seconds")

    # Final Results
    print(f'Acc_all: {acc_list}')
    print(f'Std_all: {std_list}')
    print(f"Acc_mean: {np.mean(acc_list)}")
    print(f"Std_all: {np.std(std_list)}")
