# Evaluation
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import EEGDataset

if __name__ == '__main__':

    segment_length = 6
    # Hyperparameters and configurations
    num_classes = 4
    batch_size = 128
    img_rows, img_cols, num_chan = 8, 9, 4

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
    falx = np.load("./features/1_segmented_x_89.npy")
    y = np.load("./features/1_segmented_y_89.npy")

    num_segments = falx.shape[1]
    one_y_1 = y.astype(int)  # Convert to integers
    one_y_1 = np.eye(num_classes)[one_y_1]  # Convert to one-hot encoded format

    one_falx_1 = falx.reshape((-1, segment_length, img_rows, img_cols, 5))
    one_falx = one_falx_1[:, :, :, :, 1:5]  # Only use four frequency bands

    dataset = EEGDataset(one_falx, one_y_1)

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                             pin_memory=True)

    model = torch.load('./results/model.pt')
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
    print(f"Accuracy: {accuracy:.2f}%")