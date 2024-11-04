import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Model(nn.Module):

    def __init__(self, in_features = 4, h1 = 8, h2 = 9, out_features = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


torch.manual_seed(41)
model = Model()

url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
df = pd.read_csv(url)
df['variety'] = df['variety'].replace('Setosa', 0.0)
df['variety'] = df['variety'].replace('Versicolor', 1.0)
df['variety'] = df['variety'].replace('Virginica', 2.0)

x = df.drop('variety', axis=1)
y = df['variety']

x = x.values
y = y.values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200
losses = []

for i in range(epochs):
    y_pred = model.forward(x_train)

    loss = criterion(y_pred, y_train)

    losses.append(loss.detach().numpy())

    print(f'Epoch: {i} and loss: {loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.plot(range(epochs), losses)
plt.ylabel("loss/error")
plt.xlabel("epochs")

plt.show()

with torch.no_grad():
    y_eval = model.forward(x_test)
    loss = criterion(y_eval, y_test)
    print(loss)



# Initialize model, loss function, and optimizer
#             model = EEGNet((img_rows, img_cols, 4)).to(device)
#             criterion = nn.CrossEntropyLoss()
#             optimizer = optim.Adam(model.parameters())
#
#             # Mixed precision training
#             scaler = torch.amp.GradScaler()
#
#             # Training Loop
#             model.train()
#             for epoch in range(100):
#                 epoch_start = time.time()
#                 running_loss = 0.0
#                 for batch_x, batch_y in train_loader:
#                     # Change the permutation order to (0, 2, 3, 1) to match the expected channel dimension
#                     inputs = [batch_x[:, i].permute(0, 3, 1, 2).to(device, non_blocking=True) for i in range(6)]
#                     labels = batch_y.argmax(dim=1).to(device, non_blocking=True)
#
#                     optimizer.zero_grad()
#
#                     # Mixed precision training
#                     with torch.amp.autocast("cuda"):
#                         outputs = model(inputs)
#                         loss = criterion(outputs, labels)
#
#                     scaler.scale(loss).backward()
#                     scaler.step(optimizer)
#                     scaler.update()
#
#                     running_loss += loss.item()
#
#                 epoch_end = time.time()
#                 # Print the average loss for the epoch and the time taken
#                 print(
#                     f"Epoch [{epoch + 1}/100], Loss: {running_loss / len(train_loader):.4f}, Time: {epoch_end - epoch_start:.2f} seconds")
#
#             # Evaluation
#             model.eval()
#             correct = 0
#             total = 0
#             with torch.no_grad():
#                 for batch_x, batch_y in test_loader:
#                     # Change the permutation order here as well
#                     inputs = [batch_x[:, i].permute(0, 3, 1, 2).to(device, non_blocking=True) for i in range(6)]
#                     labels = batch_y.argmax(dim=1).to(device, non_blocking=True)
#
#                     with torch.amp.autocast('cuda'):
#                         outputs = model(inputs)
#                         _, predicted = torch.max(outputs.data, 1)
#                         total += labels.size(0)
#                         correct += (predicted == labels).sum().item()
#
#             accuracy = 100 * correct / total
#             fold_accuracies.append(accuracy)
#             print(f"Fold Accuracy: {accuracy:.2f}%")
#
#         print(f"Subject {nb + 1} Mean Accuracy: {np.mean(fold_accuracies):.2f}%")
#         acc_list.append(np.mean(fold_accuracies))
#         std_list.append(np.std(fold_accuracies))
#         end = time.time()
#         print(f"Execution Time: {end - start:.2f} seconds")