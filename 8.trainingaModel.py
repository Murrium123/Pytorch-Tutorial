import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print(f"\nPreparing Training Data\n")

# Create synthetic data
# 1000 samples, 2 features each (x, y coordinates)
X = torch.randn(1000,2) # I think X is the independant variable and y is the dependant variable
# Label is 1 if the point is far from the center, 0 if it is close
y = (X.pow(2).sum(1) > 1).long()

# 2. Split into Train (800) and Test (200)
train_ds = TensorDataset(X[:800], y[:800])
test_ds = TensorDataset(X[800:], y[800:])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

print(f"\nWriting the Training Loop\n")

# Define a simple model
model = nn.Sequential(
  nn.Linear(2, 10),
  nn.Relu(),
  nn.Linear(10,2) # 2 output classes (0 and 1)
)

criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 10

for epoch in range(epochs):
  model.train() # Set model to training mode
  running_loss = 0.0

  for batch_idx, (inputs, targets) in enumerate(train_loader):
    # 1. Zero the gradients
    optimizer.zero_grad()

    # 2. Forward pass
    outputs = model(inputs)

    # 3. Calculate loss
    loss = criterion(outputs, targets)

    # 4. Backward pass (Backpropogation)
    loss.backward()

    # 5. Update weights
    optimizer.step()

    running_loss += loss.item()

  # Print average loss for the epoch
  avg_loss = running_loss / len(train_loader)
  print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

print(f"/nEvaluating Performance/n")
model.eval() # Set model to evaluation mode
correct = 0
total = 0

# Disable gradient tracking for speed and memory efficiency
with torch.no_grad():
  for inputs, targets in test_loader:
    outputs = model(inputs)

    # Get the index of the highest value (out predicted class)
    _, predicted = torch.max(outputs.data, 1)

    total += targets.size(0)
    correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on the 200 test samples: {accuracy:.2f}%")
