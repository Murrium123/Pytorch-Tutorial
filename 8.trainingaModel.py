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

