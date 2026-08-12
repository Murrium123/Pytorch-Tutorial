import torch
import torch.nn as nn

# Common Loss Functions page 50
print(f"\nCommon Loss Functions\n")

# Example: Regression Loss
criterion_mse = nn.MSELoss()

prediction = torch.tensor([10.0]) # Model predicts 10
target = torch.tensor([14.0]) # Truth is 14

loss = criterion_mse(prediction, target)
print(f"MSE Loss: {loss.item()}") # (15-10)^2 = 25
