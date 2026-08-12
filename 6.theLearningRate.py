import torch.optim as optim
import torch.nn as nn

print(f"\nThe Learning Rate\n")
# Assuming 'model' is a neural network we created
model = nn.Linear(10, 1)
print(f"model = {model}")

# Create an optimizer (Adam) and give it the model's parameters
optimizer = optim.Adam(model.parameters(), lr = 0.001)
print(f"optimizer = {optimizer}")
