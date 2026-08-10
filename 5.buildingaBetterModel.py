import torch
import torch.nn as nn

# Building a better model
print(f"\nBuilding a better model\n")

class ImprovedNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # First layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function
        self.relu = nn.ReLU()
        # Output layer
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create a model with 10 inputs, 20 hidden units, and 1 output
my_model = ImprovedNet(10, 20, 1)
print(f"My model = {my_model}")

# Forward Propogation
print(f"\nForward Propogation\n")

# 1. Create some dummy input data (1 sample, 10 features)
dummy_input = torch.randn(1, 10)

# 2. Pass the data through the model
# Pytorch calls the forward() method under the hood
prediction = my_model(dummy_input)
print(f"Prediction: {prediction}")
print(f"Prediction shape: {prediction.shape}")

# Check how many parameters our model has
for name, param in my_model.named_parameters():
    print(f"Layer:  {name} | Size: {param.size()}")

# Sum total parameters
total_params = sum(p.numel() for p in my_model.parameters())
print(f"Total parameters in model: {total_params}")