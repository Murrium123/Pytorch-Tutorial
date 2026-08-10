import torch
import torch.nn as nn

# Creating Custom Neural Networks
print(f"\nCreating Custom Neural Networks\n")

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Define layers here
        self.layer1 = nn.Linear(in_features=10, out_features=5) # nn.Linear uses the equation y = xW^T + b (W = Weights // b = bias)
        self.layer2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        # Define the data flow here
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Instantiate the model
model = SimpleNet()
print(model)
