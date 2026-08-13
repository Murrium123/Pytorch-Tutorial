import torch
import torch.optim as optim
import torch.nn as nn

# Common Loss Functions page 50
print(f"\nCommon Loss Functions\n")

# Example: Regression Loss
criterion_mse = nn.MSELoss()

prediction = torch.tensor([10.0]) # Model predicts 10
target = torch.tensor([15.0]) # Truth is 15

loss = criterion_mse(prediction, target)
print(f"MSE Loss: {loss.item()}") # (15-10)^2 = 25


print(f"\nThe Learning Rate\n")
# Assuming 'model' is a neural network we created
model = nn.Linear(10, 1)
print(f"model = {model}")

# Create an optimizer (Adam) and give it the model's parameters
optimizer = optim.Adam(model.parameters(), lr = 0.001)
print(f"optimizer = {optimizer}")

# 1. Clear the old gradients
optimizer.zero_grad()

# 2. Forward pass: get prediction
inputs = torch.randn(1, 10)
target = torch.tensor([[1.0]])
prediction = model(inputs)

# 3. Calculate Loss
loss = criterion_mse(prediction, target)

# 4. Backward pass: compute gradients
loss.backward()

# 5. Update weights
optimizer.step()

print(f"Loss for this step: {loss.item()}")

history = []

for epoch in range(5): # Run for 5 rounds
    optimizer.zero_grad()

    # Simulating training
    prediction = model(torch.randn(1, 10))
    loss = criterion_mse(prediction, torch.tensor([[1.0]]))

    loss.backward() # backpropogation
    optimizer.step()

    # Store the loss value (use .item() to get a standard Python float)
    history.append(loss.item())
    print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
