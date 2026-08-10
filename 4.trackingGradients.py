import torch

# Tracking Gradients
print(f"\nTracking Gradients \n")
# Create a tensor that we want to track
x = torch.tensor([2.0], requires_grad=True)
print(f"Tensor x: {x}")
print(f"What does x.requires_grad print currently: {x.requires_grad}")

# Perform a simple operation
y = x * x
print(f"Tensor y: {y}")

# 'y' was created from an operation involving 'x', so PyTorch 
# automatically gives it a gradient function.
print(f"y.grad_fn: {y.grad_fn}")

# Computing Gradients
print(f"\nComputing Gradients\n")
# 'y' is our final output. Let's compute the gradients.
y.backward()

# The computed gradient is stored in the .grad attribute of the input tensor 'x'
print(f"Gradient of y with respect to x: {x.grad}")

# Create input tensors. We must set requires_grad=True for both.
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([5.0], requires_grad=True)
print(f"a:{a} // b:{b}")
# Define the computation
z = 3 * a**2 + b
print(f"z = {z}")
# Compute gradients
z.backward()
# The gradient dz/da is 6*a = 12
print(f"Gradient with respect to a: {a.grad}")
# The gradient dz/db is 1
print(f"Gradient with respect to b: {b.grad}")

# Disabling Gradient Tracking
print(f"\nDisabling Gradient Tracking\n")
x = torch.tensor([2.0], requires_grad=True)
print(f"Gradient tracking is ON for x: {x.requires_grad}")
with torch.no_grad():
    print("--- Inside no_grad() block ---")
    y = x * 2
    print(f"Gradient tracking is ON for y: {y.requires_grad}")
    print(f"Does y have a grad_fn? {y.grad_fn is None}")

print("--- Outside no_grad() block ---")
z = x * 2
print(f"Gradient tracking is ON for z: {z.requires_grad}")

# Common Autograd workflows
print(f"\nCommon Autograd Workflows\n")
# Model parameters (e.g., weights), which need to learn
w = torch.tensor([0.5], requires_grad=True)

# An input value 
x = torch.tensor([2.0])

# True target value 
y_true = torch.tensor([3.0])

# --- Start of a typical training step ---

# 1. Make a prediction (Forward pass)
y_pred = w * x

# 2. Calculate the error (loss)
loss = (y_pred - y_true)**2

# 3. Compute gradients for all parameters with requires_grad=True 
loss.backward()

# 4. Check the calculated gradient for 'w'
print(f"Gradient for w: {w.grad}")

# 5. In a real scenario, you would update 'w' using this gradient.
# For now, Let's just clear the gradient for the next step.
w.grad.zero_()

print(f"Gradient after zeroing: {w.grad}")