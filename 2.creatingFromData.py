# 1. Creating from Data (page 22)
import torch

# Creating a 1D tensor (Vector) from a list
data = [1.0, 2.0, 3.0]
my_tensor = torch.tensor(data)

print(f"My tensor: {my_tensor}")

# Creating Specialized Tensors (page 22)

# Create a tensor of all zeros (shape: 2 rows, 3 columns)
zeros = torch.zeros(2, 3)
print(f"Zeros:\n{zeros}")

# Create a tensor of all ones
ones = torch.ones(3, 3)
print(f"Ones:\n{ones}")

# Create a tensor with random values between 0 and 1
# This is very common for initializing models
random_tensor = torch.rand(4, 4)
print(f"Random:\n{random_tensor}")

# Creates a tensor from 0 to 9
range_tensor = torch.arange(0, 10)
print(range_tensor)

# Tensor Data Types (23)

# Forcing an integer tensor to be a float
float_tensor = torch.tensor([1, 2, 3], dtype = torch.float32)
print(f"Tensor type for [1, 2, 3], dtype = torch.float32: {float_tensor.dtype}") # Output: torch.float32

# Creating a 2D tensor (Matrix)
matrix = torch.tensor([[[1, 2], 
                       [3, 4], 
                       [5, 6]]])

print(f"Numbers of dimensions (Rank): {matrix.ndim}")
print(f"Shape of tensor: {matrix.shape}")
print(f"Total number of elements: {matrix.numel()}")

# Basic Tensor Operations
# Element-wise Arithmatic

tensor_a = torch.tensor([1, 2, 3])
tensor_b = torch.tensor([4, 5, 6])

# Addition
print(f"Addition: {tensor_a + tensor_b}") # Output: tensor([5, 7, 9])

# Multiplication by a scalar
print(f"Multiplication by a scalar: {tensor_a * 10}") # Output: tensor([10, 20, 30])

tensor_x = torch.rand(3, 2) # A 3x2 matrix
tensor_y = torch.rand(2, 4) # A 2x4 matrix

# Matrix multiplication results in a 3x4 matrix
result = torch.matmul(tensor_x, tensor_y)
# Alternate syntax: 
resultA = tensor_x @ tensor_y

print(f"Result shape: {result.shape}")
print(f"Result: {result}\n ResultA: {resultA}")
