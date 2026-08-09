import torch 

# Chapter 3: Tensor Manipulation
# Create a 1D tensor with 12 elements
x = torch.arange(12)
print(f"Original x: {x}")
print(f"Original shape: {x.shape}")

# Reshape into 3 rows and 4 columns
reshaped_x = x.view(3, 4)
print(f"Reshaped x:\n{reshaped_x}")

# Using -1 as a 'wildcard'
# PyTorch will calculate the correct dimension automatically
auto_reshaped = x.view(2, -1)
print(f"Reshaped: {auto_reshaped}")
print(f"Shape with wildcard: {auto_reshaped.shape}") # Results in (2, 6)

# Indexing and Slicing
# Create a 2D tensor (3 rows, 3 columns)
data = torch.tensor([[10, 20, 30],
                     [40, 50, 60],
                     [70, 80, 90]])

# Get a single value (row 0, column 1)
print(f"Value at 0,1: {data[0, 1]}")

# Get all values in the first row
print(f"First row: {data[0, :]}")

# Get all values in the second column
print(f"Second column: {data[:, 1]}")

# Slice: Get the first 2 rows and the last 2 columns
print(f"Slice: \n{data[:2,1:]}")