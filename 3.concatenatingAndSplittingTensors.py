import torch
import numpy as np

tensor_a = torch.tensor([[1, 2],[3, 4]])
tensor_b = torch.tensor([[5, 6],[7, 8]])

# Concatenating and Splitting Tensors
print(f"\nConcatenating and Splitting Tensors\n")

# Join them vertically (along rows, dim=0)
row_cat = torch.cat([tensor_a, tensor_b], dim=0)
print(f"Row concatenation: \n {row_cat}")
print(f"Vertical Cat Shape: {row_cat.shape}") # (4, 2)

# Join them horizontally (along columns, dim=1)
col_cat = torch.cat([tensor_a, tensor_b], dim=1)
print(f"Column concatenation: \n {col_cat}")
print(f"Horizontal Cat Shape: {col_cat.shape}") # (2, 4)

# Stacking (torch.stack)
# Stacking creates a new dimension (3D)
stacked = torch.stack([tensor_a,tensor_b])
print(f"Stacked 2 dimension(staked):\n{stacked}")
print(f"Stacked shape: {stacked.shape}") # (2, 2, 2)
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
stacked2=torch.stack([a,b])
print(f"Stacked 1 dimension(stacked2): \n {stacked2}")
stacked3 = torch.stack([tensor_b,tensor_a])
print(f"Stacked 2 dimension reversed(stacked3): \n {stacked3}")
# Why is there a gap between the two dimensions? 

# Broadcasting
print(f"\nBroadcasting \n")

# A 2x3 matrix
matrix = torch.ones(2,3)
print(f"Matrix shape: {matrix.shape}")

# A 1x3 vector
vector = torch.tensor([1, 2, 3])
print(f"Vector shape: {vector.shape}")

# Even though shapes are different, PyTorch 'stretches' the vector
# to act like a 2x3 matrix filled with [1, 2, 3]
result = matrix + vector
print(f"Resulting matrix: \n{result}")

# Tensor to NumPy (page 32)
print(f"\nTensor to NumPy\n")
t = torch.ones(5)
n = t.numpy()
print(f"NumPy array: {n}")

# NumPy to Tensor
f = np.array([1, 2, 3])
t = torch.from_numpy(f)
print(f"Tensor from NumPy: {t}")