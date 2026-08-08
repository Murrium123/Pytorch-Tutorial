# Page 18 - Verifying the installation
import torch

# Check if Pytorch is installed
print(f"PyTorch Version: {torch.__version__}")

# Check if a GPU is available for use
cuda_available = torch.cuda.is_available()
print(f"Is a GPU available? {cuda_available}")

# Create a small piece of data (a tensor) to test
test_data  = torch.tensor([1.0, 2.0, 3.0])
print(f"Our first tensor: {test_data}")