import torch
from torch.utils.data import TensorDataset

print(f"\nUsing TensorDataset\n")

# Imagine we have 100 samples, each with 5 features
features = torch.randn(100, 5)

# And 100 corresponding Labels (0 or 1)
labels = torch.randint(0, 2, (100,))

# Wrap them into a single Dataset object
my_dataset = TensorDataset(features, labels)

# Now we can access an individual sample easily
sample_data, sample_label = my_dataset[4]
print(f"Features of sample 0: {sample_data}")
print(f"Label of sample 0: {sample_label}")

