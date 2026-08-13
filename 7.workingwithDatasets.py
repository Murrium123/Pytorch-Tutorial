import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as DL

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

print(f"\nBatch Processing\n")
# Create a DataLoader 
# We want to process 32 samples at a time
batch_size = 32

my_dataloader = DL(
    dataset=my_dataset,
    batch_size=batch_size,
    shuffle=True
)

# A DataLoader is an 'iterable'. We can loop through it. 
for batch_idx, (batch_features, batch_labels) in enumerate(my_dataloader):
    print(f"Batch: {batch_idx + 1}")
    print(f"Features shape: {batch_features.shape}") # Should be [32, 5]
    print(f"Labels shape: {batch_labels.shape}") # Should be [32]

    # In a real scenario, we would put model(batch_features) here 
    if batch_idx == 0: break # Just showing the first batch

print(f"\nShuffling and Loading Data Efficiently\n")

# Efficient DataLoader setup
train_loader = DL(
    dataset=my_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2 # Uses 2 background processes to Load data
)
print(f"train_loader = {train_loader}")

# This crashed, but I cannot figure out why. 
#for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
    #print(f"Batch: {batch_idx + 1}")
    #print(f"Features shape: {batch_features.shape}") # Should be [32, 5]
    #print(f"Labels shape: {batch_labels.shape}") # Should be [32]

    # In a real scenario, we would put model(batch_features) here 
    #if batch_idx == 0: break # Just showing the first batch

