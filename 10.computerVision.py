import torch 
from torchvision import datasets, transforms

print(f"\nWorking with Image Datasets (page 74)\n")

# Define a basic transformation
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the training data
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transform
)

print(f"Number of training images: {len(train_data)}")
print(f"Image shape: {train_data[0][0].shape}") #[1,28,28]

print(f"\nImage Transformations\n")

# A common transformation pipeline
my_transforms = transforms.Compose([
    transforms.Resize((28,28)), # Ensure size is consistent
    transforms.ToTensor(), # Convert to tensor & scale to [0,1]
    transforms.Normalize((0.5,),(0.5,)) # Shift range to [-1,1]
])

print(f"/nBuilding an Image Classifier/n")

import torch.nn as nn

class SimpleImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 784 inputs (pixels), 128 hidden neurons, 10 outputs (digits 0-9)
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        return self.main(x)

model = SimpleImageClassifier()

print(f"\nComplete Workflow Example\n")

from torch.utils.data import DataLoader

# 1. Setup Data
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 2. Get one batch of images
images, labels = next(iter(train_loader))

# 3. Pass through model
outputs = model(images)

# 4. Get predictions
_, predicted = torch.max(outputs, 1)

print(f"Predicted digits: {predicted}")
print(f"Actual digits:   {labels}")


