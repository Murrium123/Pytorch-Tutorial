import torch
import torch.nn as nn
import torch.optim

# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

model = MyModel()

# 2. Define a file path (standard extensions are .pt or .pth)
PATH = "my_model_weights.pth"

# 3. Save the state_dict
torch.save(model.state_dict(), PATH)

print(f"Model weights saved to {PATH}")

# 1. Create the model structure again 
loaded_model = MyModel()

# 2. Load the weights from the file
# torch.load() reads the file into a dictionary
weights = torch.load(PATH, weights_only=True)

# 3. Inject the weights into the model
loaded_model.load_state_dict(weights)

# 4. CRITICAL STEP: Set to evaluation mode
loaded_model.eval()

print("Model weights loaded successfully!")

# Saving a checkpoint
checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.45
}
torch.save(checkpoint, "checkpoint_epoch_10.pth")

# --- Loading a checkpoint Later ---
checkpoint = torch.load("checkpoint_epoch_10.pth")

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(f"Resuming from epoch {epoch} with loss {loss}")
