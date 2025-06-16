import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image

# Check for GPU availability and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# Define a dataset
class Dataset(Dataset):
    """
    A PyTorch Dataset class to load grayscale images from a directory.
    """
    def __init__(self, img_dir='images', size=(640, 640), transform=None):
        self.img_dir = img_dir
        self.size = size
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])

        self.image_filenames = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('L')
        return self.transform(image)

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize dataset and dataloader
# dataset = Dataset(img_dir='Pictures_Grayscale', size=(1280, 720))
dataset = Dataset(img_dir='numbers', size=(640, 640))
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, pin_memory=True if device.type == 'cuda' else False)

# Initialize model, optimizer, and loss function
autoencoder = Autoencoder().to(device)  # Move model to GPU
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
epoch_losses = []
num_epochs = 20

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in dataloader:
        # Move batch to GPU
        batch = batch.to(device)
        
        optimizer.zero_grad()
        reconstructed = autoencoder(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    # Optional: Print GPU memory usage
    if device.type == 'cuda':
        print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")

print("Training finished.")

# **Visualize Training Loss**
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', color='b')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()

# Function to visualize original vs. reconstructed images
def visualize_results(model, dataloader, num_images=5):
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        # Move batch to GPU for inference
        batch = batch.to(device)
        reconstructed = model(batch)
        
        # Move tensors back to CPU for visualization
        batch_cpu = batch.cpu()
        reconstructed_cpu = reconstructed.cpu()
        
        fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 2))
        for i in range(num_images):
            axes[i, 0].imshow(batch_cpu[i].squeeze().numpy(), cmap="gray")
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(reconstructed_cpu[i].squeeze().numpy(), cmap="gray")
            axes[i, 1].set_title("Reconstructed")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()

# Call the function
visualize_results(autoencoder, dataloader)

# Save the model
torch.save(autoencoder.state_dict(), "autoencoder.pth")
print("Model saved!")

# Clean up GPU memory
if device.type == 'cuda':
    torch.cuda.empty_cache()
    print("GPU memory cleared.")