import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

# Define a dataset
class Dataset(Dataset):
    """
    A PyTorch Dataset class to load grayscale images from a directory.
    """
    def __init__(self, img_dir='Pictures_Grayscale', transform=None):
        self.img_dir = img_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((1280, 720)),  # Resize images
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
dataset = Dataset(img_dir='Pictures_Grayscale')
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Initialize model, optimizer, and loss function
autoencoder = Autoencoder()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
epoch_losses = []

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        reconstructed = autoencoder(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_losses.append(epoch_loss)  # Save loss per epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

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
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient tracking
        batch = next(iter(dataloader))  # Get a batch of images
        reconstructed = model(batch)  # Pass images through the autoencoder

        fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 2))
        for i in range(num_images):
            axes[i, 0].imshow(batch[i].squeeze().cpu().numpy(), cmap="gray")
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(reconstructed[i].squeeze().cpu().numpy(), cmap="gray")
            axes[i, 1].set_title("Reconstructed")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()

# Call the function
visualize_results(autoencoder, dataloader)

# Save the model
torch.save(autoencoder.state_dict(), "autoencoder.pth")
print("Model saved!")

# Reload the model
autoencoder_reload = Autoencoder()
autoencoder_reload.load_state_dict(torch.load("autoencoder.pth"))
print("Model reloaded!")

# Verify model reload
image_file = '/home/byungsoo/Documents/comparison/Pictures_Grayscale/backup/grayscale_WIN_20250414_12_50_42_Pro.jpg'
image = plt.imread(image_file)

plt.imshow(image, cmap='gray')
plt.title("Input Image")
plt.axis('off')
plt.show()

input_image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
output_image = autoencoder_reload(input_image)
reconstructed_image = output_image.detach().numpy()
reconstructed_image = reconstructed_image.squeeze()


print("Input shape:", input_image.shape)
print("Output shape:", reconstructed_image.shape)

plt.imshow(reconstructed_image, cmap='gray')
plt.title("Reconstructed Image")
plt.axis('off')
plt.show()



# Calculate reconstruction error
# Mean Absolute Error (MAE): This calculates the average absolute difference between the input and output.

reconstruction_error = torch.mean(torch.abs(output_image - input_image))
print("Reconstruction Error (MAE):", reconstruction_error.item())

# Compute the difference
difference_image = np.abs(input_image - reconstructed_image)
difference_image = difference_image.squeeze()

plt.imshow(difference_image, cmap='magma')
plt.title("Difference Image")
plt.axis('off')
plt.show()
