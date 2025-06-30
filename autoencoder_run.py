import util
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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

# Reload the model
autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load("autoencoder.pth"))
print("Model reloaded!")


## Verify model reload
# OK image
image_file = '/home/byungsoo/Documents/comparison/Pictures_Matched/grayscale_WIN_20250417_11_56_29_Pro.jpg'
# NG image
image_file = '/home/byungsoo/Documents/comparison/numbers/9935_rot4.jpg'
input_image = plt.imread(image_file)
plt.imshow(input_image, cmap='gray')

# Generate reconstructed image
def reconstruct_image(input_image, autoencoder):
    input = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    output = autoencoder(input)
    reconstructed_image = output.detach().numpy()
    reconstructed_image = reconstructed_image.squeeze()
    reconstruction_error = torch.mean(torch.abs(output - input))
    return reconstructed_image, reconstruction_error

reconstructed_image, reconstruction_error = reconstruct_image(input_image, autoencoder)

plt.imshow(reconstructed_image, cmap='gray')

## Calculate reconstruction error
print("Reconstruction Error (MAE):", reconstruction_error.item())

difference = util.check_difference(input_image, reconstructed_image)

# Save the image before displaying it
plt.imshow(difference, cmap='magma')


# input_image.astype(np.float32)
