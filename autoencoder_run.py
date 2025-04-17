import os
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
image_file = '/home/byungsoo/Documents/comparison/Pictures_Matched/backup/ok.jpg'
# NG image
image_file = '/home/byungsoo/Documents/comparison/Pictures_Matched/backup/defect_05.jpg'
input_image = plt.imread(image_file)

plt.imshow(input_image, cmap='gray')

# Generate reconstructed image
input = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
output = autoencoder(input)
reconstructed_image = output.detach().numpy()
reconstructed_image = reconstructed_image.squeeze()

print("Input shape:", input.shape)
print("Output shape:", reconstructed_image.shape)

plt.imshow(reconstructed_image, cmap='gray')


## Calculate reconstruction error
# Mean Absolute Error (MAE): This calculates the average absolute difference between the input and output.

reconstruction_error = torch.mean(torch.abs(output - input))
print("Reconstruction Error (MAE):", reconstruction_error.item())

## Compute the difference
# difference_image = np.abs(input_image - reconstructed_image)
# difference_image = difference_image.squeeze()
# plt.imshow(difference_image, cmap='gray')

reconstructed_int = (reconstructed_image * 255).astype(int)

plt.imshow(input_image, cmap='gray')
plt.imshow(reconstructed_int, cmap='gray')

input_image_int = (input_image).astype(int)

image_a = cv2.subtract(input_image_int, reconstructed_int)
image_b = cv2.subtract(reconstructed_int, input_image_int)
difference = cv2.absdiff(image_a, image_b)

plt.imshow(image_a, cmap='magma')
plt.imshow(image_b, cmap='magma')
plt.imshow(difference, cmap='magma')


# input_image.astype(np.float32)
