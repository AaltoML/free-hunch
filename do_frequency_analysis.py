imagenet_data_dir = "data/imagenet"
# the format of the data dir is that it is a directory with subdirectories, each subdirectory contains .png files

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch_dct import dct_2d
from tqdm import tqdm

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a consistent size
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Scale to [-1, 1]
])

# Set the number of images to process
num_images_to_process = 500

# Load the dataset
dataset = ImageFolder(imagenet_data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize variables for accumulating DCT coefficients
dct_sum = torch.zeros((3, 256, 256))
dct_sum_sq = torch.zeros((3, 256, 256))
total_images = 0

# Process the specified number of images
for images, _ in tqdm(dataloader, desc="Processing images"):
    batch_size = images.size(0)
    
    # If this batch would exceed the desired number of images, only take what we need
    if total_images + batch_size > num_images_to_process:
        images = images[:num_images_to_process - total_images]
        batch_size = num_images_to_process - total_images

    # Compute DCT for each image in the batch
    dct_coeffs = dct_2d(images, norm='ortho')
    
    # Update sums
    dct_sum += dct_coeffs.sum(dim=0)
    dct_sum_sq += (dct_coeffs ** 2).sum(dim=0)

    total_images += batch_size

    if total_images >= num_images_to_process:
        break

# Calculate mean and variance
dct_mean = dct_sum / total_images
dct_variance = (dct_sum_sq / total_images) - (dct_mean ** 2)

print(f"Total images processed: {total_images}")
print(f"DCT coefficient variance shape: {dct_variance.shape}")

# Optionally, visualize or save the results
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(torch.log(dct_variance[i]), cmap='viridis')
    plt.title(f'Log Variance of DCT Coefficients (Channel {i})')
    plt.colorbar()
plt.tight_layout()
plt.show()

# Save the variance tensor for future use
import os
torch.save(dct_variance, os.path.join(imagenet_data_dir, 'dct_variance.pt'))