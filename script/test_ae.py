import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from train_ae import Autoencoder
# from load_dataset import process

from load_dataset import load_data

plot_path = "../plots/reconstructed/"

skeleton = [
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11),
    (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)
]


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        values = torch.tensor([v for v in sample.values()], dtype=torch.float32)
        return values


# Save path
model_path = "../model/"
model_name = "model_ld32.pth"

# Load configuration
checkpoint = torch.load(model_path + model_name)
input_dim = checkpoint['input_dim']
latent_dim = checkpoint['latent_dim']

# Define and load the model
autoencoder = Autoencoder(input_dim, latent_dim)
autoencoder.load_state_dict(checkpoint['state_dict'])

# Load data from our dataset
test_samples = load_data('test')

random_samples = random.sample(test_samples, 30)
random_dataset = CustomDataset(random_samples)

batch_size = 10
dataloader = DataLoader(random_dataset, batch_size, shuffle=True)


# Visualize the results
def visualize_pose(ax, pose, skeleton, title):
    x = pose[:, 0]
    y = pose[:, 1]
    z = pose[:, 2]
    ax.scatter(x, y, z)

    for (i, j) in skeleton:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, pad=10, fontsize=15)


# Measure the difference between original and reconstructed poses
def calculate_metrics(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    return mse, mae


sample_batch = next(iter(dataloader))
original_poses = sample_batch[:10]

with torch.no_grad():
    reconstructed_poses = autoencoder(original_poses).numpy()

# # Calculate and print metrics
# for i in range(10):
#     original_pose = original_poses[i].numpy()
#     reconstructed_pose = reconstructed_poses[i]
#     mse, mae = calculate_metrics(original_pose, reconstructed_pose)
#     print(f'Pose {i + 1} - MSE: {mse:.6f}, MAE: {mae:.6f}')

fig = plt.figure(figsize=(20, 30))
for i in range(10):
    ax = fig.add_subplot(5, 4, 2 * i + 1, projection='3d')
    visualize_pose(ax, original_poses[i], skeleton, f'Original Pose {i + 1}')

    ax = fig.add_subplot(5, 4, 2 * i + 2, projection='3d')
    visualize_pose(ax, reconstructed_poses[i], skeleton, f'Reconstructed Pose {i + 1}')

    # Calculate MSE and MAE
    original_pose = original_poses[i].numpy()
    reconstructed_pose = reconstructed_poses[i]
    mse, mae = calculate_metrics(original_pose, reconstructed_pose)

    # Add text annotation for MSE and MAE
    text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}'
    ax.text(0.05, -0.65, 0, text, transform=ax.transAxes, fontsize=10, verticalalignment='top')

plt.subplots_adjust(wspace=0.3, hspace=0.5)
plt.savefig(plot_path + 'rec_ld' + str(latent_dim) + '_1.png')
