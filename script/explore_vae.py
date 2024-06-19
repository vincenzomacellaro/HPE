import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from train_vae import VAE, CustomDataset, load_data  # Ensure the VAE class and dataset functions are in train_vae.py

plot_path = "../plots/generated/"
model_path = "../model/"
model_name = "vae_ld32.pth"


def visualize_latent_space(vae, data_loader, num_batches=100):
    vae.eval()
    with torch.no_grad():
        z_list = []
        for i, data in enumerate(data_loader):
            if i >= num_batches:
                break
            x = data
            mu, logvar = vae.encoder(x)
            z = vae.reparameterize(mu, logvar)
            z_list.append(z.cpu().numpy())
        z_array = np.concatenate(z_list, axis=0)
        plt.figure(figsize=(8, 6))
        plt.scatter(z_array[:, 0], z_array[:, 1], alpha=0.5)
        plt.title('Latent Space Distribution')
        plt.xlabel('z1')
        plt.ylabel('z2')
        plt.show()


def check_decoder_output(vae, latent_dim, num_samples=10):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim)
        generated_poses = vae.decoder(z)
        for i, pose in enumerate(generated_poses):
            print(f"Generated Pose {i+1}: {pose.numpy()}")


checkpoint = torch.load(model_path + model_name)
input_dim = checkpoint['input_dim']
latent_dim = checkpoint['latent_dim']
hidden_dim = checkpoint['hidden_dim']

vae = VAE(input_dim, hidden_dim, latent_dim)
vae.load_state_dict(checkpoint['state_dict'])

# Assuming `test_loader` is your DataLoader for the test set

# Load test data
# train_samples = load_data('train')
# train_dataset = CustomDataset(np.asarray(train_samples))
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# visualize_latent_space(vae, train_loader)
# check_decoder_output(vae, latent_dim)

# Sample different latent vectors
latent_vectors = [torch.randn(1, latent_dim) for _ in range(3)]

# Decode the poses
decoded_poses = [vae.decoder(z).detach().numpy() for z in latent_vectors]

# Print the latent vectors and corresponding decoded poses
for i, (z, pose) in enumerate(zip(latent_vectors, decoded_poses)):
    print(f"Latent Vector {i+1}: {z.numpy()}")
    print(f"Decoded Pose {i+1}: {pose}")

# Check for diversity in poses
poses_diff = [np.linalg.norm(decoded_poses[i] - decoded_poses[j])
              for i in range(len(decoded_poses)) for j in range(i+1, len(decoded_poses))]
print("Pose Differences:", poses_diff)

# # If differences are very small, try manual latent vector exploration
z_random = torch.randn(1, latent_dim)
pose_random = vae.decoder(z_random).detach().numpy()
print("Random Latent Vector:", z_random.numpy())
print("Decoded Random Pose:", pose_random)