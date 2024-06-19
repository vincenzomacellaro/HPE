import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from train_vae import VAE

skeleton = [
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11),
    (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)
]


def visualize_pose(ax, pose, skeleton, title):
    pose = pose[:51].reshape(17, 3)
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


def generate_new_pose(vae, num_samples=1, latent_dim=16):
    z = torch.randn(num_samples, latent_dim)  # Sample from standard normal distribution
    with torch.no_grad():
        new_poses = vae.decoder(z).numpy()  # Assuming the output of decoder is ready to use
    return new_poses


def gen_plot_name():
    return model_name[:-4]


if __name__ == '__main__':

    plot_path = "../plots/generated/"
    model_path = "../model/"
    model_name = "vae_hd256_ld16.pth"

    os.makedirs(plot_path, exist_ok=True)

    checkpoint = torch.load(model_path + model_name)
    input_dim = checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']
    hidden_dim = checkpoint['hidden_dim']

    vae = VAE(input_dim, hidden_dim, latent_dim)
    vae.load_state_dict(checkpoint['state_dict'])
    vae.eval()

    num_samples = 10
    generated_poses = generate_new_pose(vae, num_samples, latent_dim)

    fig = plt.figure(figsize=(20, 30))
    for i in range(num_samples):
        ax = fig.add_subplot(5, 2, i + 1, projection='3d')
        visualize_pose(ax, generated_poses[i], skeleton, f'Generated Pose {i + 1}')

    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.savefig(plot_path + gen_plot_name() + "_generated_poses.png")
    plt.show()




