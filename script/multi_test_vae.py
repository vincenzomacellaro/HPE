import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import var
from train_vae import VAE
from load_dataset import load_data

skeleton = [
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11),
    (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)
]


def generate_new_pose(vae, num_samples=1, latent_dim=16):
    z = torch.randn(num_samples, latent_dim)  # Sample from standard normal distribution
    with torch.no_grad():
        new_poses = vae.decoder(z).numpy()  # Assuming the output of decoder is ready to use
    return new_poses


def calculate_limb_lengths(pose, skeleton):
    limb_lengths = {}
    for (joint1, joint2) in skeleton:
        # Calculate the Euclidean distance between the two joints
        joint1_coords = pose[joint1 * 3:(joint1 + 1) * 3]  # slice out x, y, z coordinates for joint1
        joint2_coords = pose[joint2 * 3:(joint2 + 1) * 3]  # slice out x, y, z coordinates for joint2
        distance = np.linalg.norm(joint1_coords - joint2_coords)
        limb_lengths[(joint1, joint2)] = distance
    return limb_lengths


def process_generated_poses(generated_poses, skeleton):
    all_limb_lengths = {limb: [] for limb in skeleton}

    # Assume generated_poses is a (n_samples, n_features) array
    for pose in generated_poses:
        limb_lengths = calculate_limb_lengths(pose, skeleton)
        for limb, length in limb_lengths.items():
            all_limb_lengths[limb].append(length)

    # Calculate average lengths for each limb
    average_limb_lengths = {limb: np.mean(lengths) for limb, lengths in all_limb_lengths.items()}
    return average_limb_lengths


def calculate_euclidean_distance(limb_lengths1, limb_lengths2):
    distances = {}
    for limb in limb_lengths1:
        if limb in limb_lengths2:
            # Compute the Euclidean distance for each limb
            distances[limb] = np.abs(limb_lengths1[limb] - limb_lengths2[limb])
    return distances


def aggregate_distances(distances):
    """ Aggregate distances across limbs to a single score. """
    total_distance = np.sum(list(distances.values()))
    mean_distance = np.mean(list(distances.values()))
    return total_distance, mean_distance


if __name__ == '__main__':

    num_samples = 10000

    plot_path = "../plots/generated/"
    model_path = "../model/"
    # model_name = "vae_hd64_ld16_gamma0001.pth"

    # get ground_truth from the test set
    test_samples, test_ml_len, train_sl_len = load_data('train')
    avg_test_limb_lengths = process_generated_poses(test_samples, skeleton)
    print(f"[test_set] avg. limb_lengths: {avg_test_limb_lengths}")

    for entry in os.listdir(model_path):
        model = os.path.join(model_path, entry)
        print(model)
        if entry == "vae_hd256_ld16.pth":
            checkpoint = torch.load(model)
            input_dim = checkpoint['input_dim']
            latent_dim = checkpoint['latent_dim']
            hidden_dim = checkpoint['hidden_dim']

            vae = var.VAE(input_dim, hidden_dim, latent_dim)
            vae.load_state_dict(checkpoint['state_dict'])
            vae.eval()

            generated_poses = generate_new_pose(vae, num_samples, latent_dim)
            avg_gen_limb_lengths = process_generated_poses(generated_poses, skeleton)

            print(f"\t [gen_set] avg. limb_lengths: {avg_gen_limb_lengths}")

            # Calculate distances for each limb
            limb_distances = calculate_euclidean_distance(avg_test_limb_lengths, avg_gen_limb_lengths)

            # Aggregate these distances to a single score
            total_distance, mean_distance = aggregate_distances(limb_distances)

            print(f"\t Total Distance across all limbs: {total_distance}")
            print(f"\t Mean Distance per limb: {mean_distance}")

        elif os.path.isfile(model) and not entry.startswith('.'):
            checkpoint = torch.load(model)
            input_dim = checkpoint['input_dim']
            latent_dim = checkpoint['latent_dim']
            hidden_dim = checkpoint['hidden_dim']

            vae = VAE(input_dim, hidden_dim, latent_dim)
            vae.load_state_dict(checkpoint['state_dict'])
            vae.eval()

            generated_poses = generate_new_pose(vae, num_samples, latent_dim)
            avg_gen_limb_lengths = process_generated_poses(generated_poses, skeleton)

            print(f"\t [gen_set] avg. limb_lengths: {avg_gen_limb_lengths}")

            # Calculate distances for each limb
            limb_distances = calculate_euclidean_distance(avg_test_limb_lengths, avg_gen_limb_lengths)

            # Aggregate these distances to a single score
            total_distance, mean_distance = aggregate_distances(limb_distances)

            print(f"\t Total Distance across all limbs: {total_distance}")
            print(f"\t Mean Distance per limb: {mean_distance}")
