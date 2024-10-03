import random
import torch
import json
import numpy as np

from train_2b_vae import VAE
from script_angles.plot_utils import plot_pose_from_joint_angles
from script_angles.human36_to_angles import reconstruct_from_array
from script_angles.load_data_utils import load_data_for_train, de_normalize_sample


def reconstruct_sample(pose_sample, physique_sample, scale_factors):
    combined_sample = np.concatenate((pose_sample, physique_sample), axis=0)
    sample_reconstructed = reconstruct_from_array(combined_sample)
    sample_denorm = de_normalize_sample(sample_reconstructed, scale_factors)
    return sample_denorm


def reconstruction_test(vae, num_samples=3):
    pose_test_data, physique_test_data = load_data_for_train("test")
    # test_data loaded via [load_data_for_train] function, which performs all the processing steps
    # we need to perform the [reverse processing] of the data before plotting
    test_samples = len(pose_test_data)

    for i in range(num_samples):
        random_index = random.randint(0, test_samples - 1)
        pose_sample = pose_test_data[random_index]
        physique_sample = physique_test_data[random_index]

        pose_tensor = torch.tensor(pose_sample, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        physique_tensor = torch.tensor(physique_sample, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Perform inference using the VAE model
        pose_recon, physique_recon, pose_mu, pose_log_var, physique_mu, physique_log_var = vae(pose_tensor,
                                                                                               physique_tensor)

        reconstructed_pose_np = pose_recon.detach().cpu().numpy()
        reconstructed_physique_np = physique_recon.detach().cpu().numpy()

        # [pose_sample, physique_sample]
        original_sample = reconstruct_sample(pose_sample, physique_sample, scale_factors)
        decoded_sample = reconstruct_sample(reconstructed_pose_np[0], reconstructed_physique_np[0], scale_factors)

        print(f"original_sample: \n{original_sample}")
        print(f"decoded_sample: \n{decoded_sample}")

        plot_pose_from_joint_angles(original_sample, "[ORIGINAL] sample")
        plot_pose_from_joint_angles(decoded_sample, "[DECODED] sample")

    return


def generation_test(vae, num_samples):
    # Load test data
    pose_test_data, physique_test_data = load_data_for_train("test")

    random_index = random.randint(0, len(pose_test_data) - 1)
    pose_sample = pose_test_data[random_index]
    physique_sample = physique_test_data[random_index]

    # Reconstruct the original sample for comparison
    original_sample = reconstruct_sample(pose_sample, physique_sample, scale_factors)
    plot_pose_from_joint_angles(original_sample, "[ORIGINAL] sample")

    # Convert the samples to tensors
    pose_tensor = torch.tensor(pose_sample, dtype=torch.float32).unsqueeze(0)
    physique_tensor = torch.tensor(physique_sample, dtype=torch.float32).unsqueeze(0)

    # Encode to latent space
    pose_encoded, physique_encoded, pose_mu, pose_log_var, physique_mu, physique_log_var = vae(pose_tensor, physique_tensor)

    # Generate multiple samples by reparameterizing the latent space with noise
    gen_pose_samples = torch.stack([vae.reparameterize(pose_mu, pose_log_var) for _ in range(num_samples)])
    gen_physique_samples = torch.stack([vae.reparameterize(physique_mu, physique_log_var) for _ in range(num_samples)])

    # Ensure that the input to the decoder is correctly reshaped to match the expected batch size
    gen_pose_samples = gen_pose_samples.view(num_samples, -1)
    gen_physique_samples = gen_physique_samples.view(num_samples, -1)

    # Decode the generated samples
    with torch.no_grad():
        gen_poses = vae.pose_decoder(gen_pose_samples).detach().cpu().numpy()
        gen_physiques = vae.physique_decoder(gen_physique_samples).detach().cpu().numpy()

    # Reconstruct and plot the generated samples
    for idx, (gen_pose, gen_physique) in enumerate(zip(gen_poses, gen_physiques)):
        rec_sample = reconstruct_sample(gen_pose, gen_physique, scale_factors)
        plot_pose_from_joint_angles(rec_sample, f"[GENERATED] sample - {idx + 1}")

    return


def sample_from_latent_space(vae, num_samples, latent_dim, device='cpu'):
    # Sample random noise (z) from the standard normal distribution
    z_pose = torch.randn(num_samples, latent_dim).to(device)
    z_physique = torch.randn(num_samples, latent_dim).to(device)

    # Reshape to match the input format expected by the decoders
    z_pose = z_pose.view(num_samples, -1)
    z_physique = z_physique.view(num_samples, -1)

    # Pass the noise through the decoder to generate new data
    with torch.no_grad():
        generated_poses = vae.pose_decoder(z_pose).cpu().numpy()
        generated_physiques = vae.physique_decoder(z_physique).cpu().numpy()

    return generated_poses, generated_physiques


def pure_generation_test(vae, num_samples, latent_dim):
    # Generate samples from the latent space
    gen_poses, gen_physiques = sample_from_latent_space(vae, num_samples, latent_dim)

    # Reconstruct and plot the generated samples
    for idx, (gen_pose, gen_physique) in enumerate(zip(gen_poses, gen_physiques)):
        rec_sample = reconstruct_sample(gen_pose, gen_physique, scale_factors)
        plot_pose_from_joint_angles(rec_sample, f"3D pose from [NOISE] - {idx + 1}")

    return


def fixed_pose_vary_physique_test(vae, pose_sample, num_samples):
    # Fix pose data and generate different physiques for it;
    # If pose_sample is None, a random noise pose will be generated.

    # Load the test data to get the correct size for physique data
    pose_test_data, physique_test_data = load_data_for_train("test")

    if pose_sample is None:
        latent_dim = vae.pose_mu.out_features  # Assuming latent_dim is the output size of pose_mu
        z_pose = torch.randn(1, latent_dim)  # Generate random noise for the pose
        pose_sample = vae.pose_decoder(z_pose).detach().cpu().numpy()[0]
        original_sample = reconstruct_sample(pose_sample, np.zeros_like(pose_sample), scale_factors)
    else:
        random_index = random.randint(0, len(pose_test_data) - 1)
        pose_sample = pose_test_data[random_index]
        physique_sample = physique_test_data[random_index]
        original_sample = reconstruct_sample(pose_sample, np.zeros_like(physique_sample), scale_factors)

    print(f"FIXED POSE DATA: [joint_positions][joint_angles]")
    print(f"joint_positions: {original_sample['joint_positions']}")
    print(f"joint_angles: {original_sample['joint_angles']}")

    # Encode the pose to latent space (mu and log_var)
    pose_tensor = torch.tensor(pose_sample, dtype=torch.float32).unsqueeze(0)

    # Generate a zero physique tensor with the correct shape
    physique_shape = physique_test_data[0].shape  # Retrieve shape of the physique data
    zero_physique_tensor = torch.zeros(1, *physique_shape, dtype=torch.float32)

    _, _, pose_mu, pose_log_var, _, _ = vae(pose_tensor, zero_physique_tensor)  # Physique not used here

    # Generate random physiques while keeping pose fixed
    latent_dim = vae.physique_mu.out_features  # Assuming latent_dim is the output size of physique_mu
    for i in range(num_samples):
        z_physique = torch.randn(1, latent_dim)  # Sample random physiques

        with torch.no_grad():
            gen_physique = vae.physique_decoder(z_physique).detach().cpu().numpy()[0]

        # Reconstruct the sample with the fixed pose and new physique
        rec_sample = reconstruct_sample(pose_sample, gen_physique, scale_factors)
        # print(f"rec_sample {i + 1}: \n{rec_sample['bone_lengths']}")
        print(f"REC SAMPLE {i + 1}: "
              f"\nbone_lengths: {rec_sample['bone_lengths']}"
              f"\nbase_skeleton: {rec_sample['base_skeleton']}"
              f"\njoint_positions [FIXED]: {rec_sample['joint_positions']}"
              f"\njoint_angles [FIXED]: {rec_sample['joint_angles']}")
        plot_pose_from_joint_angles(rec_sample, f"[VARIED] Physique {i + 1} with Fixed Pose")

    return


def fixed_physique_vary_pose_test(vae, physique_sample, num_samples):
    # Fix physique data and generate different poses for it;
    # If physique_sample is None, a random noise physique will be generated.

    # Load the test data to get the correct size for pose data
    pose_test_data, physique_test_data = load_data_for_train("test")

    if physique_sample is None:
        latent_dim = vae.physique_mu.out_features  # Assuming latent_dim is the output size of physique_mu
        z_physique = torch.randn(1, latent_dim)  # Generate random noise for the physique
        physique_sample = vae.physique_decoder(z_physique).detach().cpu().numpy()[0]
        original_sample = reconstruct_sample(np.zeros_like(physique_sample), physique_sample, scale_factors)
    else:
        random_index = random.randint(0, len(physique_test_data) - 1)
        pose_sample = pose_test_data[random_index]
        physique_sample = physique_test_data[random_index]
        original_sample = reconstruct_sample(np.zeros_like(pose_sample), physique_sample, scale_factors)

    print(f"FIXED PHYSIQUE DATA: [bone_lenghts][base_skeleton]")
    print(f"bone_lengths: {original_sample['bone_lengths']}")
    print(f"base_skeleton: {original_sample['base_skeleton']}")

    # Encode the physique to latent space (mu and log_var)
    physique_tensor = torch.tensor(physique_sample, dtype=torch.float32).unsqueeze(0)

    # Generate a zero pose tensor with the correct shape
    pose_shape = pose_test_data[0].shape  # Retrieve shape of the pose data
    zero_pose_tensor = torch.zeros(1, *pose_shape, dtype=torch.float32)

    _, _, _, _, physique_mu, physique_log_var = vae(zero_pose_tensor, physique_tensor)  # Pose not used here

    # Generate random poses while keeping physique fixed
    latent_dim = vae.pose_mu.out_features  # Assuming latent_dim is the output size of pose_mu
    for i in range(num_samples):
        z_pose = torch.randn(1, latent_dim)  # Sample random poses

        with torch.no_grad():
            gen_pose = vae.pose_decoder(z_pose).detach().cpu().numpy()[0]

        # Reconstruct the sample with the new pose and fixed physique
        rec_sample = reconstruct_sample(gen_pose, physique_sample, scale_factors)
        print(f"REC SAMPLE {i + 1}: "
              f"\nbone_lengths [FIXED]: {rec_sample['bone_lengths']}"
              f"\nbase_skeleton [FIXED]: {rec_sample['base_skeleton']}"
              f"\njoint_positions: {rec_sample['joint_positions']}"
              f"\njoint_angles: {rec_sample['joint_angles']}")
        plot_pose_from_joint_angles(rec_sample, f"[VARIED] Pose {i + 1} with Fixed Physique")

    return


def modify_limb_lengths(stats, user_modifications, use_percentage=True):
    modified_bone_lengths = {}

    for limb in stats:
        avg_value = stats[limb]['avg']

        if limb in user_modifications:
            if use_percentage:
                # Apply percentage-based change
                percentage_change = user_modifications[limb]
                modified_value = avg_value * percentage_change
            else:
                # Direct value modification
                modified_value = user_modifications[limb]
        else:
            # No modification, use average
            modified_value = avg_value

        # Constrain within the min-max range
        min_value = stats[limb]['min']
        max_value = stats[limb]['max']
        modified_bone_lengths[limb] = np.clip(modified_value, min_value, max_value)

    return modified_bone_lengths


def user_def_physique_vary_pose_test(vae, num_samples):
    bone_lengths_dict = {'lefthip': 0.0,
                               'leftknee': 0.0,
                               'leftfoot': 0.0,
                               'righthip': 0.0,
                               'rightknee': 0.0,
                               'rightfoot': 0.0,
                               'leftshoulder': 0.0,
                               'leftelbow': 0.0,
                               'leftwrist': 0.0,
                               'rightshoulder': 0.0,
                               'rightelbow': 0.0,
                               'rightwrist': 0.0,
                               'neck': 0.0}

    presets = {
        'tall': {'lefthip': 1.1, 'leftknee': 1.1, 'leftfoot': 1.1, 'righthip': 1.1, 'rightknee': 1.1, 'rightfoot': 1.1,
                 'leftshoulder': 1.05, 'rightshoulder': 1.05, 'neck': 1.05},
        'short': {'lefthip': 0.9, 'leftknee': 0.9, 'leftfoot': 0.9, 'righthip': 0.9, 'rightknee': 0.9, 'rightfoot': 0.9,
                  'leftshoulder': 0.95, 'rightshoulder': 0.95, 'neck': 0.95},
        'athletic': {'leftshoulder': 1.15, 'rightshoulder': 1.15, 'neck': 0.9}
    }

    # using a preset physique
    # selected_preset = presets['tall']
    # modified_bone_lengths = modify_limb_lengths(physique_stats, selected_preset, use_percentage=True)

    # Example user input: modify some limbs by percentage (1.1 means 10% increase, 0.9 means 10% decrease)
    user_modifications = {'lefthip': 1.1, 'rightshoulder': 0.9}

    modified_bone_lengths = modify_limb_lengths(physique_stats, user_modifications)
    print(f"user_def_physique: {modified_bone_lengths}")

    # Encode the custom physique into latent space (via the decoder)
    physique_tensor = torch.tensor(list(modified_bone_lengths.values()), dtype=torch.float32).unsqueeze(0)
    latent_dim = vae.pose_mu.out_features

    for i in range(num_samples):
        z_pose = torch.randn(1, latent_dim)  # Generate random poses

        with torch.no_grad():
            # Decode the random pose
            gen_pose = vae.pose_decoder(z_pose).detach().cpu().numpy()[0]

        # Reconstruct the sample with the custom physique and varied pose
        rec_sample = reconstruct_sample(gen_pose, physique_tensor.squeeze().numpy(), scale_factors)
        plot_pose_from_joint_angles(rec_sample, f"[GENERATED] Pose {i + 1} with Custom Physique")

    return


if __name__ == '__main__':

    model_path = "../model/2b_vae/"
    model_name = "2b_vae_hd256x128_ld16.pth"

    param_file = "../angles_json/scale_factors.json"
    scale_factors = json.load(open(param_file))

    physique_stats_file = "../angles_json/bone_lengths_stats.json"
    physique_stats = json.load(open(physique_stats_file))

    checkpoint = torch.load(model_path + model_name)
    pose_input_dim = checkpoint['pose_input_dim']
    physique_input_dim = checkpoint['physique_input_dim']
    latent_dim = checkpoint['latent_dim']
    hidden_dim = checkpoint['hidden_dim']

    vae = VAE(pose_input_dim, physique_input_dim, hidden_dim, latent_dim)
    vae.load_state_dict(checkpoint['state_dict'])
    vae.eval()  # important

    # reconstruction_test(vae)
    # generation_test(vae, 3)
    # pure_generation_test(vae, 10, int(latent_dim))

    # fixed_pose_vary_physique_test(vae, 1, num_samples=3)
    # fixed_physique_vary_pose_test(vae, 1, num_samples=3)

    user_def_physique_vary_pose_test(vae, num_samples=5)
