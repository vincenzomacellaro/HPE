import random
import torch
import os
import json

from train_vae import VAE
from script_angles.plot_utils import plot_pose_from_joint_angles
from script_angles.human36_to_angles import reconstruct_from_array
from script_angles.load_data_utils import load_data_for_train, de_normalize_sample


def reconstruct_sample(sample, scale_factors):
    rec_sample = reconstruct_from_array(sample)     # [flat] to [dict] sample
    den_sample = de_normalize_sample(rec_sample, scale_factors)  # de_normalize sample
    return den_sample


def reconstruction_test(vae, num_samples=3):
    test_data = load_data_for_train("test")

    for i in range(num_samples):

        random_index = random.randint(0, len(test_data) - 1)
        sample = test_data[random_index]  # norm. and flattened sample (via the load_data_for_train script) (47,)

        # Ensure sample is a torch.Tensor with shape [1, 47]
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)
        if sample.dim() == 1:
            # Adding batch dimension if it's not there
            sample = sample.unsqueeze(0)

        encoded, decoded, _, _ = vae(sample)
        decoded_detached = decoded.detach()
        decoded_numpy = decoded_detached.cpu().numpy()  # (1, 47) -> decoded_numpy[0]

        print(f"input: \n{decoded_numpy}")
        decoded_sample = reconstruct_sample(decoded_numpy[0], scale_factors)
        print(f"output: {decoded_sample}")
        plot_pose_from_joint_angles(decoded_sample, "[RECONSTRUCTED TEST] sample")

    return


def generation_test(vae, num_samples=3):

    # noise from sample
    test_data = load_data_for_train("test")

    random_index = random.randint(0, len(test_data) - 1)
    sample = test_data[random_index]

    original_sample = reconstruct_sample(sample, scale_factors)
    plot_pose_from_joint_angles(original_sample, "[ORIGINAL] sample")

    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    encoded, _, mu, log_var = vae(sample_tensor)

    # Generate multiple samples
    gen_samples = torch.stack([vae.reparameterize(mu, log_var) for _ in range(num_samples)])

    # Pass the noise through the decoder to generate new data
    with torch.no_grad():  # Turn off gradients to speed up the process
        gen_poses = vae.decoder(gen_samples.view(-1, latent_dim)).detach().cpu().numpy()

    for idx, d_pose in enumerate(gen_poses):
        print(f">GEN_POSE SHAPE: \n{d_pose.shape}")
        rec_pose = reconstruct_sample(d_pose, scale_factors)
        plot_pose_from_joint_angles(rec_pose, f"[GENERATED] sample - {idx + 1}")

    return


def sample_from_latent_space(vae, num_samples, latent_dim, device='cpu'):
    # Sample random noise (z) from the standard normal distribution
    z = torch.randn(num_samples, latent_dim).to(device)

    # Pass the noise through the decoder to generate new data
    with torch.no_grad():  # Turn off gradients to speed up the process
        generated_samples = vae.decoder(z)

    # Move the generated samples to CPU and convert to numpy array for easy handling
    generated_samples = generated_samples.cpu().numpy()

    return generated_samples


def pure_generation_test(vae, num_samples, latent_dim):
    gen_samples = sample_from_latent_space(vae, num_samples, latent_dim)

    for sample in gen_samples:

        rec_sample = reconstruct_sample(sample, scale_factors)
        print(f"REC SAMPLE: \n{rec_sample}")
        plot_pose_from_joint_angles(rec_sample, "3D pose from [NOISE]")


if __name__ == '__main__':

    plot_path = "../plots/generated/"
    model_path = "../model/angles/"
    model_name = "vae_angle_hd128x64_ld32.pth"
    sample_keys_file = "../angles_json/sample_keys.json"

    param_file = "../angles_json/scale_factors.json"
    scale_factors = json.load(open(param_file))

    os.makedirs(plot_path, exist_ok=True)

    checkpoint = torch.load(model_path + model_name)
    input_dim = checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']
    hidden_dim = checkpoint['hidden_dim']

    vae = VAE(input_dim, hidden_dim, latent_dim)
    vae.load_state_dict(checkpoint['state_dict'])
    vae.eval()  # importante

    reconstruction_test(vae)
    # generation_test(vae, test_data)
    # pure_generation_test(vae, 10, int(latent_dim))
