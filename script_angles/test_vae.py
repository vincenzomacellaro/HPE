import random
import torch
import os

from train_vae import VAE
from human36_to_angles import reconstruct_from_array
from human36_to_angles import plot_pose_from_joint_angles
from human36_to_angles import load_data_for_train


def reconstruction_test(vae, test_data):
    num_samples = 1  # can be changed
    for i in range(num_samples):

        random_index = random.randint(0, len(test_data) - 1)
        sample = test_data[random_index]

        print(f">ORIGINAL_SAMPLE: \n{sample}")
        print(f">ORIGINAL_SAMPLE SHAPE: \n{sample.shape}")
        rec_sample = reconstruct_from_array(sample, sample_keys_file)
        print(f">RECONSTRUCTED_SAMPLE (dict): \n{rec_sample}")
        plot_pose_from_joint_angles(rec_sample, "3D plot from original [TEST] sample")

        # Ensure sample is a torch.Tensor with shape [1, 47]
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)
        if sample.dim() == 1:
            # Adding batch dimension if it's not there
            sample = sample.unsqueeze(0)

        encoded, decoded, _, _ = vae(sample)

        decoded_detached = decoded.detach()
        decoded_numpy = decoded_detached.cpu().numpy()

        print(f">DECODED_SAMPLE: \n{decoded_numpy[0]}")
        print(f">DECODED_SAMPLE SHAPE: \n{decoded_numpy[0].shape}")
        rec_decoded = reconstruct_from_array(decoded_numpy[0], sample_keys_file)
        print(f">RECONSTRUCTED_DECODED (dict): \n{rec_decoded}")
        plot_pose_from_joint_angles(rec_decoded, "3D plot from reconstructed [TEST] sample")

    return


def generation_test(vae, test_data):
    # Suppose 'sample' is your input sample from the test set
    random_index = random.randint(0, len(test_data) - 1)
    sample = test_data[random_index]

    print(f">ORIGINAL SAMPLE: \n{sample}")
    print(f">ORIGINAL SAMPLE SHAPE: \n{sample.shape}")
    rec_original = reconstruct_from_array(sample, sample_keys_file)
    plot_pose_from_joint_angles(rec_original, "3D plot from original [TEST] sample")

    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    encoded, _, mu, log_var = vae(sample_tensor)

    print(encoded.shape)

    # Generate multiple samples
    num_samples = 3
    gen_samples = torch.stack([vae.reparameterize(mu, log_var) for _ in range(num_samples)])

    # Correct call to decoder, ensuring the input is correctly shaped
    decoded_poses = vae.decoder(gen_samples.view(-1, latent_dim)).detach().cpu().numpy()

    for idx, d_pose in enumerate(decoded_poses):
        print(f">GEN_POSE: \n{d_pose}")
        print(f">GEN_POSE SHAPE: \n{d_pose.shape}")
        rec_d_pose = reconstruct_from_array(d_pose, sample_keys_file)
        print(f">RECONSTRUCTED_GEN_POSE: \n{rec_d_pose}")
        plot_pose_from_joint_angles(rec_d_pose, f"3D plot from generated sample - {idx + 1}")

    return


if __name__ == '__main__':

    plot_path = "../plots/generated/"
    model_path = "../model/angles/"
    model_name = "vae_angle_hd128x64_ld32.pth"
    sample_keys_file = "../angles_json/sample_keys.json"

    test_path = "../angles_data/test/test_data.json"
    test_data = load_data_for_train("test")

    os.makedirs(plot_path, exist_ok=True)

    checkpoint = torch.load(model_path + model_name)
    input_dim = checkpoint['input_dim']
    latent_dim = checkpoint['latent_dim']
    hidden_dim = checkpoint['hidden_dim']

    vae = VAE(input_dim, hidden_dim, latent_dim)
    vae.load_state_dict(checkpoint['state_dict'])
    vae.eval()

    # reconstruction_test(vae, test_data)
    generation_test(vae, test_data)
