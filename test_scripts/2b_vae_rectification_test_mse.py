import numpy as np
import json
import torch
import pandas as pd

from script_2b_vae.train_2b_vae import VAE
from script_angles.human36_to_angles import flatten_numeric_values, filter_samples, reconstruct_from_array
from script_angles.load_data_utils import normalize_dataset, flatten_dataset

from test_utils import convert_to_angles, reconstruct_sample


keypoints_to_index = {
    'lefthip': 6,
    'leftknee': 8,
    'leftfoot': 10,
    'righthip': 7,
    'rightknee': 9,
    'rightfoot': 11,
    'leftshoulder': 0,
    'leftelbow': 2,
    'leftwrist': 4,
    'rightshoulder': 1,
    'rightelbow': 3,
    'rightwrist': 5
}

sorted_joints = {
    'leftshoulder': 0,
    'rightshoulder': 1,
    'leftelbow': 2,
    'rightelbow': 3,
    'leftwrist': 4,
    'rightwrist': 5,
    'lefthip': 6,
    'righthip': 7,
    'leftknee': 8,
    'rightknee': 9,
    'leftfoot': 10,
    'rightfoot': 11
}

flt_joints = {
    1: 'righthip',
    2: 'rightknee',
    3: 'rightfoot',
    4: 'lefthip',
    5: 'leftknee',
    6: 'leftfoot',
    11: 'leftshoulder',
    12: 'leftelbow',
    13: 'leftwrist',
    14: 'rightshoulder',
    15: 'rightelbow',
    16: 'rightwrist'
}

# joint_names list without end-joints, as they always have value 0.000
joint_names_no_ep = ['lefthip_angles', 'leftknee_angles',
                     'righthip_angles', 'rightknee_angles',
                     'leftshoulder_angles', 'leftelbow_angles',
                     'rightshoulder_angles', 'rightelbow_angles',
                     'hips_angles', 'neck_angles']


def extract_and_sort(sample):
    rows = sample.shape[1]  # coordinates
    out = np.zeros((len(keypoints_to_index), 3))
    for i in range(rows):  # Loop through each row except the last one
        if i in flt_joints:
            sorted_idx = sorted_joints[flt_joints[i]]
            out[sorted_idx] = sample[0][i]

    return out


def get_dict_representation(raw_sample):
    sample = extract_and_sort(raw_sample)
    # shape: (12, 3) -> <class 'numpy.ndarray'>
    sample_dict = convert_to_angles(sample)
    filtered_sample = filter_samples(sample_dict)
    # dict_keys(['joint_positions', 'joint_angles', 'bone_lengths'])
    return filtered_sample


def load_unprocessed_sample(raw_sample):
    filtered_sample = get_dict_representation(raw_sample)
    num_sample = flatten_numeric_values(filtered_sample)
    rec_sample = reconstruct_from_array(num_sample, sample_keys_file)
    return rec_sample


def load_processed_sample(raw_sample):
    filtered_sample = get_dict_representation(raw_sample)
    scale_factors = json.load(open(param_file))
    proc_sample_angles_norm = normalize_dataset([filtered_sample], scale_factors)
    pose_sample, physique_sample = flatten_dataset(proc_sample_angles_norm)
    return pose_sample[0], physique_sample[0]


def mse(a, b):
    return np.mean((a - b) ** 2)


def compute_mse_per_joint(gt, pred):
    """Compute MSE for each joint between ground truth and predicted poses."""
    mse_per_joint = {}
    for joint_name, gt_angles in gt['joint_angles'].items():
        pred_angles = pred['joint_angles'].get(joint_name, None)
        if pred_angles is not None:
            mse_per_joint[joint_name] = mse(gt_angles, pred_angles)
        else:
            mse_per_joint[joint_name] = None  # Handle missing joint angles in prediction
    return mse_per_joint


def compute_aggregated_mse(mse_per_joint):
    """Compute the aggregated MSE across all joints."""
    mse_values = [v for v in mse_per_joint.values() if v is not None]  # Ignore None values
    return np.mean(mse_values)


if __name__ == '__main__':
    model_path = "../model/2b_vae/"
    model_name = "2b_vae_hd256x128_ld16.pth"
    sample_keys_file = "../angles_json/sample_keys.json"

    param_file = "../angles_json/scale_factors.json"
    scale_factors = json.load(open(param_file))

    checkpoint = torch.load(model_path + model_name)
    pose_input_dim = checkpoint['pose_input_dim']
    physique_input_dim = checkpoint['physique_input_dim']
    latent_dim = checkpoint['latent_dim']
    hidden_dim = checkpoint['hidden_dim']

    vae = VAE(pose_input_dim, physique_input_dim, hidden_dim, latent_dim)
    vae.load_state_dict(checkpoint['state_dict'])
    vae.eval()  # important

    # ground truth and pred_paths (simoni)
    gt_path = '../diffusion_data/gt.npy'
    pred_path = '../diffusion_data/predictions_ours_20hyp.npy'

    ground_truth = np.load(gt_path)
    predictions = np.load(pred_path)

    # Print out the contents of the numpy array
    print(f"Ground Truth file (shape): {ground_truth.shape}")
    print(f"Pred file (shape): {predictions.shape}")

    dim_idx = ground_truth.shape[0]  # 543344
    cnt = 0

    # Initialize an empty list to store the data
    data = []

    # Variable to set the maximum number of samples to read from file
    max_samples = 50000

    for idx in range(dim_idx):

        # p : t = r : 100
        # idx : max_samples = r : 100; r = (idx * 100) / max_samples
        print(f"curr_timestep: {idx}/{max_samples} - {int((idx * 100)/max_samples)}%")

        # timestep
        gt_sample = ground_truth[idx]  # (1, 17, 3)
        gt_sample = load_unprocessed_sample(gt_sample)

        hypothesis_for_curr_sample = predictions[idx]  # (20, 17, 3)

        for hyp_cnt, hyp in enumerate(hypothesis_for_curr_sample):

            hyp_pose_in, hyp_physique_in = load_processed_sample(hyp.reshape(1, 17, 3))
            hyp_out = reconstruct_sample(hyp_pose_in, hyp_physique_in, scale_factors)  # original hypothesis

            pose_tensor = torch.tensor(hyp_pose_in, dtype=torch.float32).unsqueeze(0)
            physique_tensor = torch.tensor(hyp_physique_in, dtype=torch.float32).unsqueeze(0)

            # inference w/VAE
            pose_recon, physique_recon, pose_mu, pose_log_var, physique_mu, physique_log_var = vae(pose_tensor,
                                                                                                   physique_tensor)

            reconstructed_pose_np = pose_recon.detach().cpu().numpy()
            reconstructed_physique_np = physique_recon.detach().cpu().numpy()

            hyp_vae = reconstruct_sample(reconstructed_pose_np[0], reconstructed_physique_np[0], scale_factors)

            # per-joint MSE
            mse_per_joint_hyp = compute_mse_per_joint(gt_sample, hyp_out)
            mse_per_joint_vae = compute_mse_per_joint(gt_sample, hyp_vae)

            # aggregated MSE (average of per-joint MSE)
            aggregated_mse_hyp = compute_aggregated_mse(mse_per_joint_hyp)
            aggregated_mse_vae = compute_aggregated_mse(mse_per_joint_vae)

            row = {}

            # Add pre-VAE per-joint MSE
            for joint in joint_names_no_ep:
                row[('pre_vae', joint)] = mse_per_joint_hyp[joint]

            # Add post-VAE per-joint MSE
            for joint in joint_names_no_ep:
                row[('post_vae', joint)] = mse_per_joint_vae[joint]

            # Also add aggregated MSE for both pre-VAE and post-VAE
            row[('pre_vae', 'aggregated_mse')] = aggregated_mse_hyp
            row[('post_vae', 'aggregated_mse')] = aggregated_mse_vae

            # Add the timestep and hypothesis index for tracking
            row['timestep'] = idx
            row['hypothesis'] = hyp_cnt

            # Append the row to the data list
            data.append(row)

        cnt += 1
        if cnt == max_samples:
            break

    # Create DataFrame from the collected data
    df = pd.DataFrame(data)
    df.set_index(['timestep', 'hypothesis'], inplace=True)  # multi-level index for better querying (ts & hyp)

    # Save or manipulate the DataFrame as needed
    csv_name = '../diffusion_data/2b_per_joint_mse_' + str(max_samples) + '.csv'
    df.to_csv(csv_name)

    # Aligning pre-VAE and post-VAE columns for each joint
    pre_vae_cols = df[[col for col in df.columns if col[0] == 'pre_vae']]
    post_vae_cols = df[[col for col in df.columns if col[0] == 'post_vae']]

    # Check if both pre_vae and post_vae columns have multi-level indexing
    if not isinstance(pre_vae_cols.columns, pd.MultiIndex):
        pre_vae_cols.columns = pd.MultiIndex.from_tuples(pre_vae_cols.columns)

    if not isinstance(post_vae_cols.columns, pd.MultiIndex):
        post_vae_cols.columns = pd.MultiIndex.from_tuples(post_vae_cols.columns)

    # Ensure both pre_vae and post_vae have matching inner joint names (second level of columns)
    assert pre_vae_cols.columns.droplevel(0).equals(
        post_vae_cols.columns.droplevel(0)), "Columns must match at the joint level"

    # Compute per-joint mean and std deviation for pre-VAE
    pre_vae_means = pre_vae_cols.mean()
    # pre_vae_stds = pre_vae_cols.std()

    # Compute per-joint mean and std deviation for post-VAE
    post_vae_means = post_vae_cols.mean()
    # post_vae_stds = post_vae_cols.std()

    # delta MSE (pre-VAE - post-VAE) for each joint
    delta_mse = pre_vae_cols.values - post_vae_cols.values

    # Convert back to DataFrame with the same joint names (use the second level of the column index)
    delta_mse_df = pd.DataFrame(delta_mse, columns=pre_vae_cols.columns.droplevel(0), index=df.index)

    # Create DataFrames for Pre-VAE, Post-VAE, and Delta MSE
    pre_vae_df = pre_vae_means.reset_index(drop=True)
    post_vae_df = post_vae_means.reset_index(drop=True)
    delta_mse_df = delta_mse_df.mean().reset_index(drop=True)

    # Combine into a single DataFrame
    combined_df = pd.DataFrame({
        'Joint': pre_vae_means.index.droplevel(0),  # Extract joint names from index
        'Pre-VAE Mean MSE': pre_vae_means.values,
        'Post-VAE Mean MSE': post_vae_means.values,
        'Delta MSE (Pre-VAE - Post-VAE)': delta_mse_df.values
    })

    # Save to CSV
    combined_csv_name = '../diffusion_data/2b_combined_mse_stats_' + str(max_samples) + '.csv'
    combined_df.to_csv(combined_csv_name, index=False)

