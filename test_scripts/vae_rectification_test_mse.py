import numpy as np
import json
import os
import torch
import pandas as pd

from script_vae.train_vae import VAE
from script_angles import mat_utils as utils
from script_angles.align_poses import procrustes
from script_angles.general_utils import from_numpy, to_numpy
from script_angles.conversion_utils import convert_to_dictionary
from script_angles.conversion_utils import add_hips_and_neck
from script_angles.conversion_utils import get_bone_lengths
from script_angles.conversion_utils import get_base_skeleton
from script_angles.conversion_utils import calculate_joint_angles
from script_angles.conversion_utils import build_enhanced_sample
from script_angles.human36_to_angles import flatten_numeric_values
from script_angles.human36_to_angles import filter_samples
from script_angles.human36_to_angles import reconstruct_from_array
from script_angles.load_data_utils import normalize_sample, de_normalize_sample
from test_scripts.test_utils import get_avg_pose


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


def reconstruct_sample(sample, scale_factors):
    # reconstruct_from_array wasn't modified to accept the 2b technique
    rec_sample = reconstruct_from_array(sample)     # [flat] to [dict] sample
    den_sample = de_normalize_sample(rec_sample, scale_factors)  # de_normalize sample
    return den_sample


def convert_to_angles_mod(sample):
    print(sample.shape)

    global avg_pose
    if avg_pose is None:
        avg_pose = get_avg_pose()

    frame_kpts = np.array(sample).reshape(1, 12, 3)
    oriented_pose = procrustes(frame_kpts[0], avg_pose).reshape(1, 12, 3)

    R = utils.get_R_z(np.pi / 2)
    for kpt_num in range(oriented_pose.shape[1]):
        oriented_pose[0, kpt_num] = R @ oriented_pose[0, kpt_num]

    kpts = convert_to_dictionary(oriented_pose)
    add_hips_and_neck(kpts)
    get_bone_lengths(kpts)
    get_base_skeleton(kpts)

    calculate_joint_angles(kpts)
    en_sample = build_enhanced_sample(kpts)

    sample_dict = from_numpy(en_sample)
    sample_dict['joints'] = ["lefthip", "leftknee", "leftfoot",
                             "righthip", "rightknee", "rightfoot",
                             "leftshoulder", "leftelbow", "leftwrist",
                             "rightshoulder", "rightelbow", "rightwrist",
                             "hips", "neck"]

    return sample_dict


def filter_and_sort(sample):
    rows = sample.shape[1]  # coordinates
    out = np.zeros((len(keypoints_to_index), 3))
    for i in range(rows):  # Loop through each row except the last one
        if i in flt_joints:
            sorted_idx = sorted_joints[flt_joints[i]]
            # print(f"i: {i} - {flt_joints[i]} -> sorted_idx: {sorted_idx}")
            out[sorted_idx] = sample[0][i]

    return out


def prepare_for_input(raw_sample):
    # extract_and_sort: filters <12> joints out of the <20> original H3.6Ms, as specified by our approach
    # convert_to_angles_mod: converts the <12> numeric joints into their <dict> representation
    # to_numpy: converts the <dict> values into their numpy representation
    # filter_samples: extracts joints needed for plotting
    # normalize_sample: normalizes the current <single> sample according to the "global" scale factors
    # flatten_numeric_values: flattens the dict sample into the (47,) shaped array needed for the autoencoder

    proc_sample = filter_and_sort(raw_sample)
    sample_w_angles = convert_to_angles_mod(proc_sample)
    sample_w_angles_np = to_numpy(sample_w_angles)
    filtered_sample = filter_samples(sample_w_angles_np)
    norm_sample = normalize_sample(filtered_sample, scale_factors)  # normalizes single sample
    num_sample = flatten_numeric_values(norm_sample)  # ex-flattened sample

    return num_sample


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
    vae.eval()

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

    # Suppose we have joint names
    # joint_names = ['lefthip_angles', 'leftknee_angles', 'leftfoot_angles', 'righthip_angles',
    #                'rightknee_angles', 'rightfoot_angles', 'leftshoulder_angles', 'leftelbow_angles',
    #                'leftwrist_angles', 'rightshoulder_angles', 'rightelbow_angles', 'rightwrist_angles',
    #                'hips_angles', 'neck_angles']

    # joint_names list without endpoint joints, as they always have value 0.000
    joint_names_no_ep = ['lefthip_angles', 'leftknee_angles',
                         'righthip_angles', 'rightknee_angles',
                         'leftshoulder_angles', 'leftelbow_angles',
                         'rightshoulder_angles', 'rightelbow_angles',
                         'hips_angles', 'neck_angles']

    # Initialize an empty list to store the data
    data = []

    # Variable to set the maximum number of samples to read from file
    max_samples = 50000

    for idx in range(dim_idx):

        print(f"curr_timestep: {idx}")

        # timestep
        gt_sample = ground_truth[idx]  # (1, 17, 3)
        gt_sample_in = prepare_for_input(gt_sample)  # (47,)
        gt_sample_out = reconstruct_sample(gt_sample_in, scale_factors)

        hypothesis_for_curr_sample = predictions[idx]  # (20, 17, 3)

        for hyp_cnt, hyp in enumerate(hypothesis_for_curr_sample):

            # hypothesis
            # prepare_for_input function requires a (1, 17, 3)-shaped input array
            hyp_in = prepare_for_input(hyp.reshape(1, 17, 3))  # (47,) as expected
            hyp_out = reconstruct_sample(hyp_in, scale_factors)

            in_hyp = torch.tensor(hyp_in, dtype=torch.float32).unsqueeze(0)  # torch.size([47])

            encoded, decoded, _, _ = vae(in_hyp)
            decoded_detached = decoded.detach()
            decoded_numpy = decoded_detached.cpu().numpy()

            hyp_vae = reconstruct_sample(decoded_numpy[0], scale_factors)

            # Compute per-joint MSE for each prediction
            mse_per_joint_hyp = compute_mse_per_joint(gt_sample_out, hyp_out)
            mse_per_joint_vae = compute_mse_per_joint(gt_sample_out, hyp_vae)

            # Compute aggregated MSE (average of per-joint MSE)
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

    # Set multi-level index for better querying (timestep and hypothesis)
    df.set_index(['timestep', 'hypothesis'], inplace=True)

    # Save or manipulate the DataFrame as needed
    csv_name = '../diffusion_data/per_joint_mse_' + str(max_samples) + '.csv'
    df.to_csv(csv_name)

    # Aligning pre-VAE and post-VAE columns for each joint
    pre_vae_cols = df[[col for col in df.columns if col[0] == 'pre_vae']]
    post_vae_cols = df[[col for col in df.columns if col[0] == 'post_vae']]

    # Check if both pre_vae and post_vae columns have multi-level indexing
    if not isinstance(pre_vae_cols.columns, pd.MultiIndex):
        # print("Converting pre_vae columns to MultiIndex...")
        pre_vae_cols.columns = pd.MultiIndex.from_tuples(pre_vae_cols.columns)

    if not isinstance(post_vae_cols.columns, pd.MultiIndex):
        # print("Converting post_vae columns to MultiIndex...")
        post_vae_cols.columns = pd.MultiIndex.from_tuples(post_vae_cols.columns)

    # Ensure both pre_vae and post_vae have matching inner joint names (second level of columns)
    assert pre_vae_cols.columns.droplevel(0).equals(
        post_vae_cols.columns.droplevel(0)), "Columns must match at the joint level"

    # Compute per-joint mean and std deviation for pre-VAE
    pre_vae_means = pre_vae_cols.mean()
    pre_vae_stds = pre_vae_cols.std()

    # Compute per-joint mean and std deviation for post-VAE
    post_vae_means = post_vae_cols.mean()
    post_vae_stds = post_vae_cols.std()

    # Compute delta MSE (pre-VAE - post-VAE) for each joint
    delta_mse = pre_vae_cols.values - post_vae_cols.values

    # # Convert back to DataFrame with the same joint names (use the second level of the column index)
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
    combined_csv_name = '../diffusion_data/combined_mse_stats_' + str(max_samples) + '.csv'
    combined_df.to_csv(combined_csv_name, index=False)

