import numpy as np
import json
import os
import torch

from script_angles.train_vae import VAE
from script_angles import mat_utils as utils
from script_angles.align_poses import procrustes
from script_angles.general_utils import from_numpy, to_numpy
from script_angles.plot_utils import plot_subplots
from script_angles.conversion_utils import convert_to_dictionary
from script_angles.conversion_utils import add_hips_and_neck
from script_angles.conversion_utils import get_bone_lengths
from script_angles.conversion_utils import get_base_skeleton
from script_angles.conversion_utils import calculate_joint_angles
from script_angles.conversion_utils import build_enhanced_sample
from script_angles.human36_to_angles import flatten_numeric_values
from script_angles.human36_to_angles import filter_samples
from script_angles.human36_to_angles import reconstruct_from_array
from script_angles.load_data_utils import normalize_single_sample, reverse_normalize_sample


avg_pose = None

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
    rec_sample = reconstruct_from_array(sample)     # [flat] to [dict] sample
    den_sample = reverse_normalize_sample(rec_sample, scale_factors)  # de_normalize sample
    return den_sample


def get_avg_pose(avg_pose_file="../angles_json/avg_pose.json"):
    with open(avg_pose_file, "r") as f:
        avg_pose = json.load(f)
        np_avg_pose = np.array(avg_pose)
        return np_avg_pose


def convert_to_angles_mod(sample):

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
    # filter_and_sort: filters <12> joints out of the <20> original H3.6Ms, as specified by our approach
    # convert_to_angles_mod: converts the <12> numeric joints into their <dict> representation
    # to_numpy: converts the <dict> values into their numpy representation
    # filter_samples: extracts joints needed for plotting
    # normalize_single_sample: normalizes the current <single> sample according to the "global" scale factors
    # flatten_numeric_values: flattens the dict sample into the (47,) shaped array needed for the autoencoder

    proc_sample = filter_and_sort(raw_sample)
    sample_w_angles = convert_to_angles_mod([proc_sample])
    sample_w_angles_np = to_numpy(sample_w_angles)
    filtered_sample = filter_samples(sample_w_angles_np)
    norm_sample = normalize_single_sample(filtered_sample, scale_factors)  # normalizes single sample
    num_sample = flatten_numeric_values(norm_sample)  # ex-flattened sample

    return num_sample


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
    gt_path = '../ref_data/gt.npy'
    pred_path = '../ref_data/predictions_ours_20hyp.npy'

    ground_truth = np.load(gt_path)
    predictions = np.load(pred_path)

    # Print out the contents of the numpy array
    print(f"Ground Truth file (shape): {ground_truth.shape}")
    print(f"Pred file (shape): {predictions.shape}")

    gt_tensor = torch.tensor(ground_truth, dtype=torch.float32)
    pred_tensor = torch.tensor(predictions, dtype=torch.float32)

    dim_idx = gt_tensor.shape[0]

    cnt = 0

    for idx in range(dim_idx):

        gt_sample = gt_tensor[idx]
        gt_sample_in = prepare_for_input(gt_sample)
        gt_sample_out = reconstruct_sample(gt_sample_in, scale_factors)
        # plot_pose_from_joint_angles(gt_sample_out, "[GROUND TRUTH]")

        pred_sample = pred_tensor[idx] # torch.Size([20, 17, 3])
        # torch.mean on dim=1 to collapse the 20 hypothesis into a single aggregated one
        avg_pred = torch.mean(pred_sample, dim=0, keepdim=True)  # avg_pred: torch.Size([1, 17, 3])

        pred_sample_in = prepare_for_input(avg_pred.numpy())
        pred_sample_out = reconstruct_sample(pred_sample_in, scale_factors)
        # plot_pose_from_joint_angles(pred_sample_out, "[AGGREGATED PRED][ORIGINAL]")

        # AUTOENCODER PASS
        in_hyp = torch.tensor(pred_sample_in, dtype=torch.float32)
        if in_hyp.dim() == 1:
            in_hyp = in_hyp.unsqueeze(0)

        encoded, decoded, _, _ = vae(in_hyp)
        decoded_detached = decoded.detach()
        decoded_numpy = decoded_detached.cpu().numpy()

        out_pose = reconstruct_sample(decoded_numpy[0], scale_factors)
        # plot_pose_from_joint_angles(out_pose, "[AGGREGATED PRED][RECTIFIED]")

        kpts_list = [gt_sample_out, pred_sample_out, out_pose]
        plot_subplots(kpts_list)

        cnt += 1
        if cnt == 3:
            exit(1)

