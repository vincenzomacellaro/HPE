import numpy as np
import json
import os
import torch

from script_angles.train_vae import VAE
from script_angles.conversion_utils import convert_to_dictionary
from script_angles.conversion_utils import add_hips_and_neck
from script_angles.conversion_utils import get_bone_lengths
from script_angles.conversion_utils import get_base_skeleton
from script_angles.conversion_utils import calculate_joint_angles
from script_angles.conversion_utils import build_enhanced_sample
from script_angles.general_utils import from_numpy, to_numpy
from script_angles.align_poses import procrustes
from script_angles.human36_to_angles import flatten_numeric_values
from script_angles.human36_to_angles import filter_samples
from script_angles.human36_to_angles import reconstruct_from_array
from script_angles.plot_utils import plot_subplots
from script_angles.load_data_utils import normalize_single_sample, reverse_normalize_sample
from script_angles import mat_utils as utils


# coco19_joints = {"neck": 0, "nose": 1, "pelv": 2, "lshoulder": 3, "lelbow": 4 ,"lwrist" : 5, "lhip": 6, "lknee": 7,
#                 "lankle":8, "rshoulder":9, "relbow":10, "rwrist":11, "rhip": 12, "rknee": 13, "rankle": 14,
#                 "leye":15, "lear":16, "reye": 17, "rear":18 }


# {'Pelvis' 'RHip' 'RKnee' 'RAnkle' 'LHip' 'LKnee' 'LAnkle' 'Spine1' 'Neck' 'Head' 'Site' 'LShoulder' 'LElbow' 'LWrist' 'RShoulder' 'RElbow' 'RWrist};

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

coco19_joints = {0: 'neck',
                 1: 'nose',
                 2: 'pelv (hips)',
                 3: 'leftshoulder',
                 4: 'leftelbow',
                 5: 'leftwrist',
                 6: 'lefthip',
                 7: 'leftknee',
                 8: 'leftankle',
                 9: 'rightshoulder',
                 10: 'rightelbow',
                 11: 'rightwrist',
                 12: 'righthip',
                 13: 'rightknee',
                 14: 'rightankle',
                 15: 'lefteye',
                 16: 'leftear',
                 17: 'righteye',
                 18: 'rightear'}

coco19_flt_joints = {3: 'leftshoulder',
                     4: 'leftelbow',
                     5: 'leftwrist',
                     6: 'lefthip',
                     7: 'leftknee',
                     8: 'leftfoot',
                     9: 'rightshoulder',
                     10: 'rightelbow',
                     11: 'rightwrist',
                     12: 'righthip',
                     13: 'rightknee',
                     14: 'rightfoot'}


avg_pose = None


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

    # skeleton = [
    #     ('leftshoulder', 'leftelbow'),  # Left Shoulder to Left Elbow
    #     ('leftshoulder', 'rightshoulder'),  # Left Shoulder to Right Shoulder
    #     ('lefthip', 'leftknee'),  # Left Hip to Left Knee
    #     ('lefthip', 'righthip'),  # Left Hip to Right Hip
    #     ('leftknee', 'leftfoot'),  # Left Knee to Left Ankle
    #     ('rightshoulder', 'rightelbow'),  # Right Shoulder to Right Elbow
    #     ('righthip', 'rightknee'),  # Right Hip to Right Knee
    #     ('rightknee', 'rightfoot'),  # Right Knee to Right Ankle
    #     ('leftelbow', 'leftwrist'),  # Left Elbow to Left Wrist
    #     ('rightelbow', 'rightwrist'),  # Right Elbow to Right Wrist
    #     ('hips', 'neck')  # Hips Neck
    # ]
    #
    # plot_base_skeleton(kpts, skeleton)

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
    for i in range(rows - 1):  # Loop through each row except the last one
        if i in coco19_flt_joints:
            sorted_idx = sorted_joints[coco19_flt_joints[i]]
            # print(f"i: {i} - {coco19_flt_joints[i]} -> sorted_idx: {sorted_idx}")
            out[sorted_idx] = sample[0][i]

    return out


def flatten_array(sample):
    f_sample = filter_samples(sample)   # filters the complete sample to get only the needed data
    sample_np = to_numpy(f_sample)      # converts to numpy values
    flat_sample = flatten_numeric_values(sample_np)
    return flat_sample


def reconstruct(vae, sample):
    if not isinstance(sample, torch.Tensor):
        sample = torch.tensor(sample, dtype=torch.float32)
    if sample.dim() == 1:
        sample = sample.unsqueeze(0)

    encoded, decoded, _, _ = vae(sample)

    decoded_detached = decoded.detach()
    decoded_numpy = decoded_detached.cpu().numpy()

    # decoded_sample = reconstruct_sample(decoded_numpy, scale_factors)
    # plot_pose_from_joint_angles(decoded_sample, "[RECONSTRUCTED TEST] sample")

    return decoded_numpy


def prepare_for_input(raw_sample):
    proc_sample = filter_and_sort(raw_sample)

    sample_w_angles = convert_to_angles_mod([proc_sample])
    sample_w_angles_np = to_numpy(sample_w_angles)  # converts to numpy values
    filtered_sample = filter_samples(sample_w_angles_np)  # extracts joints needed for plotting
    norm_sample = normalize_single_sample(filtered_sample, scale_factors)  # normalizes single sample
    num_sample = flatten_numeric_values(norm_sample)  # ex-flattened sample

    return num_sample


def reconstruct_sample(num_sample, scale_factors):
    rec_sample = reconstruct_from_array(num_sample)
    den_sample = reverse_normalize_sample(rec_sample, scale_factors)
    return den_sample


def process_original_sample(sample):
    proc_sample = filter_and_sort(sample)
    angles_sample = convert_to_angles_mod([proc_sample])
    angles_sample_np = to_numpy(angles_sample)

    angles_sample_np['joints'] = [
        "lefthip", "leftknee", "leftfoot",
        "righthip", "rightknee", "rightfoot",
        "leftshoulder", "leftelbow", "leftwrist",
        "rightshoulder", "rightelbow", "rightwrist",
        "hips", "neck"
    ]

    return angles_sample_np


def test_npy():
    # loads [ground_truth] and [predictions] data from the diffusion model, then passes them through the VAE model

    gt_path = '../diffusion_data/gt_00001.npy'
    pred_path = '../diffusion_data/pred_00001.npy'

    # Load the .npy file
    gt = np.load(gt_path)
    pred = np.load(pred_path)
    cnt = 0

    # # Print out the contents of the numpy array
    # print(f"Ground Truth file: {gt.shape}")  # ground truth, pose per ogni frame (64, 1, 20, 3)
    # print(f"Pred file: {pred.shape}")  # 20 ipotesi per ogni frame, (64, 1, 20, 1, 20, 3)

    idx_dim = gt.shape[0]

    for idx in range(idx_dim):
        gt_sample = gt[idx]  # (1, 20, 3) Coco19 (stock)
        # orig_sample = process_original_sample(gt_sample)  # [prepare_for_input] BUT for the original sample
        # previous line creates the <dict>-based sample for plotting
        # print(orig_sample)

        gt_sample_in = prepare_for_input(gt_sample)
        gt_sample_out = reconstruct_sample(gt_sample_in, scale_factors)

        # plot_pose_from_joint_angles(orig_sample, "[GROUND TRUTH][ORIGINAL]")
        # plot_pose_from_joint_angles(out_sample, "[GROUND TRUTH][REC]")

        # pred[idx].shape -> (1, 20, 1, 20, 3)
        # pred[idx][0].shape -> (20, 1, 20, 3)

        pred_tensor = torch.tensor(pred[idx][0], dtype=torch.float32)
        pred_tensor_avg = torch.mean(pred_tensor, dim=0, keepdim=True)[0]  # avg_pred: torch.Size([1, 20, 3])

        orig_pred = process_original_sample(pred_tensor_avg)  # <dict>-based repr. for plotting of original hypothesis
        in_pred = prepare_for_input(pred_tensor_avg.numpy())
        in_pred_tensor = torch.tensor(in_pred, dtype=torch.float32)

        if in_pred_tensor.dim() == 1:
            # Adding batch dimension if it's not there
            in_pred_tensor = in_pred_tensor.unsqueeze(0)

        encoded, decoded, _, _ = vae(in_pred_tensor)
        decoded_detached = decoded.detach()
        decoded_numpy = decoded_detached.cpu().numpy()

        out_hyp = reconstruct_sample(decoded_numpy[0], scale_factors)

        kpts_list = [gt_sample_out, orig_pred, out_hyp]
        plot_subplots(kpts_list)

        cnt += 1
        if cnt == 3:
            return


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
    vae.eval()  # This will make sure that BatchNorm1D uses running mean/variance instead of batch statistics.

    test_npy()

