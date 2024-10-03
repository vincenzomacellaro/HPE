import numpy as np
import json
import torch

from script_2b_vae.train_2b_vae import VAE
from script_angles.conversion_utils import convert_to_dictionary
from script_angles.conversion_utils import add_hips_and_neck
from script_angles.conversion_utils import get_bone_lengths
from script_angles.conversion_utils import get_base_skeleton
from script_angles.conversion_utils import calculate_joint_angles
from script_angles.conversion_utils import build_enhanced_sample
from script_angles.general_utils import from_numpy, to_numpy
from script_angles.human36_to_angles import flatten_numeric_values
from script_angles.human36_to_angles import filter_samples
from script_angles.human36_to_angles import reconstruct_from_array
from script_angles.plot_utils import plot_subplots
from script_angles.load_data_utils import normalize_dataset, de_normalize_sample, flatten_dataset
from script_angles import mat_utils as utils
from test_utils import get_avg_pose, convert_to_angles, reconstruct_sample

from script_angles.plot_utils import plot_pose_from_joint_angles
from script_angles.load_data_utils import load_data_for_train, de_normalize_sample


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


# this version of the "extract_and_sort" function is made specifically for the coco19 dataset
# therefore it is different from the one in the 2b_vae_rectification_test_mse.py script
def extract_and_sort(sample):
    rows = sample.shape[1]  # coordinates
    out = np.zeros((len(keypoints_to_index), 3))
    for i in range(rows - 1):  # Loop through each row except the last one
        if i in coco19_flt_joints:
            sorted_idx = sorted_joints[coco19_flt_joints[i]]
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


def compute_per_joint_mse(dict1, dict2):
    mse_per_joint = {}
    for key in dict1:
        mse_per_joint[key] = (dict1[key] - dict2[key]) ** 2
    return mse_per_joint


def compute_average_mse(mse_per_joint):
    return sum(mse_per_joint.values()) / len(mse_per_joint)


def test_npy():
    # this cript loads [ground_truth] and [predictions] data diffusion model
    # then passes them through the VAE model
    gt_path = '../diffusion_data/gt_00001.npy'
    pred_path = '../diffusion_data/pred_00001.npy'

    # Load the .npy file
    ground_truth = np.load(gt_path)
    pred = np.load(pred_path)
    cnt = 0

    # print(f"Ground Truth file: {gt.shape}")  # ground truth, pose per ogni frame (64, 1, 20, 3)
    # print(f"Pred file: {pred.shape}")  # 20 ipotesi per ogni frame, (64, 1, 20, 1, 20, 3)

    idx_dim = ground_truth.shape[0]

    for idx in range(idx_dim):
        gt_sample = ground_truth[idx]  # (1, 20, 3) Coco19 (stock) -> (1, 12, 3) representation, compatible with the 2b_vae model
        gt_sample = load_unprocessed_sample(gt_sample)
        print(f"gt_sample['bone_lengths'] \n{gt_sample['bone_lengths']}")
        plot_pose_from_joint_angles(gt_sample, "[COCO19] GROUND TRUTH", padding=0.15)

        # pred[idx].shape -> (1, 20, 1, 20, 3)
        # pred[idx][0].shape -> (20, 1, 20, 3)

        pred_tensor = torch.tensor(pred[idx][0], dtype=torch.float32)
        pred_tensor_avg = torch.mean(pred_tensor, dim=0, keepdim=True)[0]  # avg_pred: torch.Size([1, 20, 3])
        # [20 hyp] avg tensor used as a sample we want to process with our VAE model

        pred_tensor_avg_unprocessed = load_unprocessed_sample(pred_tensor_avg.numpy())
        print(f"pred_tensor_unprocessed['bone_lengths'] \n{pred_tensor_avg_unprocessed['bone_lengths']}")
        plot_pose_from_joint_angles(pred_tensor_avg_unprocessed, "[COCO19]AVG PRED TENSOR", padding=0.15)

        pose_pred, physique_pred = load_processed_sample(pred_tensor_avg.numpy())

        pose_tensor = torch.tensor(pose_pred, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        physique_tensor = torch.tensor(physique_pred, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        pose_recon, physique_recon, pose_mu, pose_log_var, physique_mu, physique_log_var = vae(pose_tensor,
                                                                                               physique_tensor)

        reconstructed_pose_np = pose_recon.detach().cpu().numpy()
        reconstructed_physique_np = physique_recon.detach().cpu().numpy()

        # [pose_sample, physique_sample]
        # we are using [scale factors] that calculated on the Human3.6M dataset, may not be suitable for COCO19
        # spine segment looks indeed shorter than the original one, after processing.

        pred_vae = reconstruct_sample(reconstructed_pose_np[0], reconstructed_physique_np[0], scale_factors)
        print(f"pred_vae['bone_lengths']: \n{pred_vae['bone_lengths']}")

        plot_pose_from_joint_angles(pred_vae, "[COCO19]PROCESSED AVG PRED TENSOR")

        # Calculate MSE for unprocessed prediction (processed prediction) vs ground truth
        mse_per_joint_unprocessed = compute_per_joint_mse(gt_sample['bone_lengths'],
                                                          pred_tensor_avg_unprocessed['bone_lengths'])
        mse_per_joint_processed = compute_per_joint_mse(gt_sample['bone_lengths'],
                                                        pred_vae['bone_lengths'])

        print(f"\n[MSE] [GT - Unprocessed avg. tensor] {mse_per_joint_unprocessed}")
        print(f"[MSE] [GT - Processed (VAE) avg. tensor] {mse_per_joint_processed}")

        # Calculate delta MSE (improvement or degradation after VAE processing)
        # delta_mse = mse_processed - mse_unprocessed
        # print(f"Delta MSE (VAE Processed - Unprocessed): {delta_mse}")

        # Compute average MSE for unprocessed and processed (VAE) tensors
        avg_mse_unprocessed = compute_average_mse(mse_per_joint_unprocessed)
        avg_mse_processed = compute_average_mse(mse_per_joint_processed)

        # Improvement (>0) or degradation (<0) after VAE processing
        delta_mse = avg_mse_unprocessed - avg_mse_processed
        print(f"Delta MSE (VAE Unrocessed - Processed): {delta_mse}")

        cnt += 1
        if cnt == 1:
            return


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

    test_npy()
