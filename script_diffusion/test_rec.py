import numpy as np
import json
import os
import torch
import random

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
from script_angles import mat_utils as utils

from script_angles.plot_utils import plot_pose_from_joint_angles
from script_angles.load_data_utils import load_json, normalize_single_sample, reverse_normalize_sample


avg_pose = None


def get_avg_pose(avg_pose_file="../angles_json/avg_pose.json"):
    with open(avg_pose_file, "r") as f:
        avg_pose = json.load(f)
        np_avg_pose = np.array(avg_pose)
        return np_avg_pose


def flatten_array(sample):
    f_sample = filter_samples(sample)   # filters the complete sample to get only the needed data
    sample_np = to_numpy(f_sample)      # converts to numpy values
    flat_sample = flatten_numeric_values(sample_np)
    return flat_sample


def reconstruct(vae, sample):
    # Ensure sample is a torch.Tensor with shape [1, 47]
    if not isinstance(sample, torch.Tensor):
        sample = torch.tensor(sample, dtype=torch.float32)

    if sample.dim() == 1:
        # Adding batch dimension if it's not there
        sample = sample.unsqueeze(0)

    encoded, decoded, _, _ = vae(sample)

    decoded_detached = decoded.detach()
    decoded_numpy = decoded_detached.cpu().numpy()
    rec_decoded = reconstruct_from_array(decoded_numpy[0], sample_keys_file)

    return rec_decoded


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
    sample_dict['joints'] = [
        "lefthip", "leftknee", "leftfoot",
        "righthip", "rightknee", "rightfoot",
        "leftshoulder", "leftelbow", "leftwrist",
        "rightshoulder", "rightelbow", "rightwrist",
        "hips", "neck"
    ]

    return sample_dict


def prepare_for_input(raw_sample):
    raw_sample_np = to_numpy(raw_sample)  # converts to numpy values
    filtered_sample = filter_samples(raw_sample_np)  # filters the full sample to get the needed data only
    norm_sample = normalize_single_sample(filtered_sample, scale_factors)  # normalizes single sample
    num_sample = flatten_numeric_values(norm_sample)  # ex-flattened sample

    return num_sample


def prepare_for_plotting(num_sample):
    rec_sample = reconstruct_from_array(num_sample, sample_keys_file)
    den_sample = reverse_normalize_sample(rec_sample, scale_factors)

    return den_sample


def test_rec():

    sample_path = "../angles_data/test/test_data.json"
    joints_data = load_json(sample_path)

    rand_idx = random.randint(0, len(joints_data) - 1)

    raw_sample = joints_data[rand_idx]  # raw_sample = dict

    in_sample = prepare_for_input(raw_sample)
    out_sample = prepare_for_plotting(in_sample)
    plot_pose_from_joint_angles(out_sample, "[REC] sample")
    return


if __name__ == '__main__':
    # debugging purposes only, it checks if the norm. and reverse norm. operations work as expected

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

    test_rec()

    # this script loads the [../angles_data/test/test_data.json] dataset, then a random sample from it, and
    # checks if the reconstruction process works as expected

