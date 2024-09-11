import numpy as np
import json

from script_angles.general_utils import load_json, to_numpy
from script_angles.human36_to_angles import (extract_coordinates,
                                             extract_subject_number,
                                             filter_samples,
                                             flatten_numeric_values)


angles_path_dict = {
    "train": "../angles_data/train/train_data.json",
    "val": "../angles_data/val/val_data.json",
    "test": "../angles_data/test/test_data.json"
}

pos_path_dict = {
    "train": ["../data/train/train_data.json"],
    "val": ["../data/val/val_data.json"],
    "test": ["../data/test/test_data.json"]
}


def load_ref_kpts(filename):
    # read keypoints from ref_kpts.dat file
    # 12 keypoints in ZXY order
    num_keypoints = 12
    fin = open(filename, 'r')

    kpts = []
    while True:
        line = fin.readline()
        if line == '':
            break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (num_keypoints, -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts


def extract_joints(joint_files_path, scale=True):
    output = {}
    for file_path in joint_files_path:
        curr_file_data = load_json(file_path)
        coord_dict, out = extract_coordinates(curr_file_data, scale)
        key = extract_subject_number(file_path)

        if key is None:
            key = "data"  # generic ahh key

        output[key] = out

    return output


def estimate_height_and_proportions(bone_lengths):
    # Define the vertical paths for height estimation
    leg_bones = ['lefthip', 'leftknee', 'leftfoot', 'righthip', 'rightknee', 'rightfoot']
    spine_bones = ['hips', 'neck']

    # Calculate leg lengths
    left_leg_length = sum(bone_lengths[bone] for bone in ['lefthip', 'leftknee', 'leftfoot'])
    right_leg_length = sum(bone_lengths[bone] for bone in ['righthip', 'rightknee', 'rightfoot'])

    # Choose the longer leg for height calculation
    max_leg_length = max(left_leg_length, right_leg_length)

    # Calculate spine length (hips to neck)
    spine_length = sum(bone_lengths.get(bone, 0) for bone in spine_bones)

    # Estimate total height
    estimated_height = max_leg_length + spine_length

    # Calculate limb proportions
    limb_proportions = {bone: length / estimated_height for bone, length in bone_lengths.items()}

    return estimated_height, limb_proportions


def load_data(choice, scale=True):
    path = pos_path_dict[choice]

    # qua dobbiamo leggere i pos_data_path
    raw_joints_data = extract_joints(path, scale)
    return raw_joints_data


def load_angles_data(choice):
    path = angles_path_dict[choice]

    joints_data = load_json(path)
    joints_data_np = []

    for sample in joints_data:

        f_sample = filter_samples(sample)                  # filters the complete sample to get only the needed data
        sample_np = to_numpy(f_sample)                     # converts to numpy values
        # flat_sample = flatten_numeric_values(sample_np)    # flattens numeric values extracted

        joints_data_np.append(sample_np)

    return joints_data_np


def calculate_global_scale_factors(data):
    # Initialize variables to track maximum values
    max_pos = 0
    max_angle = np.pi  # Since angles are in radians
    max_norm = 0

    # Iterate through the dataset
    for sample in data:

        # Update maximum position value
        current_max_pos = np.max([np.abs(sample['joint_positions'][joint]) for joint in sample['joint_positions']])
        max_pos = max(max_pos, current_max_pos)

        # Update maximum normalization value
        max_norm = max(max_norm, sample['normalization'])

    output = {
        "max_pos": max_pos,
        "max_angle": max_angle,
        "max_norm": max_norm  # Include this in the output
    }

    return output


def normalize_dataset(data, scale_factors):
    max_pos = float(scale_factors['max_pos'])
    max_angle = float(scale_factors['max_angle'])
    max_norm = float(scale_factors['max_norm'])

    normalized_dataset = []
    for sample in data:
        norm_sample = sample.copy()

        norm_sample['joint_positions'] = {}
        for joint, positions in sample['joint_positions'].items():
            # Ensure positions are numpy arrays
            positions = np.array(positions, dtype=float)  # Convert to numpy array if not already
            norm_sample['joint_positions'][joint] = positions / max_pos

        norm_sample['joint_angles'] = {}
        for joint, angles in sample['joint_angles'].items():
            # Ensure angles are numpy arrays
            angles = np.array(angles, dtype=float)
            norm_sample['joint_angles'][joint] = angles / max_angle

        norm_sample['normalization'] = sample['normalization'] / max_norm

        normalized_dataset.append(norm_sample)

    return normalized_dataset


def normalize_single_sample(sample, scale_factors):
    # Extract scale factors
    max_pos = float(scale_factors['max_pos'])
    max_angle = float(scale_factors['max_angle'])
    max_norm = float(scale_factors['max_norm'])

    # Copy the sample to avoid modifying the original
    norm_sample = sample.copy()

    # Normalize joint positions
    norm_sample['joint_positions'] = {}
    for joint, positions in sample['joint_positions'].items():
        # Ensure positions are numpy arrays
        positions = np.array(positions, dtype=float)  # Convert to numpy array if not already
        norm_sample['joint_positions'][joint] = positions / max_pos

    # Normalize joint angles
    norm_sample['joint_angles'] = {}
    for joint, angles in sample['joint_angles'].items():
        # Ensure angles are numpy arrays
        angles = np.array(angles, dtype=float)
        norm_sample['joint_angles'][joint] = angles / max_angle

    # Normalize the normalization factor
    norm_sample['normalization'] = sample['normalization'] / max_norm

    return norm_sample


def reverse_normalize_data(normalized_data, scale_factors):
    # Convert max_pos to float if it's a string
    max_pos = float(scale_factors['max_pos'])
    max_angle = float(scale_factors['max_angle'])
    max_norm = float(scale_factors['max_norm'])

    original_data = []

    for sample in normalized_data:
        orig_sample = sample.copy()  # Copy to revert to original
        # Reverse normalize positions
        for joint in sample['joint_positions']:
            orig_sample['joint_positions'][joint] *= max_pos

        # Reverse normalize angles
        for angle in sample['joint_angles']:
            orig_sample['joint_angles'][angle] *= max_angle

        for norm in sample['normalization']:
            orig_sample['normalization'][norm] *= max_norm

        original_data.append(orig_sample)

    return original_data


def reverse_normalize_sample(norm_sample, scale_factors):
    # Convert scale factors to float to ensure numerical operations are accurate
    max_pos = float(scale_factors['max_pos'])
    max_angle = float(scale_factors['max_angle'])
    max_norm = float(scale_factors['max_norm'])

    # Copy the normalized sample to avoid modifying the original input
    orig_sample = norm_sample.copy()

    # Reverse normalize joint positions
    orig_sample['joint_positions'] = {joint: positions * max_pos for joint, positions in norm_sample['joint_positions'].items()}

    # Reverse normalize joint angles
    orig_sample['joint_angles'] = {angle: angles * max_angle for angle, angles in norm_sample['joint_angles'].items()}

    # Reverse normalize the normalization factor
    orig_sample['normalization'] = norm_sample['normalization'] * max_norm

    return orig_sample


def flatten_dataset(data):
    output = []
    for dict_sample in data:
        output.append(flatten_numeric_values(dict_sample))

    return output


def load_data_for_train(choice, is_train=False):
    data = load_angles_data(choice)
    param_file = "../angles_json/scale_factors.json"

    if is_train:
        scale_factors = calculate_global_scale_factors(data)

        with open(param_file, 'w') as f:
            json.dump(scale_factors, f, indent=4)

        normalized_dataset = normalize_dataset(data, scale_factors)
        flattened_dataset = flatten_dataset(normalized_dataset)

        return flattened_dataset

    else:
        # param_file is supposed to be already existing in the project
        scale_factors = json.load(open(param_file))
        normalized_dataset = normalize_dataset(data, scale_factors)
        flattened_dataset = flatten_dataset(normalized_dataset)

        return flattened_dataset
