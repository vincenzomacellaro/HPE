import numpy as np
import json

from general_utils import load_json, to_numpy
from human36_to_angles import (extract_coordinates, extract_subject_number, filter_samples, flatten_numeric_values,
                               max_abs_scaling, apply_scale)


angles_path_dict = {
    "train": ["../angles_data/train/train_data.json"],
    "val": ["../angles_data/val/val_data.json"],
    "test": ["../angles_data/test/test_data.json"]
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
    while (True):
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

        f_sample = filter_samples(sample)                # filters the complete sample to get only the needed data
        sample_np = to_numpy(f_sample)                   # converts to numpy values
        flat_sample = flatten_numeric_values(sample_np)  # flattens the numeric values extracted in the previous sample

        joints_data_np.append(flat_sample)

    return joints_data_np


def save_parameters(scale_params, filepath):
    with(open(filepath, 'w')) as f:
        json.dump({'scale_params': scale_params.tolist()}, f)


def load_data_for_train(choice, is_train=False):
    param_file = "../angles_json/data_parameters.json"
    data = np.array(load_angles_data(choice))

    if is_train:
        normalized_data, scale_params = max_abs_scaling(data)
        save_parameters(scale_params, param_file)
        return normalized_data
    else:
        scale_params = json.load(open(param_file))
        scale_params_numpy = np.array(scale_params["scale_params"])
        normalized_data = apply_scale(data, scale_params_numpy)
        return normalized_data