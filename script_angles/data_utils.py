import os
import numpy as np
import json


def find_min_max(data, path="../pkl_data/norm_data.json"):
    distances = data[:, 32:]

    min_distance = distances.min()
    max_distance = distances.max()

    print("min_distance", min_distance)
    print("max_distance", max_distance)

    norm_values = {
        "min_distance": min_distance,
        "max_distance": max_distance
    }

    with open(path, 'w') as f:
        json.dump(norm_values, f)

    return


def pkl_denorm(norm_data):
    norm_data_path = "../euler_data/norm_data.json"
    with open(norm_data_path, 'r') as f:
        min_max_data = json.load(f)

    min_dist = min_max_data['min_distance']
    max_dist = min_max_data['max_distance']

    out = []
    for sample in norm_data:
        # Split the sample into normalized angles and distances
        angles_normalized = sample[:32]
        distances_normalized = sample[32:]

        # Reverse angle normalization: Map [0, 1] -> [-1, 1] -> [-pi, pi]
        angles_original = (angles_normalized * 2 - 1) * np.pi

        # Reverse distance normalization: Map [0, 1] -> [min_dist, max_dist]
        distances_original = distances_normalized * (max_dist - min_dist) + min_dist

        # Combine reversed angles and distances
        reversed_sample = np.concatenate((angles_original, distances_original))
        out.append(reversed_sample)

    return np.array(out)


def pkl_norm(data):
    norm_data_path = "../euler_data/norm_data.json"
    if not os.path.exists(norm_data_path):
        find_min_max(np.array(data))

    with open(norm_data_path, 'r') as f:
        min_max_data = json.load(f)

    min_dist = min_max_data['min_distance']
    max_dist = min_max_data['max_distance']

    out = []
    # split sample_vector
    for sample in data:
        angles = sample[:32]
        distances = sample[32:]
        _ang_norm = angles / np.pi
        _ang_norm_map = (_ang_norm + 1) / 2
        distances_normalized = (distances - min_dist) / (max_dist - min_dist)
        normalized_vector = np.concatenate((_ang_norm_map, distances_normalized))
        out.append(normalized_vector)

    return out


def pkl_transform(data):
    conv_data = []
    for _sample in data:
        _orient_values = _sample['orientations'].values()
        _flattened_orient = [angle for pair in _sample['orientations'].values() for angle in pair]  # 32
        _distances_values = list(_sample['distances'].values())  # 16

        _orient_np = np.array(_flattened_orient)
        _distances_np = np.array(_distances_values)

        combined = np.concatenate((_orient_np, _distances_np))
        conv_data.append(combined)

    norm_data = pkl_norm(conv_data)

    return norm_data


def load_pkl(choice):
    if choice == 'train':
        path = "../euler_data/train/train_data.json"
    elif choice == 'val':
        path = "../euler_data/val/val_data.json"
    else:
        path = ""

    with open(path, 'r') as f:
        data = json.load(f)

    _euler_data = pkl_transform(data)

    return _euler_data

