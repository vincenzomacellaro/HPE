import json
import numpy as np

from load_data_utils import load_data_for_train


def explore_bone_lengths(data):
    bone_lengths = {}

    # Initialize the bone_lengths dict with lists for each joint
    sample_keys = list(data[0]['bone_lengths'].keys())
    for key in sample_keys:
        bone_lengths[key] = []

    # Collect bone_lengths from all samples in the dataset
    for sample in data:
        for key in sample['bone_lengths']:
            bone_lengths[key].append(sample['bone_lengths'][key])

    # Calculate min, max, and average for each joint
    stats = {}
    for key in bone_lengths:
        stats[key] = {
            'min': np.min(bone_lengths[key]),
            'max': np.max(bone_lengths[key]),
            'avg': np.mean(bone_lengths[key])
        }

    bone_lengths_file = "../angles_json/bone_lengths_stats.json"

    # Save stats to JSON file
    with open(bone_lengths_file, 'w') as f:
        json.dump(stats, f, indent=4)

    return stats


bone_lengths_dict = {'lefthip': 0.0,
                     'leftknee': 0.0,
                     'leftfoot': 0.0,
                     'righthip': 0.0,
                     'rightknee': 0.0,
                     'rightfoot': 0.0,
                     'leftshoulder': 0.0,
                     'leftelbow': 0.0,
                     'leftwrist': 0.0,
                     'rightshoulder': 0.0,
                     'rightelbow': 0.0,
                     'rightwrist': 0.0,
                     'neck': 0.0}


def num_to_dict(data):
    out = []
    for sample in data:
        sample_dict = {}
        for idx, key in enumerate(bone_lengths_dict.keys()):
            sample_dict[key] = sample[idx]

        out.append({'bone_lengths': sample_dict})

    return out


# Example usage
_, physique_data = load_data_for_train("train")
dict_samples = num_to_dict(physique_data)
stats = explore_bone_lengths(dict_samples)
print(f"Bone length stats: {stats}")
