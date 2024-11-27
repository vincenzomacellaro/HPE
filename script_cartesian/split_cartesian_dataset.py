import json
import os
import joblib
import numpy as np


human36_joints = {
    'pelvis': 0,
    'righthip': 1,
    'rightknee': 2,
    'rightfoot': 3,
    'lefthip': 4,
    'leftknee': 5,
    'leftfoot': 6,
    'spine1': 7,
    'neck': 8,
    'head': 9,
    'site': 10,
    'leftshoulder': 11,
    'leftelbow': 12,
    'leftwrist': 13,
    'rightshoulder': 14,
    'rightelbow': 15,
    'rightwrist': 16
}


human36_bones = [
    ('pelvis', 'spine1'),
    ('spine1', 'neck'),
    ('neck', 'head'),
    ('neck', 'leftshoulder'),
    ('neck', 'rightshoulder'),
    ('leftshoulder', 'leftelbow'),
    ('leftelbow', 'leftwrist'),
    ('rightshoulder', 'rightelbow'),
    ('rightelbow', 'rightwrist'),
    ('pelvis', 'lefthip'),
    ('lefthip', 'leftknee'),
    ('leftknee', 'leftfoot'),
    ('pelvis', 'righthip'),
    ('righthip', 'rightknee'),
    ('rightknee', 'rightfoot'),
    ('head', 'site')
]


def compute_mean_std(dataset, norm_data_file="../cartesian_data/norm_data.json"):
    # Normalizes the dataset to zero mean and unit variance.
    if not os.path.exists(norm_data_file):

        mean = np.mean(dataset, axis=0)
        std = np.std(dataset, axis=0)

        stats = {"mean": mean.tolist(), "std": std.tolist()}
        with open(norm_data_file, "w") as f:
            json.dump(stats, f)

    else:
        with open(norm_data_file, 'r') as f:
            norm_data = json.load(f)
        mean = norm_data['mean']
        std = norm_data['std']

    return mean, std


def normalize_bone_lengths(relative_pose):
    # Normalizes each relative vector (bone) to unit length.
    normalized_pose = relative_pose / np.linalg.norm(relative_pose, axis=1, keepdims=True)
    return normalized_pose


def compute_relative_coordinates(pose, bones, joints):
    # pose shape (17, 3)
    relative_coordinates = []
    for parent, child in bones:
        parent_idx = joints[parent]
        child_idx = joints[child]
        relative_coordinates.append(pose[child_idx] - pose[parent_idx])
    return np.array(relative_coordinates)


def compute_bone_angles(relative_pose_normalized):
    # Computes angles between consecutive bones in the skeleton.
    angles = []
    for i in range(len(relative_pose_normalized) - 1):  # Assuming sequential bone list
        u, v = relative_pose_normalized[i], relative_pose_normalized[i + 1]
        cos_theta = np.clip(np.dot(u, v), -1.0, 1.0)  # Clip to avoid numerical issues
        angle = np.arccos(cos_theta)  # Angle in radians
        angles.append(angle)

    return np.array(angles)


def preprocess(choice, pose, bones, joints):
    # Step 1: Compute relative joint coordinates
    relative_pose = compute_relative_coordinates(pose, bones, joints)

    if choice == 'stats':
        return relative_pose

    # Step 2: Compute and store original bone lengths
    bone_lengths = np.linalg.norm(relative_pose, axis=1)

    # Step 2: Normalize bone lengths
    relative_pose_normalized = normalize_bone_lengths(relative_pose)

    # Step 3: Compute bone angles (skip normalization here)
    angles = compute_bone_angles(relative_pose_normalized)

    # Step 4: Flatten and concatenate
    flattened_pose = relative_pose_normalized.flatten()
    processed_pose = np.concatenate([flattened_pose, angles, bone_lengths])  # Include bone lengths

    return processed_pose


def load_pkl(choice):
    pkl_path = "../pkl/"
    pkl_train = "h36m_train.pkl"
    pkl_val = "h36m_validation.pkl"

    if choice == "train":
        data = joblib.load(pkl_path + pkl_train)
        save_path = "../cartesian_data/train/train_data.json"
        print(f" {choice} samples: {len(data)}")

    elif choice == "val":
        data = joblib.load(pkl_path + pkl_val)
        save_path = "../cartesian_data/val/val_data.json"
        print(f" {choice} samples: {len(data)}")
    else:
        # computing data stats
        data = joblib.load(pkl_path + pkl_train)
        stats_data = []
        for sample in data:
            relative_pose = preprocess(choice, sample['joints_3d'], human36_bones, human36_joints)
            pos_flattened = relative_pose.reshape(-1)  # (45,)
            stats_data.append(pos_flattened)

        _, _ = compute_mean_std(stats_data)
        return

    _cartesian_data = []
    for sample in data:
        _pose = preprocess(choice, sample['joints_3d'], human36_bones, human36_joints)
        _cartesian_data.append(_pose.tolist())

    with open(save_path, 'w') as f:
        json.dump(_cartesian_data, f)

    return


if __name__ == '__main__':

    # choice: "train", "val", "stats"

    # load_pkl("train")
    load_pkl("val")
    # load_pkl("stats")

