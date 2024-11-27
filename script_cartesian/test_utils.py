import numpy as np
import json
import torch
import os
import matplotlib.pyplot as plt

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


def plot_human36m_pose(sample):
    """
    Plots a single Human3.6M pose sample given a (1, 17, 3) numpy array.

    Parameters:
    - sample: np.ndarray with shape (1, 17, 3), containing 3D coordinates of 17 joints
    """
    # Ensure sample is of shape (17, 3) by removing extra dimensions
    joints = sample.reshape(17, 3)

    # Extract X, Y, Z coordinates
    x_vals = joints[:, 0]
    y_vals = joints[:, 1]
    z_vals = joints[:, 2]

    # Plot setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_vals, y_vals, z_vals, color='blue', label="Joints")

    # Draw connections to form the skeleton
    for joint1, joint2 in human36_bones:
        idx1 = human36_joints[joint1]
        idx2 = human36_joints[joint2]

        x_coords = [x_vals[idx1], x_vals[idx2]]
        y_coords = [y_vals[idx1], y_vals[idx2]]
        z_coords = [z_vals[idx1], z_vals[idx2]]

        ax.plot(x_coords, y_coords, z_coords, color='black')

    # Setting labels and plot limits for better visualization
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Pose Plot (Human3.6M)")
    ax.legend()
    plt.show()


def reconstruct_pose(processed_pose, bones, joints):
    # Reconstructs the original XYZ joint positions from the processed input pose.

    # Step 1: Split the processed pose
    relative_coords_normalized = processed_pose[:48]  # First 16*3 elements
    bone_lengths = processed_pose[-16:]  # Last 16 elements (stored bone lengths)

    # Step 2: Undo normalization of relative coordinates
    relative_coords_normalized = relative_coords_normalized.reshape(16, 3)
    relative_coords = relative_coords_normalized * bone_lengths[:, np.newaxis]  # Scale by bone lengths

    # Step 3: Reconstruct absolute joint positions
    absolute_positions = np.zeros((17, 3))  # Initialize joint positions (17 joints
    bone_to_idx = {bone: i for i, bone in enumerate(bones)}

    for parent, child in bones:
        parent_idx = joints[parent]
        child_idx = joints[child]

        bone_idx = bone_to_idx[(parent, child)]

        absolute_positions[child_idx] = (
                absolute_positions[parent_idx] + relative_coords[bone_idx]
        )

    return absolute_positions


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


def load_stats(file="../cartesian_data/norm_data.json"):
    with open(file) as f:
        data = json.load(f)

    mean = data['mean']
    std = data['std']

    return mean, std


def preprocess(pose):
    bones = human36_bones
    joints = human36_joints

    # Step 1: Compute relative joint coordinates
    relative_pose = compute_relative_coordinates(pose, bones, joints)

    # Step 2: Compute and store original bone lengths
    bone_lengths = np.linalg.norm(relative_pose, axis=1)

    # Step 3: Normalize bone lengths
    relative_pose_normalized = normalize_bone_lengths(relative_pose)

    # Step 4: Compute bone angles
    angles = compute_bone_angles(relative_pose_normalized)

    # Step 5: Flatten and concatenate
    flattened_pose = relative_pose_normalized.flatten()
    processed_pose = np.concatenate([flattened_pose, angles, bone_lengths])  # Include bone lengths

    return processed_pose


def postprocess(pose):
    # mean, std = load_stats()
    rec_pose = reconstruct_pose(pose, human36_bones, human36_joints)
    return rec_pose


def load_diffusion_data():
    gt_path = '../diffusion_data/gt.npy'
    pred_path = '../diffusion_data/predictions_ours_20hyp.npy'

    ground_truth = np.load(gt_path)
    predictions = np.load(pred_path)

    # Ground Truth file (shape): (543344, 1, 17, 3)
    # Pred file (shape): (543344, 20, 17, 3)

    return ground_truth, predictions


def compute_relative_coordinates_batch(pose_batch, bones, joints):
    """
    Compute relative joint coordinates for a batch of poses.
    """
    batch_size = pose_batch.shape[0]
    relative_coordinates = []

    for parent, child in bones:
        parent_idx = joints[parent]
        child_idx = joints[child]

        # Compute relative coordinates for each bone in the batch
        relative_coordinates.append(pose_batch[:, child_idx] - pose_batch[:, parent_idx])

    return np.stack(relative_coordinates, axis=1)  # Shape: (batch_size, num_bones, 3)


def normalize_bone_lengths_batch(relative_pose_batch):
    """
    Normalize bone lengths for a batch of relative joint coordinates.
    """
    norms = np.linalg.norm(relative_pose_batch, axis=2, keepdims=True)  # Shape: (batch_size, num_bones, 1)
    normalized_pose_batch = relative_pose_batch / (norms + 1e-8)  # Avoid division by zero
    return normalized_pose_batch


def compute_bone_angles_batch(relative_pose_normalized_batch):
    """
    Compute angles between consecutive bones for a batch of normalized relative poses.
    """
    angles_batch = []
    num_bones = relative_pose_normalized_batch.shape[1]

    for i in range(num_bones - 1):
        u = relative_pose_normalized_batch[:, i]
        v = relative_pose_normalized_batch[:, i + 1]

        # Compute cosine of angles using dot products
        cos_theta = np.clip(np.sum(u * v, axis=1), -1.0, 1.0)  # Shape: (batch_size,)
        angles = np.arccos(cos_theta)  # Shape: (batch_size,)
        angles_batch.append(angles)

    return np.stack(angles_batch, axis=1)  # Shape: (batch_size, num_angles)


def preprocess_batch(poses):
    bones = human36_bones
    joints = human36_joints

    relative_pose_batch = compute_relative_coordinates_batch(poses, bones, joints)
    bone_lengths_batch = np.linalg.norm(relative_pose_batch, axis=2)
    relative_pose_normalized_batch = normalize_bone_lengths_batch(relative_pose_batch)
    angles_batch = compute_bone_angles_batch(relative_pose_normalized_batch)

    flattened_pose_batch = relative_pose_normalized_batch.reshape(relative_pose_normalized_batch.shape[0], -1)
    processed_pose_batch = np.concatenate([flattened_pose_batch, angles_batch, bone_lengths_batch], axis=1)

    return torch.from_numpy(processed_pose_batch)


def postprocess_batch(poses):
    batch_size = poses.shape[0]

    # Step 1: Split processed pose into relative coords and bone lengths
    relative_coords_normalized = poses[:, :48]  # [batch_size, 48]
    bone_lengths = poses[:, -16:]  # [batch_size, 16]

    # Step 2: Undo normalization (vectorized for the entire batch)
    relative_coords_normalized = relative_coords_normalized.reshape(batch_size, 16, 3)  # [batch_size, 16, 3]
    relative_coords = relative_coords_normalized * bone_lengths.unsqueeze(-1)  # [batch_size, 16, 3]

    # Step 3: Reconstruct absolute joint positions
    absolute_positions = torch.zeros((batch_size, 17, 3), device=poses.device)  # [batch_size, 17, 3]

    bone_to_idx = {bone: i for i, bone in enumerate(human36_bones)}

    for parent, child in human36_bones:
        parent_idx = human36_joints[parent]
        child_idx = human36_joints[child]

        bone_idx = bone_to_idx[(parent, child)]

        # Add relative coordinates to parent's absolute position
        absolute_positions[:, child_idx, :] = (
                absolute_positions[:, parent_idx, :] + relative_coords[:, bone_idx, :]
        )

    return absolute_positions



