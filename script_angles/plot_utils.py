import numpy as np
import sys
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_pose(title, keypoints_array, skeleton):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print(f"Plotting - {title} \n{keypoints_array}")

    # Plot joints
    for i, (x, y, z) in enumerate(keypoints_array):
        ax.scatter(x, y, z, label=f'Joint {i}')

    # Plot connections
    for start, end in skeleton:
        xs, ys, zs = zip(keypoints_array[start], keypoints_array[end])
        ax.plot(xs, ys, zs, 'r')

    # Setting labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='center left', bbox_to_anchor=(-0.35, 0.5))
    ax.set_title(title)

    plt.show()


def plot_pose_from_dict(title, keypoints_dict, skeleton):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a list of keys that represent joints
    joint_keys = keypoints_dict['joints']

    # Create a list to store coordinates for easy indexing later
    keypoints_array = np.array([keypoints_dict[joint] for joint in joint_keys])
    # print(f"Plotting: \n{keypoints_array}")

    # Plot joints
    for i, (x, y, z) in enumerate(keypoints_array):
        ax.scatter(x, y, z, label=f'Joint {i} - {joint_keys[i]}')  # FOR DEBUG PURPOSES
        # print(f"Joint {i} - Coordinates: {(x, y, z)}")
        # ax.scatter(x, y, z, label=f'Joint {i}')

    # Connections
    for start, end in skeleton:
        xs, ys, zs = zip(keypoints_dict[start], keypoints_dict[end])
        ax.plot(xs, ys, zs, 'r')

    # Setting labels and legend outside the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='center left', bbox_to_anchor=(-0.40, 0.5))
    ax.set_title(title)

    plt.show()


def plot_base_skeleton(kpts, skeleton):
    base_skeleton_dict = {'joints': []}

    for key in kpts['base_skeleton'].keys():
        base_skeleton_dict[key] = kpts['base_skeleton'][key]
        base_skeleton_dict['joints'].append(key)

    plot_pose_from_dict("Default Skeleton", base_skeleton_dict, skeleton)


def get_rotation_chain(joint, hierarchy, frame_rotations):
    hierarchy = hierarchy[::-1]

    # this code assumes ZXY rotation order
    R = np.eye(3)

    for parent in hierarchy:
        angles = frame_rotations[parent + "_angles"][0]
        _R = utils.get_R_z(angles[0]) @ utils.get_R_x(angles[1]) @ utils.get_R_y(angles[2])
        R = R @ _R

    return R


def plot_pose_from_joint_angles(kpts, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Initialize lists to store all coordinates for limits calculation
    all_x, all_y, all_z = [], [], []

    #get a dictionary containing the rotations for the current frame
    frame_rotations = kpts['joint_angles']

    #for plotting
    for _j in kpts['joints']:
        if _j == 'hips':
            continue

        #get hierarchy of how the joint connects back to root joint
        hierarchy = kpts['hierarchy'][_j]

        #get the current position of the parent joint
        r1 = kpts['joint_positions']['hips']/kpts['normalization']

        for parent in hierarchy:
            if parent == 'hips':
                continue

            R = get_rotation_chain(parent, kpts['hierarchy'][parent], frame_rotations)
            r1 = r1 + R @ kpts['base_skeleton'][parent]
        r2 = r1 + get_rotation_chain(hierarchy[0], hierarchy, frame_rotations) @ kpts['base_skeleton'][_j]

        # Append points for limits calculation
        all_x.extend([r1[0], r2[0]])
        all_y.extend([r1[1], r2[1]])
        all_z.extend([r1[2], r2[2]])

        plt.plot(xs=[r1[0], r2[0]], ys=[r1[1], r2[1]], zs=[r1[2], r2[2]], color='red')

    # Set axis limits based on the min and max of each coordinate
    ax.set_xlim([min(all_x) - 0.5, max(all_x) + 0.5])
    ax.set_ylim([min(all_y) - 0.5, max(all_y) + 0.5])
    ax.set_zlim([min(all_z) - 0.5, max(all_z) + 0.5])

    ax.azim = 90
    ax.elev = -85

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.show()