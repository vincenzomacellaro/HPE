import numpy as np
import script_angles.mat_utils as utils
import matplotlib.pyplot as plt


# cambiare posizione di get_rotation_chain forse
def get_rotation_chain(joint, hierarchy, frame_rotations):
    hierarchy = hierarchy[::-1]

    # this code assumes ZXY rotation order
    R = np.eye(3)

    for parent in hierarchy:
        angles = frame_rotations[parent + "_angles"][0]
        _R = utils.get_R_z(angles[0]) @ utils.get_R_x(angles[1]) @ utils.get_R_y(angles[2])
        R = R @ _R

    return R


def plot_pose_from_dict(title, keypoints_dict, skeleton):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print(f"> plot pose from dict")
    print(f"keypoints_dict: \n{keypoints_dict}")
    print(f"skeleton: \n{skeleton}")

    # Create a list of keys that represent joints
    joint_keys = keypoints_dict['joints']

    # Create a list to store coordinates for easy indexing later
    keypoints_array = np.array([keypoints_dict[joint] for joint in joint_keys])
    # print(f"Plotting: \n{keypoints_array}")

    # Plot joints
    for i, (x, y, z) in enumerate(keypoints_array):
        ax.scatter(x, y, z, label=f'Joint {i} - {joint_keys[i]}')  # FOR DEBUG PURPOSES
        print(f"Joint {i} - Coordinates: {(x, y, z)}")
        # ax.scatter(x, y, z, label=f'Joint {i}')

    # Connections
    for start, end in skeleton:
        print(f"Connection: {start} - {end}")
        print(f"{keypoints_dict[start]} - {keypoints_dict[end]}")
        xs, ys, zs = zip(keypoints_dict[start], keypoints_dict[end])
        ax.plot(xs, ys, zs, 'r')

    # Setting labels and legend outside the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='center left', bbox_to_anchor=(-0.40, 0.5))
    ax.set_title(title)

    ax.azim = 90
    ax.elev = -85

    plt.show()


def plot_base_skeleton(kpts, skeleton):
    base_skeleton_dict = {'joints': []}

    for key in kpts['base_skeleton'].keys():
        base_skeleton_dict[key] = kpts['base_skeleton'][key]
        base_skeleton_dict['joints'].append(key)

    plot_pose_from_dict("T-Pose", base_skeleton_dict, skeleton)


def plot_pose_from_joint_angles(kpts, title, padding=2.5):
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

    # Set axis limits with increased padding
    ax.set_xlim([min(all_x) - padding, max(all_x) + padding])
    ax.set_ylim([min(all_y) - padding, max(all_y) + padding])
    ax.set_zlim([min(all_z) - padding, max(all_z) + padding])

    ax.azim = 90
    ax.elev = -85

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    plt.show()


def plot_pose_from_joint_angles_mod(ax, kpts, title, color='red'):
    # Initialize lists to store all coordinates for limits calculation
    all_x, all_y, all_z = [], [], []

    # Get a dictionary containing the rotations for the current frame
    frame_rotations = kpts['joint_angles']

    # For plotting
    for _j in kpts['joints']:
        if _j == 'hips':
            continue

        # Get hierarchy of how the joint connects back to root joint
        hierarchy = kpts['hierarchy'][_j]

        # Get the current position of the parent joint
        r1 = kpts['joint_positions']['hips'] / kpts['normalization']

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

        # Plot the joints on the provided axis
        ax.plot(xs=[r1[0], r2[0]], ys=[r1[1], r2[1]], zs=[r1[2], r2[2]], color=color)

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


def plot_subplots(kpts_list):
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), subplot_kw={'projection': '3d'})
    colors = ['red', 'green', 'blue']

    # Plot each pose in its individual subplot
    for i, (kpts, color) in enumerate(zip(kpts_list, colors)):
        title = ""
        if i == 0:
            title = "[GROUND TRUTH]"
        elif i == 1:
            title = "[AGGR. PRED][ORIGINAL]"
        else:
            title = "[AGGR. PRED][RECTIFIED]"

        plot_pose_from_joint_angles_mod(axes[i], kpts, title, color=color)


    plt.tight_layout()
    plt.show()

# def plot_pose_from_joint_angles_subplots(kpts_list, title):
#     fig = plt.figure(figsize=(25, 25))
#
#     num_poses = len(kpts_list)
#     num_cols = 4  # Set number of columns for subplots
#     num_rows = (num_poses + num_cols - 1) // num_cols  # Calculate number of rows required
#
#     for idx, kpts in enumerate(kpts_list):
#         ax = fig.add_subplot(num_rows, num_cols, idx + 1, projection='3d')
#
#         # Initialize lists to store all coordinates for limits calculation
#         all_x, all_y, all_z = [], [], []
#
#         # Get a dictionary containing the rotations for the current frame
#         frame_rotations = kpts['joint_angles']
#
#         # For plotting
#         for _j in kpts['joints']:
#             if _j == 'hips':
#                 continue
#
#             # Get hierarchy of how the joint connects back to root joint
#             hierarchy = kpts['hierarchy'][_j]
#
#             # Get the current position of the parent joint
#             r1 = np.array(kpts['joint_positions']['hips']) / kpts['normalization']
#
#             for parent in hierarchy:
#                 if parent == 'hips':
#                     continue
#
#                 R = get_rotation_chain(parent, kpts['hierarchy'][parent], frame_rotations)
#                 r1 = r1 + R @ np.array(kpts['base_skeleton'][parent])
#
#             r2 = r1 + get_rotation_chain(hierarchy[0], hierarchy, frame_rotations) @ kpts['base_skeleton'][_j]
#
#             # Append points for limits calculation
#             all_x.extend([r1[0], r2[0]])
#             all_y.extend([r1[1], r2[1]])
#             all_z.extend([r1[2], r2[2]])
#
#             ax.plot(xs=[r1[0], r2[0]], ys=[r1[1], r2[1]], zs=[r1[2], r2[2]], color='red')
#
#         # Set axis limits based on the min and max of each coordinate
#         ax.set_xlim([min(all_x) - 0.5, max(all_x) + 0.5])
#         ax.set_ylim([min(all_y) - 0.5, max(all_y) + 0.5])
#         ax.set_zlim([min(all_z) - 0.5, max(all_z) + 0.5])
#
#         ax.azim = 90
#         ax.elev = -85
#
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         ax.set_title(f'Pose {idx + 1}')
#
#     plt.suptitle(title, fontsize=30)
#     plt.show()