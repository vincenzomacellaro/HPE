import json
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# skeleton connections
skeleton = [
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11),
    (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)
]


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def extract_coordinates(data):
    action_dict = {}
    for action_id, actions in data.items():
        # print("action_id: " + str(action_id))
        sub_action_entries = {}
        for sub_action_id, sub_actions in actions.items():
            # print("sub_action_id: " + str(sub_action_id))
            frame_entries = {}  # key: id del frame, value, l'array delle 17x3 coordinate
            for frame_id, frames in sub_actions.items():
                joint_coords = {}
                # print("frame_id: " + str(frame_id))
                # print(len(sub_actions[frame_id]))
                # print(type(sub_actions[frame_id]))

                if type(sub_actions[frame_id]) == dict:
                    for key in sub_actions[frame_id].keys():
                        joint_coords[int(key)] = sub_actions[frame_id][key]
                else:
                    for joint in range(0, len(sub_actions[frame_id])):
                        joint_coords[int(joint)] = sub_actions[frame_id][joint]

                frame_entries[int(frame_id)] = joint_coords
            sub_action_entries[int(sub_action_id)] = frame_entries
        action_dict[int(action_id)] = sub_action_entries

    return action_dict


def get_coordinates(coordinates, subject_id, action_id, subaction_id):
    return coordinates.get(subject_id, {}).get(action_id, {}).get(subaction_id, {})


def get_frame_coordinates(coordinates, subject_id, action_id, subaction_id, frame_index):
    return coordinates.get(subject_id, {}).get(action_id, {}).get(subaction_id, {})[frame_index]



def extract_subject_number(file_path):
    match = re.search(r'subject(\d+)', file_path)
    if match:
        return int(match.group(1))
    return None


def extract_joints(joint_files_path):
    output = {}
    for file_path in joint_files_path:
        curr_file_data = load_json(file_path)
        coord_dict = extract_coordinates(curr_file_data)
        key = extract_subject_number(file_path)
        output[key] = coord_dict

    return output


def standardize_features(features, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
    standardized = (features - mean) / std
    return standardized, mean, std


def preprocess_data(frame):
    # Extract coordinates and convert them to a numpy array
    coords = np.array([frame[str(j)] for j in sorted(frame.keys(), key=int)])
    # Standardize coordinates
    standardized_coords, mean_coords, std_coords = standardize_features(coords)
    # Return flattened array of standardized coordinates
    return standardized_coords.flatten(), mean_coords, std_coords


def calculate_limb_lengths(frame, skeleton):
    limb_lengths = {}
    for (joint1, joint2) in skeleton:
        # Assuming frame is a dictionary with joint indices as keys and coordinates as values
        coord1 = np.array(frame[joint1])
        coord2 = np.array(frame[joint2])
        distance = np.linalg.norm(coord1 - coord2)
        limb_lengths[(joint1, joint2)] = distance
    return limb_lengths


def plot_joints_and_limbs(joint_coords, skeleton=skeleton, title="3D Joint Plot"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Reshape joint coordinates if necessary (assuming flat array of [x1, y1, z1, x2, y2, z2, ...])
    num_joints = joint_coords.shape[0] // 3
    reshaped_coords = joint_coords.reshape(num_joints, 3)

    # Plot joint positions
    ax.scatter(reshaped_coords[:, 0], reshaped_coords[:, 1], reshaped_coords[:, 2], c='red', label='Joints')

    # Initialize a variable to track if 'Limb' has been added to the legend
    limb_in_legend = False

    # Plot limbs
    for joint1, joint2 in skeleton:
        x = [reshaped_coords[joint1, 0], reshaped_coords[joint2, 0]]
        y = [reshaped_coords[joint1, 1], reshaped_coords[joint2, 1]]
        z = [reshaped_coords[joint1, 2], reshaped_coords[joint2, 2]]
        if not limb_in_legend:
            ax.plot(x, y, z, 'blue', label='Limb')
            limb_in_legend = True
        else:
            ax.plot(x, y, z, 'blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()


# def visualize_pose(original_frame, norm_frame):
#     frame_values = []
#     for entry in original_frame.values():
#         for value in entry:
#             frame_values.append(value)
#
#     print(f"Original values: {frame_values}")
#     print(f"Normalized values: {frame_values}")
#
#     plot_joints_and_limbs(np.array(frame_values), skeleton, title="Before Normalization")
#     plot_joints_and_limbs(norm_frame, skeleton, title="After Normalization")
#
#     print(" *** " + 3) # used to make the program crash at a given point


def pre_process_joints(data):
    global skeleton
    joints_data = []
    limb_lengths_accum = {pair: [] for pair in skeleton}  # Dictionary to store lengths per limb

    for dict_idx in data.keys():
        curr = data[dict_idx]
        for subject_id in curr.keys():
            curr_actions = curr[subject_id]
            for action_id in curr_actions.keys():
                curr_sub_actions = curr_actions[action_id]
                for sub_id in curr_sub_actions.keys():
                    curr_frames = curr_sub_actions[sub_id]
                    for frame_idx in curr_frames.keys():
                        frame = curr_frames[frame_idx]

                        frame_data, mean_coords, std_coords = preprocess_data(frame)  # Normalize
                        joints_data.append(frame_data)

                        # Use normalized data for limb length calculation to preserve proportionality
                        limb_lengths = calculate_limb_lengths(frame_data.reshape(-1, 3), skeleton)
                        for limb, length in limb_lengths.items():
                            limb_lengths_accum[limb].append(length)

    # Optionally, compute overall stats about limb lengths
    mean_limb_length = np.mean([np.mean(limb_lengths_accum[limb]) for limb in limb_lengths_accum])
    std_limb_length = np.std([np.mean(limb_lengths_accum[limb]) for limb in limb_lengths_accum])

    return joints_data, mean_limb_length, std_limb_length


def load_data(choice):
    train_path = "../data/train/train_data.json"
    val_path = "../data/val/val_data.json"
    test_path = "../data/test/test_data.json"

    if choice == "train":
        path = [train_path]
    elif choice == "val":
        path = [val_path]
    elif choice == "test":
        path = [test_path]
    else:
        return []

    raw_joints_data = extract_joints(path)
    norm_joints_data, mean_limb_lengths, std_limb_lengths = pre_process_joints(raw_joints_data)

    return norm_joints_data, mean_limb_lengths, std_limb_lengths




