import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import torch

# skeleton connections
skeleton = [
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11),
    (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)
]

annotations_path = "../Human3.6/annotations/"
prefix = "Human36M_subject"
suffix = "_joint_3d.json"

train_path = "../data/train/train_data.json"
val_path = "../data/val/val_data.json"
test_path = "../data/test/test_data.json"


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Create a function to plot and save a single frame
def plot_frame(joints_dict):
    # Extract the x, y, z coordinates from the dictionary
    x_coords = [coords[0] for coords in joints_dict.values()]
    y_coords = [coords[1] for coords in joints_dict.values()]
    z_coords = [coords[2] for coords in joints_dict.values()]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')

    # Plot the skeleton connections
    for (i, j) in skeleton:
        ax.plot(
            [joints_dict[i][0], joints_dict[j][0]],
            [joints_dict[i][1], joints_dict[j][1]],
            [joints_dict[i][2], joints_dict[j][2]],
            'r-'
        )

    # Add labels to the points
    for joint, coords in joints_dict.items():
        ax.text(coords[0], coords[1], coords[2], str(joint), color='red')

    # Set labels for axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Render the plot to a numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


# Main function to create video from sequence of frames
def create_video(coordinates, action_id, subaction_id, video_path='output_video_01.mp4', fps=50):
    # Extract the frames
    frames = get_coordinates(coordinates, action_id, subaction_id)

    # Initialize video writer
    height, width = None, None
    video = None

    for frame_idx, joints_dict in frames.items():
        img = plot_frame(joints_dict)

        if video is None:
            height, width, _ = img.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the lower case
            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Write the frame to the video
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Release the video writer object
    video.release()
    print(f'Video saved as {video_path}')


# funziona bene per i dati raw, perché cazzo non funziona per gli altri
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


def find_joint_files(directory_path):
    output = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if '_joint_3d' in file:
                output.append(os.path.join(root, file))

    return output


def print_dict_info(path, data):
    print(f"\n{path}")
    print(f" [{len(data.keys())}] subjects: {data.keys()}")
    for subject_id in data.keys():
        print(f"\t subject_id: [{subject_id}] - [{len(data[subject_id].keys())}] actions")
        for action_id in data[subject_id].keys():
            print(f"\t action_id: [{action_id}] - [{len(data[subject_id][action_id].keys())}] sub-actions")
            for sub_action_id in data[subject_id][action_id].keys():
                print(f"\t\t [{sub_action_id}] - {len(data[subject_id][action_id][sub_action_id].keys())} frames")


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
        # print_dict_info(file_path, coord_dict)

    return output


def calculate_distances(frame):
    global skeleton
    distances = []

    for joint1_idx, joint2_idx in skeleton:
        coord1 = np.array(frame[str(joint1_idx)])
        coord2 = np.array(frame[str(joint2_idx)])
        distance = np.linalg.norm(coord1 - coord2)
        distances.append(distance)

    return distances


# Normalize the angles
def normalize_angles(angles, min_angle, max_angle):
    return [(a - min_angle) / (max_angle - min_angle) for a in angles]


# Function to normalize data to the range [0, 1]
def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


def standardize_data(data):
    """ Standardize data to have zero mean and unit variance. """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data, mean, std


def standardize_features(features, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
    standardized = (features - mean) / std
    return standardized, mean, std


def calculate_distances(coords, skeleton):
    distances = []
    for joint1_idx, joint2_idx in skeleton:
        point1 = coords[joint1_idx]
        point2 = coords[joint2_idx]
        distances.append(np.linalg.norm(point1 - point2))
    return np.array(distances)


def preprocess_data(frame):
    """ Process frame data to standardize coordinates. """
    # Extract coordinates and convert them to a numpy array
    coords = np.array([frame[str(j)] for j in sorted(frame.keys(), key=int)])
    # Standardize coordinates
    standardized_coords, mean_coords, std_coords = standardize_features(coords)
    # Return flattened array of standardized coordinates
    return standardized_coords.flatten(), mean_coords, std_coords
# def process_frame(frame):
#     # Calculate relative coordinates (example: relative to the 'neck' joint)
#     neck_coords = np.array(frame['9'])
#     relative_coords = {joint: np.array(coords) - neck_coords for joint, coords in frame.items()}
#
#     # Calculate distances
#     distances = calculate_distances(relative_coords)
#
#     # Convert dict to array and standardize coordinates and distances separately
#     coords_array = np.concatenate([relative_coords[joint].flatten() for joint in sorted(relative_coords.keys())])
#     standardized_coords, _, _ = standardize_data(coords_array)
#
#     standardized_distances, _, _ = standardize_data(np.array(distances))
#
#     # Combine standardized coordinates and distances
#     full_frame_data = np.concatenate((standardized_coords, standardized_distances))
#
#     return full_frame_data


def calculate_limb_lengths(frame, skeleton):
    limb_lengths = {}
    for (joint1, joint2) in skeleton:
        # Assuming frame is a dictionary with joint indices as keys and coordinates as values
        coord1 = np.array(frame[joint1])
        coord2 = np.array(frame[joint2])
        distance = np.linalg.norm(coord1 - coord2)
        limb_lengths[(joint1, joint2)] = distance
    return limb_lengths


def pre_process_joints(data):
    global skeleton
    joints_data = []
    limb_lengths_accum = {pair: [] for pair in skeleton}  # Dictionary to store lengths per limb

    for dict_idx in data.keys():
        curr = data[dict_idx]
        for subject_id in curr.keys():
            # print(f"{subject_id}")
            curr_actions = curr[subject_id]
            for action_id in curr_actions.keys():
                # print(f"\t{action_id}")
                curr_sub_actions = curr_actions[action_id]
                for sub_id in curr_sub_actions.keys():
                    curr_frames = curr_sub_actions[sub_id]
                    # print(f"\t\t{sub_id} - {len(curr_frames)} frames")
                    for frame_idx in curr_frames.keys():
                        frame = curr_frames[frame_idx]

                        frame_data, mean_coords, std_coords = preprocess_data(frame)

                        # Calculate limb lengths for the current frame from raw data
                        limb_lengths = calculate_limb_lengths(frame_data, skeleton)
                        for limb, length in limb_lengths.items():
                            limb_lengths_accum[limb].append(length)

                        # Normalize the coordinates after limb lengths are calculated
                        # frame_data, mean_coords, std_coords = preprocess_data(frame)
                        joints_data.append(frame_data)

    # Calculate mean lengths for each limb
    expected_lengths = {limb: np.mean(lengths) for limb, lengths in limb_lengths_accum.items()}

    return joints_data, expected_lengths


def load_data(choice):
    if choice == "train":
        path = [train_path]
    elif choice == "val":
        path = [val_path]
    elif choice == "test":
        path = [test_path]
    else:
        return []

    raw_joints_data = extract_joints(path)
    # train_samples, expected_lengths = pre_process_joints(train_data, skeleton)
    norm_joints_data, expected_lengths = pre_process_joints(raw_joints_data)

    return norm_joints_data, expected_lengths


def split_data(data, train_ratio=0.7, val_ratio=0.15):
    train_data = {}
    val_data = {}
    test_data = {}

    for subject_id, subjects in data.items():
        # print(" *** subject: " + str(subject_id))

        train_data[subject_id] = {}
        val_data[subject_id] = {}
        test_data[subject_id] = {}

        for action_id, actions in data[subject_id].items():
            # print(" *** action: " + str(action_id))

            train_data[subject_id][action_id] = {}
            val_data[subject_id][action_id] = {}
            test_data[subject_id][action_id] = {}

            for sub_action_id, sub_actions in data[subject_id][action_id].items():
                # print(" *** sub_action_id: " + str(sub_action_id))

                frame_ids = list(data[subject_id][action_id][sub_action_id].keys())
                random.shuffle(frame_ids)

                train_end = int(len(frame_ids) * train_ratio)
                val_end = train_end + int(len(frame_ids) * val_ratio)

                train_frames = frame_ids[:train_end]
                val_frames = frame_ids[train_end:val_end]
                test_frames = frame_ids[val_end:]

                train_data[subject_id][action_id][sub_action_id] = {fid: sub_actions[fid] for fid in train_frames}
                val_data[subject_id][action_id][sub_action_id] = {fid: sub_actions[fid] for fid in val_frames}
                test_data[subject_id][action_id][sub_action_id] = {fid: sub_actions[fid] for fid in test_frames}

                # print(len(train_data[subject_id][action_id][sub_action_id]))
                # print(len(val_data[subject_id][action_id][sub_action_id]))
                # print(len(test_data[subject_id][action_id][sub_action_id]))

    # print(len(train_data))
    print_dict_info("TRAIN", train_data)
    # print(len(val_data))
    print_dict_info("VAL", val_data)
    # print(len(test_data))
    print_dict_info("TEST", test_data)

    return train_data, val_data, test_data


def save_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)


def load_data_for_train():
    # load and split raw data in [train][val][test]
    joint_files_path = find_joint_files(annotations_path)
    raw_joints_data = extract_joints(joint_files_path)
    train_data, val_data, test_data = split_data(raw_joints_data)

    # saving the freshly-split data to file
    save_data(train_data, train_path)
    save_data(val_data, val_path)
    save_data(test_data, test_path)


# load_data_for_train()
def stats(path):
    data = extract_joints(path)
    print(type(data))
    print(f"{path}")
    for dict_id in data.keys():
        # print(f"\t subject_id: [{subject_id}] - [{len(data[subject_id].keys())}] actions")
        for subject_id in data[dict_id].keys():
            print(f"subject_id: [{subject_id}] - [{len(data[dict_id][subject_id].keys())}] actions")
            for action_id in data[dict_id][subject_id].keys():
                print(f"\t action_id: [{action_id}] - sub_actions: {len(data[dict_id][subject_id][action_id].keys())} ")
                for frame_id in data[dict_id][subject_id][action_id]:
                    print(f"\t\t frames: {len(data[dict_id][subject_id][action_id][frame_id].keys())}")

# search_path = find_joint_files(annotations_path)
# search_path = ["../data/train/train_data.json"]
# stats(search_path)

# load_data_for_train()
