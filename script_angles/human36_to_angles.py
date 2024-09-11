import re
import numpy as np
import os

from script_angles.general_utils import load_json
from script_angles.conversion_utils import create_sample_keys_file
from script_angles.get_global_scale_factor import load_global_scale_factor

avg_pose_file = "../angles_json/avg_pose.json"
global_scale_factor_file = "../angles_json/global_scale_factor.json"

scale_factor = None
avg_pose = None

# named_skeleton = [
#     ('leftshoulder', 'leftelbow'),  # Left Shoulder to Left Elbow
#     ('leftshoulder', 'rightshoulder'),  # Left Shoulder to Right Shoulder
#     ('lefthip', 'leftknee'),  # Left Hip to Left Knee
#     ('lefthip', 'righthip'),  # Left Hip to Right Hip
#     ('leftknee', 'leftfoot'),  # Left Knee to Left Ankle
#     ('rightshoulder', 'rightelbow'),  # Right Shoulder to Right Elbow
#     ('righthip', 'rightknee'),  # Right Hip to Right Knee
#     ('rightknee', 'rightfoot'),  # Right Knee to Right Ankle
#     ('leftelbow', 'leftwrist'),  # Left Elbow to Left Wrist
#     ('rightelbow', 'rightwrist')  # Right Elbow to Right Wrist
# ]

# named_skeleton_hips_neck = [
#     ('leftshoulder', 'leftelbow'),  # Left Shoulder to Left Elbow
#     ('leftshoulder', 'rightshoulder'),  # Left Shoulder to Right Shoulder
#     ('lefthip', 'leftknee'),  # Left Hip to Left Knee
#     ('lefthip', 'righthip'),  # Left Hip to Right Hip
#     ('leftknee', 'leftfoot'),  # Left Knee to Left Ankle
#     ('rightshoulder', 'rightelbow'),  # Right Shoulder to Right Elbow
#     ('righthip', 'rightknee'),  # Right Hip to Right Knee
#     ('rightknee', 'rightfoot'),  # Right Knee to Right Ankle
#     ('leftelbow', 'leftwrist'),  # Left Elbow to Left Wrist
#     ('rightelbow', 'rightwrist'),  # Right Elbow to Right Wrist
#     ('hips', 'neck')  # Hips Neck
# ]

# Mapping from current dataset joint indices to the required indices (1-based index)
joint_index_map = {
    15: [1, "Right Shoulder"],  # Right Shoulder
    16: [3, "Right Elbow"],  # Right Elbow
    17: [5, "Right Wrist"],  # Right Wrist
    12: [0, "Left Shoulder"],  # Left Shoulder
    13: [2, "Left Elbow"],  # Left Elbow
    14: [4, "Left Wrist"],  # Left Wrist
    2: [7, "Right Hip"],  # Right Hip
    3: [9, "Right Knee"],  # Right Knee
    4: [11, "Right Ankle"],  # Right Ankle
    5: [6, "Left Hip"],  # Left Hip
    6: [8, "Left Knee"],  # Left Knee
    7: [10, "Left Ankle"]  # Left Ankle
}


# skeleton_map = {
#     11: "Head",
#     10: "Nose",
#     15: "Right Shoulder",
#     16: "Right Elbow",
#     17: "Right Wrist",
#     12: "Left Shoulder",
#     13: "Left Elbow",
#     14: "Left Wrist",
#     1: "Center Hip",
#     2: "Right Hip",
#     3: "Right Knee",
#     4: "Right Ankle",
#     8: "Thorax",
#     5: "Left Hip",
#     6: "Left Knee",
#     7: "Left Ankle",
#     9: "Neck",
# }


# function that converts the 17-joints representation of the Human3.6M dataset into the 12-joints one
def extract_relevant_joints(data):
    frame_data = [None] * 36  # 12 joints x 3 coordinates

    global scale_factor
    if not scale_factor:
        scale_factor = load_global_scale_factor()
        # scale factor between human3.6M and joint_angles_calculate

    for idx, joint_coords in enumerate(data.items()):
        joint_idx = idx + 1
        if joint_idx in joint_index_map:
            # Calculate where to place these coordinates in the new list
            # e.g. -> Dict: 15: [1, "Right Shoulder"], new_idx extracts "1" from [1, "Right Shoulder"],

            new_idx = joint_index_map[joint_idx][0] * 3
            extracted_coords = joint_coords[1]
            scaled_array = [x * scale_factor for x in extracted_coords]
            frame_data[new_idx:new_idx + 3] = scaled_array

    return frame_data


def extract_relevant_joints_no_scale(data):
    frame_data = [None] * 36  # 12 joints x 3 coordinates

    for idx, joint_coords in enumerate(data.items()):
        joint_idx = idx + 1
        if joint_idx in joint_index_map:
            # Calculate where to place these coordinates in the new list
            # e.g. -> Dict: 15: [1, "Right Shoulder"], new_idx extracts "1" from [1, "Right Shoulder"],

            new_idx = joint_index_map[joint_idx][0] * 3
            extracted_coords = joint_coords[1]
            frame_data[new_idx:new_idx + 3] = extracted_coords

    return frame_data


def extract_coordinates(data, scale):
    out = []
    action_dict = {}
    for action_id, actions in data.items():
        sub_action_entries = {}
        for sub_action_id, sub_actions in actions.items():
            frame_entries = {}  # key: id del frame, value, array delle 17x3 coordinate -> 12 x 3 coordinate

            for frame_id, frames in sub_actions.items():
                joint_coords = {}

                if type(sub_actions[frame_id]) == dict:
                    for key in sub_actions[frame_id].keys():
                        frame_entry = sub_actions[frame_id][key]
                        if scale:
                            filtered_frame_entry = extract_relevant_joints(frame_entry)
                        else:
                            filtered_frame_entry = extract_relevant_joints_no_scale(frame_entry)

                        joint_coords[int(key)] = filtered_frame_entry
                        out.append(filtered_frame_entry)

                frame_entries[int(frame_id)] = joint_coords
            sub_action_entries[int(sub_action_id)] = frame_entries
        action_dict[int(action_id)] = sub_action_entries

    return action_dict, out


def extract_subject_number(file_path):
    match = re.search(r'subject(\d+)', file_path)
    if match:
        return int(match.group(1))
    return None


def filter_samples(original_sample):
    # function that filters the original sample to extract only the fields needed for printing
    keys = ['joint_angles', 'hierarchy', 'normalization']
    filtered_sample = {}

    for key in original_sample:

        if key == 'joint_positions':
            internal_key = 'hips'
            filtered_sample[key] = {}
            filtered_sample[key][internal_key] = original_sample[key][internal_key]
            continue

        if key == 'joint_angles' or key == 'base_skeleton':
            filtered_sample[key] = {}
            for internal_key in original_sample[key]:

                values = np.array(original_sample[key][internal_key])
                non_zero_values = values[np.nonzero(values)]

                if len(non_zero_values) > 0:
                    filtered_sample[key][internal_key] = non_zero_values
            continue

        if key in keys:
            filtered_sample[key] = original_sample[key]

    return filtered_sample


def flatten_numeric_values(obj):
    """
    Recursively extracts all numeric values from nested dictionaries and lists into a flat numpy array.
    Logs the keys of dictionaries and indices of lists/arrays being processed.
    """
    result = []

    def recurse(item, path="root"):
        if isinstance(item, dict):
            for key, value in item.items():
                recurse(value, path=f"{path}/{key}")
        elif isinstance(item, (list, np.ndarray)):
            for idx, value in enumerate(item):
                recurse(value, path=f"{path}[{idx}]")
        elif isinstance(item, (int, float, complex)):
            result.append(item)

    recurse(obj)
    return np.array(result)


def map_to_base_skeleton(values):
    dict = {}

    sk_map = {
        'lefthip': [1.0, 0., 0.],
        'leftknee': [0., 1.0, 0.],
        'leftfoot': [0., 1.0, 0.],
        'righthip': [1.0, 0., 0.],
        'rightknee': [0., 1.0, 0.],
        'rightfoot': [0., 1.0, 0.],
        'leftshoulder': [1.0, 0., 0.],
        'leftelbow': [1.0, 0., 0.],
        'leftwrist': [1.0, 0., 0.],
        'rightshoulder': [1.0, 0., 0.],
        'rightelbow': [1.0, 0., 0.],
        'rightwrist': [1.0, 0., 0.],
        'neck': [0., 1., 0.],
        'hips': [0., 0., 0.],
    }

    sk_list = list(sk_map.keys())
    for idx, v in enumerate(values):
        sk_map_key = sk_list[idx]

        sk_map_values = np.array(sk_map[sk_list[idx]])
        dict[sk_map_key] = v * sk_map_values

    dict['hips'] = np.array([0.0, 0.0, 0.0])

    return dict


def reconstruct_from_array(flat_array, file="../angles_json/sample_keys.json"):
    # Helper function to reconstruct numpy array from flat numpy array.
    if not os.path.exists(file):
        create_sample_keys_file(file)

    end_points = ["leftfoot_angles", "rightfoot_angles", "leftwrist_angles", "rightwrist_angles"]

    template = {
            "joints": {},
            "joint_positions": {},
            "joint_angles": {},
            "hierarchy": {},
            "root_joint": {},
            "base_skeleton": {},
            "normalization": {}
        }

    # keys from file
    data_dicts = load_json(file)
    flat_idx = 0
    sample = template.copy()

    for temp_key in template:

        if temp_key == 'joints':
            sample[temp_key] = data_dicts["joint_positions_keys"]
            continue

        if temp_key == 'joint_positions':
            sample[temp_key]['hips'] = flat_array[flat_idx: flat_idx + 3]
            flat_idx += 3
            continue

        if temp_key == 'bone_lengths':
            curr_key = temp_key + '_keys'
            for pos_key in data_dicts[curr_key]:
                sample[temp_key][pos_key] = {}
                sample[temp_key][pos_key] = float(flat_array[flat_idx])
                flat_idx += 1
            continue

        if temp_key == 'hierarchy':
            sample[temp_key] = {}
            for entry in data_dicts[temp_key]:
                sample[temp_key][entry] = np.array(data_dicts[temp_key][entry])
            continue

        if temp_key == 'root_joint':
            sample[temp_key] = 'hips'
            continue

        if temp_key == 'normalization':
            sample[temp_key] = flat_array[-1]
            continue

        if temp_key == "base_skeleton":
            values_to_map = flat_array[flat_idx: flat_idx + 13]

            sample[temp_key] = {}
            sample[temp_key] = map_to_base_skeleton(values_to_map)

            flat_idx += 13

        if temp_key == "joint_angles":
            data_key = temp_key + "_keys"

            for entry in data_dicts[data_key]:
                sample[temp_key][entry] = {}
                if entry in end_points:
                    sample[temp_key][entry] = np.array([0.0, 0.0, 0.0])
                    # no need to update the flat_idx in this case as we are "creating" values for the end_points.
                else:
                    sample[temp_key][entry] = (flat_array[flat_idx: flat_idx + 3]).reshape(1, 3)
                    flat_idx += 3

    return sample

# def test_script():
#     # tests the entire script with a single sample taken from the 'train', 'val' OR 'test' script
#     data = load_angles_data('test')
#     sample = data[0]
#     plot_pose_from_joint_angles(sample, "3D plot from original sample")
#
#     sample_keys_file = "../angles_json/sample_keys.json"
#
#     # [parse_and_save_keys] used to save the template structure for the samples reconstruction
#     # it saves a new sample_keys.json file only if not existent
#     parse_and_save_keys(sample, sample_keys_file)
#     flat_sample = flatten_numeric_values(sample)
#
#     reconstructed_data = reconstruct_from_array(flat_sample)
#     plot_pose_from_joint_angles(reconstructed_data, "3D plot from [reconstructed] sample")
#
#
# if __name__ == '__main__':
#     test_script()
