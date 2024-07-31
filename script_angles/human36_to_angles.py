import json
import re
import numpy as np
import os
import utils

from align_poses import procrustes
from plot_utils import plot_pose_from_joint_angles

avg_pose_file = "../angles_json/avg_pose.json"
global_scale_factor_file = "../angles_json/global_scale_factor.json"

angles_path_dict = {
    "train": "../angles_data/train/train_data.json",
    "val": "../angles_data/val/val_data.json",
    "test": "../angles_data/test/test_data.json"
}

pos_path_dict = {
    "train": "../data/train/train_data.json",
    "val": "../data/val/val_data.json",
    "test": "../data/test/test_data.json"
}

scale_factor = None
avg_pose = None

named_skeleton = [
    ('leftshoulder', 'leftelbow'),  # Left Shoulder to Left Elbow
    ('leftshoulder', 'rightshoulder'),  # Left Shoulder to Right Shoulder
    ('lefthip', 'leftknee'),  # Left Hip to Left Knee
    ('lefthip', 'righthip'),  # Left Hip to Right Hip
    ('leftknee', 'leftfoot'),  # Left Knee to Left Ankle
    ('rightshoulder', 'rightelbow'),  # Right Shoulder to Right Elbow
    ('righthip', 'rightknee'),  # Right Hip to Right Knee
    ('rightknee', 'rightfoot'),  # Right Knee to Right Ankle
    ('leftelbow', 'leftwrist'),  # Left Elbow to Left Wrist
    ('rightelbow', 'rightwrist')  # Right Elbow to Right Wrist
]

named_skeleton_hips_neck = [
    ('leftshoulder', 'leftelbow'),  # Left Shoulder to Left Elbow
    ('leftshoulder', 'rightshoulder'),  # Left Shoulder to Right Shoulder
    ('lefthip', 'leftknee'),  # Left Hip to Left Knee
    ('lefthip', 'righthip'),  # Left Hip to Right Hip
    ('leftknee', 'leftfoot'),  # Left Knee to Left Ankle
    ('rightshoulder', 'rightelbow'),  # Right Shoulder to Right Elbow
    ('righthip', 'rightknee'),  # Right Hip to Right Knee
    ('rightknee', 'rightfoot'),  # Right Knee to Right Ankle
    ('leftelbow', 'leftwrist'),  # Left Elbow to Left Wrist
    ('rightelbow', 'rightwrist'),  # Right Elbow to Right Wrist
    ('hips', 'neck')  # Hips Neck
]

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


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# function that converts the 17-joints representation of the Human3.6M dataset into the 12-joints one
def extract_relevant_joints(data):
    frame_data = [None] * 36  # 12 joints x 3 coordinates

    global scale_factor
    if not scale_factor:
        with open(global_scale_factor_file, "r") as f:
            scale_factor = float(json.load(f))

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
        # print("action_id: " + str(action_id))
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
                        # print(filtered_frame_entry)
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


def extract_joints(joint_files_path, scale=True):
    output = {}
    for file_path in joint_files_path:
        curr_file_data = load_json(file_path)
        coord_dict, out = extract_coordinates(curr_file_data, scale)
        key = extract_subject_number(file_path)

        if key == None:
            key = "data"  # generic ahh key
        # output[key] = coord_dict
        output[key] = out

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


def load_data(choice, scale=True):
    # choice = "train", "val", "test" -> pesca direttamente dal dict
    path = angles_path_dict[choice]
    raw_joints_data = extract_joints(path, scale)

    return raw_joints_data


def convert_to_angles(dicts, target_path):
    serialized_data = []

    global avg_pose
    if not avg_pose:
        # Read the JSON file
        with open(avg_pose_file, 'r') as json_file:
            avg_pose_list = json.load(json_file)

        # Convert the list back to a numpy array
        avg_pose = np.array(avg_pose_list)

    for joints_array in dicts:
        frame_kpts = np.array(joints_array).reshape(1, 12, 3)
        oriented_pose = procrustes(frame_kpts[0], avg_pose).reshape(1, 12, 3)

        R = utils.get_R_z(np.pi / 2)
        for kpt_num in range(oriented_pose.shape[1]):
            oriented_pose[0, kpt_num] = R @ oriented_pose[0, kpt_num]

        kpts = convert_to_dictionary(oriented_pose)
        add_hips_and_neck(kpts)
        get_bone_lengths(kpts)
        get_base_skeleton(kpts)

        calculate_joint_angles(kpts)
        en_sample = build_enhanced_sample(kpts)
        json_data = from_numpy(en_sample)

        serialized_data.append(json_data)

        # Serialize to JSON and write to a file
    with open(target_path, 'w') as f:
        json.dump(serialized_data, f, indent=4)


def filter_samples(original_sample):
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
                    # 'hips': array([0., 0., 0.]) -> to be added during reconstruction

            continue

        # else
        if key in keys:
            filtered_sample[key] = original_sample[key]

    return filtered_sample


def load_angles_data(choice):
    path = angles_path_dict[choice]

    joints_data = load_json(path)
    joints_data_np = []

    for sample in joints_data:

        f_sample = filter_samples(sample)                # filters the complete sample to get only the needed data from it
        sample_np = to_numpy(f_sample)                   # converts to numpy values
        flat_sample = flatten_numeric_values(sample_np)  # flattens the numeric values extracted in the previous sample

        joints_data_np.append(flat_sample)

    return joints_data_np


def max_abs_scaling(data):
    max_abs = np.max(np.abs(data), axis=0)
    max_abs[max_abs == 0] = 1  # to avoid division by zero
    scaled_data = data / max_abs
    return scaled_data, max_abs


def save_parameters(scale_params, filepath):
    with(open(filepath, 'w')) as f:
        json.dump({'scale_params': scale_params.tolist()}, f)


def apply_scale(data, scale_params):
    return data / scale_params


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


def convert_to_dictionary(kpts):
    #its easier to manipulate keypoints by joint name
    keypoints_to_index = {'lefthip': 6, 'leftknee': 8, 'leftfoot': 10,
                          'righthip': 7, 'rightknee': 9, 'rightfoot': 11,
                          'leftshoulder': 0, 'leftelbow': 2, 'leftwrist': 4,
                          'rightshoulder': 1, 'rightelbow': 3, 'rightwrist': 5}

    kpts_dict = {}
    for key, k_index in keypoints_to_index.items():
        # Access all frames for a specific joint, here ':' means all frames, adjust as needed
        # print(f"key: {key} - kpts[0][{k_index}]: {kpts[0][k_index]}")
        kpts_dict[key] = kpts[0][k_index]  # Adjusted to handle 3D data (frames, joints, coordinates)
        # print(f"{key} - {k_index} -> {kpts_dict[key]}")

    kpts_dict['joints'] = list(keypoints_to_index.keys())

    return kpts_dict


def add_hips_and_neck(kpts):
    #we add two new keypoints which are the mid point between the hips and mid point between the shoulders

    #add hips kpts
    difference = kpts['lefthip'] - kpts['righthip']
    difference = difference / 2
    hips = kpts['righthip'] + difference
    kpts['hips'] = hips
    kpts['joints'].append('hips')

    #add neck kpts
    difference = kpts['leftshoulder'] - kpts['rightshoulder']
    difference = difference / 2
    neck = kpts['rightshoulder'] + difference
    kpts['neck'] = neck
    kpts['joints'].append('neck')

    #define the hierarchy of the joints
    hierarchy = {'hips': [],
                 'lefthip': ['hips'], 'leftknee': ['lefthip', 'hips'], 'leftfoot': ['leftknee', 'lefthip', 'hips'],
                 'righthip': ['hips'], 'rightknee': ['righthip', 'hips'],
                 'rightfoot': ['rightknee', 'righthip', 'hips'],
                 'neck': ['hips'],
                 'leftshoulder': ['neck', 'hips'], 'leftelbow': ['leftshoulder', 'neck', 'hips'],
                 'leftwrist': ['leftelbow', 'leftshoulder', 'neck', 'hips'],
                 'rightshoulder': ['neck', 'hips'], 'rightelbow': ['rightshoulder', 'neck', 'hips'],
                 'rightwrist': ['rightelbow', 'rightshoulder', 'neck', 'hips']
                 }

    kpts['hierarchy'] = hierarchy
    kpts['root_joint'] = 'hips'

    return kpts


def get_bone_lengths(kpts):
    # Calculate the length of each bone from data in a single frame.
    #
    # We have to define an initial skeleton pose(T pose).
    # In this case we need to know the length of each bone.
    # Here we calculate the length of each bone from data
    #
    # >> modified version of the original get_bone_lengths function

    bone_lengths = {}
    for joint in kpts['joints']:
        if joint == 'hips':
            continue

        parent = kpts['hierarchy'][joint][0]  # Parent joint

        # Coordinates for the joint and its parent
        joint_kpts = kpts[joint]
        parent_kpts = kpts[parent]

        # Calculate the bone vector by subtracting the parent coordinates from the joint coordinates
        _bone = joint_kpts - parent_kpts

        # Calculate the length of the bone (Euclidean distance)
        _bone_length = np.sqrt(np.sum(np.square(_bone)))

        # Store the bone length
        bone_lengths[joint] = _bone_length

    # Optional: Store or return the bone lengths
    kpts['bone_lengths'] = bone_lengths
    return


# we define the T-pose and normalize it by the length of the [hips to neck] distance
def get_base_skeleton(kpts, normalization_bone='neck'):
    # this defines a generic skeleton to which we can apply rotations to
    body_lengths = kpts['bone_lengths']

    #define skeleton offset directions
    offset_directions = {}

    offset_directions['lefthip'] = np.array([1, 0, 0])
    offset_directions['leftknee'] = np.array([0, -1, 0])
    offset_directions['leftfoot'] = np.array([0, -1, 0])

    offset_directions['righthip'] = np.array([-1, 0, 0])
    offset_directions['rightknee'] = np.array([0, -1, 0])
    offset_directions['rightfoot'] = np.array([0, -1, 0])

    offset_directions['neck'] = np.array([0, 1, 0])

    offset_directions['leftshoulder'] = np.array([1, 0, 0])
    offset_directions['leftelbow'] = np.array([1, 0, 0])
    offset_directions['leftwrist'] = np.array([1, 0, 0])

    offset_directions['rightshoulder'] = np.array([-1, 0, 0])
    offset_directions['rightelbow'] = np.array([-1, 0, 0])
    offset_directions['rightwrist'] = np.array([-1, 0, 0])

    #set bone normalization length; set this value to 1 if you don't want normalization
    normalization = kpts['bone_lengths'][normalization_bone]
    # normalization = 1

    # base skeleton set by multiplying offset directions by measured bone lengths.
    # in this case we use the average of two-sided limbs, e.g left and right hip averaged
    base_skeleton = {'hips': np.array([0.0, 0.0, 0.0])}

    def _set_length(joint_type):
        base_skeleton['left' + joint_type] = offset_directions['left' + joint_type] * (
                (body_lengths['left' + joint_type] + body_lengths['right' + joint_type]) / (2 * normalization))
        base_skeleton['right' + joint_type] = offset_directions['right' + joint_type] * (
                (body_lengths['left' + joint_type] + body_lengths['right' + joint_type]) / (2 * normalization))

    _set_length('hip')
    _set_length('knee')
    _set_length('foot')
    _set_length('shoulder')
    _set_length('elbow')
    _set_length('wrist')
    base_skeleton['neck'] = offset_directions['neck'] * (body_lengths['neck'] / normalization)

    kpts['offset_directions'] = offset_directions
    kpts['base_skeleton'] = base_skeleton
    kpts['normalization'] = normalization

    return


# calculate rotation matrix and joint angles of the input joint
def get_joint_rotations(joint_name, joints_hierarchy, joints_offsets, frame_rotations, frame_pos):
    _invR = np.eye(3)
    for i, parent_name in enumerate(joints_hierarchy[joint_name]):
        if i == 0:
            continue

        _r_angles = frame_rotations[parent_name]
        R = utils.get_R_z(_r_angles[0]) @ utils.get_R_x(_r_angles[1]) @ utils.get_R_y(_r_angles[2])
        _invR = _invR @ R.T

    b = _invR @ (frame_pos[joint_name] - frame_pos[joints_hierarchy[joint_name][0]])
    _R = utils.Get_R2(joints_offsets[joint_name], b)
    tz, ty, tx = utils.Decompose_R_ZXY(_R)

    joint_rs = np.array([tz, tx, ty])

    return joint_rs


def calculate_joint_angles(kpts):
    #set up emtpy container for joint angles
    for joint in kpts['joints']:
        kpts[joint + '_angles'] = []

    for framenum in range(1):

        #get the keypoints positions in the current frame
        frame_pos = {}
        for joint in kpts['joints']:
            frame_pos[joint] = kpts[joint]

        frame_pos['joints'] = kpts['joints']

        root_position, root_rotation = get_hips_position_and_rotation(frame_pos)

        frame_rotations = {'hips': root_rotation}

        # center the body pose
        for joint in kpts['joints']:
            frame_pos[joint] = frame_pos[joint] - root_position

        # get the max joints connections
        max_connected_joints = 0
        for joint in kpts['joints']:
            if len(kpts['hierarchy'][joint]) > max_connected_joints:
                max_connected_joints = len(kpts['hierarchy'][joint])

        depth = 2

        while depth <= max_connected_joints:

            for joint in kpts['joints']:
                if len(kpts['hierarchy'][joint]) == depth:
                    joint_rs = get_joint_rotations(joint,
                                                   kpts['hierarchy'],
                                                   kpts['offset_directions'],
                                                   frame_rotations,
                                                   frame_pos)
                    parent = kpts['hierarchy'][joint][0]
                    frame_rotations[parent] = joint_rs

            depth += 1

        # add zero rotation angles for endpoints. [not necessary as they are never used]
        for _j in kpts['joints']:
            if _j not in list(frame_rotations.keys()):
                frame_rotations[_j] = np.array([0., 0., 0.])

        # update dictionary with current angles.
        for joint in kpts['joints']:
            kpts[joint + '_angles'].append(frame_rotations[joint])

    #convert joint angles list to numpy arrays.
    for joint in kpts['joints']:
        kpts[joint + '_angles'] = np.array(kpts[joint + '_angles'])

    return


#calculate the rotation of the root joint with respect to the world coordinates
def get_hips_position_and_rotation(frame_pos, root_joint='hips', root_define_joints=['lefthip', 'neck']):
    #root position is saved directly
    root_position = frame_pos[root_joint]

    #calculate unit vectors of root joint
    root_u = frame_pos[root_define_joints[0]] - frame_pos[root_joint]
    root_u = root_u / np.sqrt(np.sum(np.square(root_u)))
    root_v = frame_pos[root_define_joints[1]] - frame_pos[root_joint]
    root_v = root_v / np.sqrt(np.sum(np.square(root_v)))
    root_w = np.cross(root_u, root_v)

    # build the rotation matrix
    C = np.array([root_u, root_v, root_w]).T
    thetaz, thetay, thetax = utils.Decompose_R_ZXY(C)
    root_rotation = np.array([thetaz, thetax, thetay])

    return root_position, root_rotation


def build_enhanced_sample(kpts):
    sample = {"joint_positions": {},
              "joint_angles": {},
              "bone_lengths": {},
              "hierarchy": {},
              "root_joint": {},
              "base_skeleton": {},
              "normalization": {}
              }

    joints_list = kpts['joints']
    for key in joints_list:
        sample["joint_positions"][key] = kpts[key]
        sample["joint_angles"][key + "_angles"] = kpts[key + "_angles"]
        sample["hierarchy"][key] = kpts["hierarchy"][key]
        sample["base_skeleton"][key] = kpts["base_skeleton"][key]

    bone_lenghts_list = kpts['bone_lengths']
    for key in bone_lenghts_list:
        sample["bone_lengths"][key] = kpts["bone_lengths"][key]

    sample["root_joint"] = kpts["root_joint"]
    sample["normalization"] = kpts["normalization"]

    return sample


def from_numpy(obj):
    # Recursively convert numpy arrays in a dictionary to lists for JSON serialization.
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: from_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [from_numpy(item) for item in obj]
    else:
        return obj


def to_numpy(obj):
    # Recursively convert lists in a dictionary or list back to numpy arrays.
    if isinstance(obj, list):  # Checks if it is a list
        try:
            return np.array(obj)
        except:  # If conversion to np.array fails, process as a nested list
            return [to_numpy(item) for item in obj]
    elif isinstance(obj, dict):  # Recursive case for dictionaries
        return {key: to_numpy(value) for key, value in obj.items()}
    else:
        return obj


# Function to parse the original sample
def parse_and_save_keys(original_sample, filename):
    if not os.path.exists(filename):
        # Extract the keys from the original sample
        joint_positions_keys = list(original_sample['joint_positions'].keys())
        joint_angles_keys = list(original_sample['joint_angles'].keys())
        bone_lengths_keys = list(original_sample['bone_lengths'].keys())
        base_skeleton_keys = list(original_sample['base_skeleton'].keys())
        hierarchy = {key: list(original_sample['hierarchy'][key]) for key in original_sample['hierarchy']}

        # Create the keys dictionary
        keys = {
            "joint_positions_keys": joint_positions_keys,
            "joint_angles_keys": joint_angles_keys,
            "bone_lengths_keys": bone_lengths_keys,
            "base_skeleton_keys": base_skeleton_keys,
            "hierarchy": hierarchy
        }

        # Save the keys dictionary to a JSON file
        with open(filename, 'w') as f:
            json.dump(keys, f, indent=4)

        print(f"File {filename} created and saved.")


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
            # map to base skeleton
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
                    # non serve aggiornare il flat_idx in questo caso
                else:
                    sample[temp_key][entry] = (flat_array[flat_idx: flat_idx + 3]).reshape(1, 3)
                    flat_idx += 3
            # aggiungere gli end_points

    return sample


def test_script():
    # tests the entire script with a single sample taken from the 'train', 'val' or 'test' script
    data = load_angles_data('test')
    sample = data[0]

    plot_pose_from_joint_angles(sample, "3D plot from original sample")

    sample_keys_file = '../angles_json/sample_keys.json'

    # [parse_and_save_keys] used to save the template structure for the samples reconstruction
    parse_and_save_keys(sample, sample_keys_file)
    flat_sample = flatten_numeric_values(sample)

    reconstructed_data = reconstruct_from_array(flat_sample, sample_keys_file)
    plot_pose_from_joint_angles(reconstructed_data, "3D plot from [reconstructed] sample")


if __name__ == '__main__':
    test_script()
