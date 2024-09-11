import json
import numpy as np
import os
import script_angles.mat_utils as utils

from script_angles.general_utils import from_numpy
from script_angles.align_poses import procrustes
from script_angles.get_avg_pose import get_avg_pose

avg_pose = None


def parse_and_save_keys(original_sample, sample_keys_file):
    if not os.path.exists(sample_keys_file):
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
        with open(sample_keys_file, 'w') as f:
            json.dump(keys, f, indent=4)

        print(f"File {sample_keys_file} created and saved.")
    return


def create_sample_keys_file(sample_keys_file):
    from load_data_utils import load_angles_data
    data = load_angles_data('test')
    sample = data[0]
    parse_and_save_keys(sample, sample_keys_file)
    return


def convert_to_dictionary(kpts):
    keypoints_to_index = {'lefthip': 6,
                          'leftknee': 8,
                          'leftfoot': 10,
                          'righthip': 7,
                          'rightknee': 9,
                          'rightfoot': 11,
                          'leftshoulder': 0,
                          'leftelbow': 2,
                          'leftwrist': 4,
                          'rightshoulder': 1,
                          'rightelbow': 3,
                          'rightwrist': 5}

    kpts_dict = {}
    for key, k_index in keypoints_to_index.items():
        # Access all frames for a specific joint, here ':' means all frames, adjust as needed
        # print(f"key: {key} - kpts[0][{k_index}]: {kpts[0][k_index]}")
        kpts_dict[key] = kpts[0][k_index]  # Adjusted to handle 3D data (frames, joints, coordinates)

    kpts_dict['joints'] = list(keypoints_to_index.keys())

    return kpts_dict


def add_hips_and_neck(kpts):
    # adding two new keypoints: mid point between the [hips] and mid point between the [shoulders]

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
                 'lefthip': ['hips'],
                 'leftknee': ['lefthip', 'hips'],
                 'leftfoot': ['leftknee', 'lefthip', 'hips'],
                 'righthip': ['hips'],
                 'rightknee': ['righthip', 'hips'],
                 'rightfoot': ['rightknee', 'righthip', 'hips'],
                 'neck': ['hips'],
                 'leftshoulder': ['neck', 'hips'],
                 'leftelbow': ['leftshoulder', 'neck', 'hips'],
                 'leftwrist': ['leftelbow', 'leftshoulder', 'neck', 'hips'],
                 'rightshoulder': ['neck', 'hips'],
                 'rightelbow': ['rightshoulder', 'neck', 'hips'],
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

    bone_lengths = {}
    for joint in kpts['joints']:
        if joint == 'hips':
            continue

        parent = kpts['hierarchy'][joint][0]  # Parent joint

        # Coordinates for the joint and its parent
        joint_kpts = kpts[joint]
        parent_kpts = kpts[parent]

        # Calculate the [bone vector] by subtracting the parent coordinates from the joint coordinates
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

    # print("Raw Bone Lengths:", kpts['bone_lengths'])
    # print("Offset Directions:", kpts['offset_directions'])
    # print("Base Skeleton Joint Coordinates:")
    # for joint, coords in kpts['base_skeleton'].items():
    #     print(f"{joint}: {coords}")

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


def convert_to_angles(dicts, target_path):
    serialized_data = []

    global avg_pose
    if avg_pose is None:
        avg_pose = get_avg_pose()

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
