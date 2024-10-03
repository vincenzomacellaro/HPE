import numpy as np
import json

from script_angles.align_poses import procrustes
from script_angles import mat_utils as utils
from script_angles.conversion_utils import convert_to_dictionary
from script_angles.conversion_utils import add_hips_and_neck
from script_angles.conversion_utils import get_bone_lengths
from script_angles.conversion_utils import get_base_skeleton
from script_angles.conversion_utils import calculate_joint_angles
from script_angles.conversion_utils import build_enhanced_sample
from script_angles.human36_to_angles import reconstruct_from_array
from script_angles.load_data_utils import de_normalize_sample


avg_pose = None


def reconstruct_sample(pose_sample, physique_sample, scale_factors):
    combined_sample = np.concatenate((pose_sample, physique_sample), axis=0)
    sample_reconstructed = reconstruct_from_array(combined_sample)
    sample_denorm = de_normalize_sample(sample_reconstructed, scale_factors)
    return sample_denorm


def get_avg_pose(avg_pose_file="../angles_json/avg_pose.json"):
    with open(avg_pose_file, "r") as f:
        avg_pose = json.load(f)
        np_avg_pose = np.array(avg_pose)
        return np_avg_pose


def convert_to_angles(sample):
    # input: <numerical> representation of a sample in the dataset;
    # output: <dict> representation of the same sample in the dataset;

    global avg_pose
    if avg_pose is None:
        avg_pose = get_avg_pose()

    frame_kpts = np.array(sample).reshape(1, 12, 3)
    oriented_pose = procrustes(frame_kpts[0], avg_pose).reshape(1, 12, 3)

    R = utils.get_R_z(np.pi / 2)
    for kpt_num in range(oriented_pose.shape[1]):
        oriented_pose[0, kpt_num] = R @ oriented_pose[0, kpt_num]

    kpts = convert_to_dictionary(oriented_pose)
    add_hips_and_neck(kpts)
    get_bone_lengths(kpts)
    get_base_skeleton(kpts)

    calculate_joint_angles(kpts)

    # Unpack the tuple returned by build_enhanced_sample
    pose_sample, physique_sample = build_enhanced_sample(kpts)
    # Combine pose and physique data if needed for further processing
    sample_dict = {**pose_sample, **physique_sample}

    # sample_dict = from_numpy(en_sample)
    sample_dict['joints'] = ["lefthip", "leftknee", "leftfoot",
                             "righthip", "rightknee", "rightfoot",
                             "leftshoulder", "leftelbow", "leftwrist",
                             "rightshoulder", "rightelbow", "rightwrist",
                             "hips", "neck"]

    return sample_dict
