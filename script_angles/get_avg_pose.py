import numpy as np
import json
import os


def compute_avg_pose(file):
    from load_data_utils import load_ref_kpts
    kpts = load_ref_kpts(file)
    average_pose = np.mean(kpts, axis=0)
    return average_pose


def get_avg_pose(avg_pose_file="../angles_json/avg_pose.json"):
    kpts_file = "../ref_data/ref_kpts.dat"
    if not os.path.exists(avg_pose_file):
        avg_pose = compute_avg_pose(kpts_file)
        avg_pose_list = avg_pose.tolist()

        with open(avg_pose_file, 'w') as json_file:
            json.dump(avg_pose_list, json_file, indent=4)

        print(f"Average pose written to {avg_pose_file}")
        return avg_pose
    else:
        with open(avg_pose_file, "r") as f:
            avg_pose = json.load(f)
            np_avg_pose = np.array(avg_pose)
            return np_avg_pose


if __name__ == '__main__':
    # Calculates the average pose for all keypoints from the ref_kpts.dat dataset
    # avg_pose_file = "../angles_json/avg_pose.json"
    get_avg_pose()



