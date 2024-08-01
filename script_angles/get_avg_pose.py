import numpy as np
import json
import os

from read_utils import read_keypoints


def compute_avg_pose(file):
    kpts = read_keypoints(file)
    average_pose = np.mean(kpts, axis=0)
    return average_pose


if __name__ == '__main__':
    # Calculates the average pose for all keypoints from the ref_kpts.dat dataset

    avg_pose_file = "../angles_json/avg_pose.json"
    if not os.path.exists(avg_pose_file):
        kpts_file = "../ref_data/ref_kpts.dat"
        avg_pose = compute_avg_pose(kpts_file)

        avg_pose_list = avg_pose.tolist()

        with open(avg_pose_file, 'w') as json_file:
            json.dump(avg_pose_list, json_file, indent=4)

        print(f"Average pose written to {avg_pose_file}")


