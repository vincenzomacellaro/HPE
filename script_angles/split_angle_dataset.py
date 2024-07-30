import os

from human36_to_angles import load_data
from human36_to_angles import convert_to_angles


def save_angles_data(train_path, val_path, test_path):
    types = ["train", "val", "test"]
    target_path = ""

    for choice in types:
        if choice == "train":
            target_path = train_path
        elif choice == "val":
            target_path = val_path
        elif choice == "test":
            target_path = test_path

        data = load_data(choice)
        dicts = data["data"]
        convert_to_angles(dicts, target_path)

        return


if __name__ == '__main__':

    pos_data_path = "../data/train/"

    ang_train_path = "../angles_data/train/train_data.json"
    ang_val_path = "../angles_data/val/val_data.json"
    ang_test_path = "../angles_data/test/test_data.json"

    avg_pose_file = "../angles_json/avg_pose.json"
    global_scale_factor_file = "../angles_json/global_scale_factor.json"

    if not os.path.exists(pos_data_path):
        print("Positional data missing. Please run the \"script/split_dataset.py\" script first.")
    elif not os.path.exists(avg_pose_file):
        print("Average Pose file missing. Please run the \"get_avg_pose.py\" script first.")
    elif not os.path.exists(global_scale_factor_file):
        print("Global Scale Factor file missing. Please run the \"get_global_scale_factor.py\" script first.")
    elif not os.path.exists(ang_train_path):
        save_angles_data(ang_train_path, ang_val_path, ang_test_path)
        print("Correctly saved angles data.")