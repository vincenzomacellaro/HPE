import os

from load_data_utils import load_data, load_angles_data
from conversion_utils import convert_to_angles, parse_and_save_keys
from get_global_scale_factor import load_global_scale_factor
from get_avg_pose import get_avg_pose


def save_angles_data(train_path, val_path, test_path):
    # this function does not return anything, it only writes [data] to file
    types = ["train", "val", "test"]
    target_path = ""

    for choice in types:
        if choice == "train":
            target_path = train_path
        elif choice == "val":
            target_path = val_path
        elif choice == "test":
            target_path = test_path

        data = load_data("pos", choice)
        dicts = data["data"]
        convert_to_angles(dicts, target_path)
        return


def save_sample_keys_file(sample_keys_file):
    data = load_angles_data('test')
    sample = data[0]
    parse_and_save_keys(sample, sample_keys_file)
    return


if __name__ == '__main__':

    pos_data_path = "../data/train/"

    ang_train_path = "../angles_data/train/train_data.json"
    ang_val_path = "../angles_data/val/val_data.json"
    ang_test_path = "../angles_data/test/test_data.json"

    if not os.path.exists(pos_data_path):
        print("Positional data missing. Please run the \"script/split_dataset.py\" script first.")

    if not os.path.exists(ang_train_path):
        save_angles_data(ang_train_path, ang_val_path, ang_test_path)
        print("Angular dataset correctly created and saved to file")
        # get_avg_pose
        global_scale_factor_file = "../angles_json/global_scale_factor.json"
        if not (os.path.exists(global_scale_factor_file)):
            load_global_scale_factor()
            print(" Global scale factor correctly created and saved to file")

        avg_pose_file = "../angles_json/avg_pose.json"
        if not (os.path.exists(avg_pose_file)):
            get_avg_pose(avg_pose_file)

        sample_keys_file = "../angles_json/sample_keys.json"
        if not (os.path.exists(sample_keys_file)):
            save_sample_keys_file(sample_keys_file)

