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


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def extract_coordinates(data):
    action_dict = {}
    for action_id, actions in data.items():
        sub_action_entries = {}
        for sub_action_id, sub_actions in actions.items():
            frame_entries = {}  # key: frame id - value: 17x3 coordinates array
            for frame_id, frames in sub_actions.items():
                joint_coords = {}

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

    return output


def split_data(data, train_ratio=0.7, val_ratio=0.15):
    train_data = {}
    val_data = {}
    test_data = {}

    for subject_id, subjects in data.items():

        train_data[subject_id] = {}
        val_data[subject_id] = {}
        test_data[subject_id] = {}

        for action_id, actions in data[subject_id].items():

            train_data[subject_id][action_id] = {}
            val_data[subject_id][action_id] = {}
            test_data[subject_id][action_id] = {}

            for sub_action_id, sub_actions in data[subject_id][action_id].items():

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

    print_dict_info("TRAIN", train_data)
    print_dict_info("VAL", val_data)
    print_dict_info("TEST", test_data)

    return train_data, val_data, test_data


def save_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)


def load_data_for_train(train_path, val_path, test_path):
    # load and split raw data in [train][val][test]
    joint_files_path = find_joint_files(annotations_path)
    raw_joints_data = extract_joints(joint_files_path)
    train_data, val_data, test_data = split_data(raw_joints_data)

    # saving the freshly-split data to file
    save_data(train_data, train_path)
    save_data(val_data, val_path)
    save_data(test_data, test_path)


if __name__ == '__main__':

    annotations_path = "../Human3.6/annotations/"
    prefix = "Human36M_subject"
    suffix = "_joint_3d.json"

    if not os.path.exists(annotations_path):
        print("Human3.6M annotations missing. Please download Human3.6M annotations first. ")
    else:
        train_path = "../data/train/train_data.json"
        val_path = "../data/val/val_data.json"
        test_path = "../data/test/test_data.json"

        if not os.path.exists(train_path):
            load_data_for_train(train_path, val_path, test_path)