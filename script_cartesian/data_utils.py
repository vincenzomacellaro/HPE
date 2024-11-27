import os
import numpy as np
import json


def load_pkl_cartesian(choice):
    train_path = "../cartesian_data/train/train_data.json"
    val_path = "../cartesian_data/val/val_data.json"

    if choice == "train":
        with open(train_path, 'r') as f:
            data = json.load(f)

    elif choice == "val":
        with open(val_path, 'r') as f:
            data = json.load(f)

    return np.array(data)
