import numpy as np
import json


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


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data