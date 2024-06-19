import numpy as np


class Normalizer:
    def __init__(self, root_joint=0):
        self.root_joint = root_joint
        self.min_coords = None
        self.max_coords = None

    def root_centric_normalize(self, joints_data):
        if isinstance(joints_data, dict):
            joints_data = [joints_data]

        normalized_joints_list = []

        for joints in joints_data:
            root_coords = np.array(joints[str(self.root_joint)])
            normalized_joints = {}
            for joint_id, coords in joints.items():
                normalized_coords = np.array(coords) - root_coords
                normalized_joints[joint_id] = normalized_coords.tolist()
            normalized_joints_list.append(normalized_joints)

        return normalized_joints_list

    def fit(self, normalized_joints_list):
        all_coords = np.concatenate([np.array(list(frame.values())) for frame in normalized_joints_list])
        self.min_coords = np.min(all_coords, axis=0)
        self.max_coords = np.max(all_coords, axis=0)

    # def min_max_normalize(self, normalized_joints_list):
    #     if self.min_coords is None or self.max_coords is None:
    #         raise ValueError("Min-Max normalization parameters not set. Call 'fit' with data first.")
    #
    #     scaled_joints_list = []
    #     for joints in normalized_joints_list:
    #         scaled_joints = {}
    #         for joint_id, coords in joints.items():
    #             coords_array = np.array(coords)
    #             scaled_coords = (coords_array - self.min_coords) / (self.max_coords - self.min_coords) * 2 - 1
    #             scaled_joints[joint_id] = scaled_coords.tolist()
    #         scaled_joints_list.append(scaled_joints)
    #
    #     return np.array(scaled_joints_list)

    def min_max_scaling(self, data):
        # Extract all coordinates from the raw data
        all_coordinates = [joint_coord for pose in data for joint_coord in pose.values()]

        # Convert to numpy array for easier manipulation
        all_coordinates = np.array(all_coordinates)

        # Compute the minimum and maximum values for each dimension (x, y, z)
        min_values = np.min(all_coordinates, axis=0)
        max_values = np.max(all_coordinates, axis=0)

        # Apply Min-Max Scaling
        scaled_data = []
        for pose in data:
            scaled_pose = {}
            for joint_id, joint_coord in pose.items():
                # Scale each coordinate individually using the min-max formula
                scaled_coord = [(coord - min_val) / (max_val - min_val) for coord, min_val, max_val in
                                zip(joint_coord, min_values, max_values)]
                scaled_pose[joint_id] = scaled_coord
            scaled_data.append(scaled_pose)

        return scaled_data