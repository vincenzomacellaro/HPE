import os.path
import json
import numpy as np
import mat_utils as utils

from conversion_utils import convert_to_dictionary, add_hips_and_neck, get_bone_lengths

global_scale_factor_file = "../angles_json/global_scale_factor.json"


def get_hum_scale_factor():
    from load_data_utils import load_data
    data = load_data("train", False)
    # from the "Human36_to_angles" script;
    # param [scale=False] is needed for the "hum_scale_factor" calculation
    dicts = data["data"]

    hum_scale_factors = []

    for joints_array in dicts:
        frame_kpts = np.array(joints_array).reshape(1, 12, 3)

        # Convert joints_array to the format expected by the joint angles function
        kpts = convert_to_dictionary(frame_kpts)
        add_hips_and_neck(kpts)
        get_bone_lengths(kpts)

        # Calculate the average bone length for the current sample
        sample_lengths = sum(kpts['bone_lengths'].values())
        sample_count = len(kpts['bone_lengths'])

        if sample_count > 0:
            sample_scale_factor = sample_lengths / sample_count
            hum_scale_factors.append(sample_scale_factor)
        else:
            print(f"Warning: No bones found in sample, skipping...")

    hum_scale_factor = 0
    if hum_scale_factors:
        hum_scale_factor = sum(hum_scale_factors) / len(hum_scale_factors)
        print(f"[Human3.6M] scale factor: {hum_scale_factor}")
    else:
        print("No valid scale factors calculated; check data inputs.")

    return hum_scale_factor


def get_jac_scale_factor():
    from load_data_utils import load_ref_kpts
    # jac is short for [joint_angles_calculate]
    kpts_file = "../ref_data/ref_kpts.dat"
    kpts = load_ref_kpts(kpts_file)

    # rotate to orient the pose properly
    R = utils.get_R_z(np.pi / 2)

    total_frames = []

    for framenum in range(kpts.shape[0]):
        for kpt_num in range(kpts.shape[1]):
            kpts[framenum, kpt_num] = R @ kpts[framenum, kpt_num]

        sample_kpts = convert_to_dictionary(kpts)
        add_hips_and_neck(sample_kpts)
        get_bone_lengths(sample_kpts)
        get_bone_lengths(sample_kpts)

        total_frames.append(sample_kpts)

    org_scale_factors = []

    for frame in total_frames:

        sample_lengths = sum(frame['bone_lengths'].values())
        sample_count = len(frame['bone_lengths'])

        if sample_count > 0:
            sample_scale_factor = sample_lengths / sample_count
            org_scale_factors.append(sample_scale_factor)
        else:
            print(f"Warning: No bones found in sample, skipping...")

    org_scale_factor = 0
    if org_scale_factors:
        org_scale_factor = sum(org_scale_factors) / len(org_scale_factors)
        print(f"[original dataset] scale factor: {org_scale_factor}")
    else:
        print("No valid scale factors calculated; check data inputs.")

    return org_scale_factor


def load_global_scale_factor():
    if not os.path.exists(global_scale_factor_file):
        org_scale_factor = get_jac_scale_factor()
        hum_scale_factor = get_hum_scale_factor()

        if org_scale_factor != 0 and hum_scale_factor != 0:
            final_scale_factor = org_scale_factor / hum_scale_factor
            with open(global_scale_factor_file, "w") as f:
                json.dump(final_scale_factor, f)
                print(f"Global scale factor saved to {global_scale_factor_file}")
            return final_scale_factor
        else:
            print("No valid scale factors calculated; check data inputs.")

    else:
        with open(global_scale_factor_file, "r") as f:
            global_scale_factor = float(json.load(f))

        return global_scale_factor


if __name__ == "__main__":
    load_global_scale_factor()
