import random
import json

from script_angles.load_data_utils import load_data_for_train, load_json
from script_angles.general_utils import to_numpy
from script_angles.human36_to_angles import flatten_numeric_values, filter_samples, reconstruct_from_array
from script_angles.plot_utils import plot_pose_from_joint_angles
from test_utils import reconstruct_sample


avg_pose = None


def load_unprocessed_sample(samples, idx):
    # doesn't perform any normalization of the data
    raw_sample = samples[idx]
    raw_sample_np = to_numpy(raw_sample)  # converts to numpy values
    filtered_sample = filter_samples(raw_sample_np)  # filters the complete sample to get the [needed] data only
    num_sample = flatten_numeric_values(filtered_sample)
    rec_sample = reconstruct_from_array(num_sample, sample_keys_file)
    return rec_sample


def load_processed_sample(pose_samples, physique_samples, idx):
    # performs the same processing as the VAE input data
    pose_sample = pose_samples[idx]
    physique_sample = physique_samples[idx]
    prc_sample = reconstruct_sample(pose_sample, physique_sample, scale_factors)
    return prc_sample


def test_rec():
    # LOAD ORIGINAL SAMPLE
    sample_path = "../angles_data/test/test_data.json"
    joints_data = load_json(sample_path)  # 79345

    # LOAD PROCESSED SAMPLE
    pose_test_data, physique_test_data = load_data_for_train("test")  # 79345

    samples_num = len(joints_data)
    rand_idx = random.randint(0, samples_num - 1)

    unprocessed_sample = load_unprocessed_sample(joints_data, rand_idx)
    processed_sample = load_processed_sample(pose_test_data, physique_test_data, rand_idx)

    plot_pose_from_joint_angles(unprocessed_sample, "[UNPROCESSED] sample")
    plot_pose_from_joint_angles(processed_sample, "[PROCESSED] sample")

    return


if __name__ == '__main__':
    # script for debugging purposes only;
    # it checks if norm. and inverse-norm operations work as expected

    sample_keys_file = "../angles_json/sample_keys.json"

    param_file = "../angles_json/scale_factors.json"
    scale_factors = json.load(open(param_file))

    test_rec()

    # this script loads the [../angles_data/test/test_data.json] dataset, then a random sample from it, and
    # checks if the reconstruction process works as expected

