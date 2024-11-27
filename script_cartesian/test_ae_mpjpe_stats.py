from test_utils import *
from tqdm import tqdm
from model import AE
from mean_mpjpe import mpjpe_diffusion_all_min

human36_joints = {
    'pelvis': 0,
    'righthip': 1,
    'rightknee': 2,
    'rightfoot': 3,
    'lefthip': 4,
    'leftknee': 5,
    'leftfoot': 6,
    'spine1': 7,
    'neck': 8,
    'head': 9,
    'site': 10,
    'leftshoulder': 11,
    'leftelbow': 12,
    'leftwrist': 13,
    'rightshoulder': 14,
    'rightelbow': 15,
    'rightwrist': 16
}

human36_connections = {
    'pelvis': ['righthip', 'lefthip', 'spine1'],
    'righthip': ['rightknee'],
    'rightknee': ['rightfoot'],
    'lefthip': ['leftknee'],
    'leftknee': ['leftfoot'],
    'spine1': ['neck', 'leftshoulder', 'rightshoulder'],
    'neck': ['head', 'site'],
    'leftshoulder': ['leftelbow'],
    'leftelbow': ['leftwrist'],
    'rightshoulder': ['rightelbow'],
    'rightelbow': ['rightwrist']
}

human36_ord_list = [
    'righthip', 'lefthip', 'spine1', 'rightknee', 'rightfoot', 'leftknee', 'leftfoot',
    'neck', 'leftshoulder', 'rightshoulder', 'head', 'site', 'leftelbow', 'leftwrist',
    'rightelbow', 'rightwrist'
]

human36_bones = [
    ('pelvis', 'spine1'),
    ('spine1', 'neck'),
    ('neck', 'head'),
    ('neck', 'leftshoulder'),
    ('neck', 'rightshoulder'),
    ('leftshoulder', 'leftelbow'),
    ('leftelbow', 'leftwrist'),
    ('rightshoulder', 'rightelbow'),
    ('rightelbow', 'rightwrist'),
    ('pelvis', 'lefthip'),
    ('lefthip', 'leftknee'),
    ('leftknee', 'leftfoot'),
    ('pelvis', 'righthip'),
    ('righthip', 'rightknee'),
    ('rightknee', 'rightfoot')
]


def compute_distance(hypothesis, ground_truth):
    """Computes the Euclidean distance between a hypothesis and ground truth."""
    return np.linalg.norm(hypothesis - ground_truth, axis=-1)  # axis=-1 for 3D points


def compute_noise_level(ground_truth, hypotheses):
    """
    Computes the mean, median, and std error between each hypothesis and the ground truth for each sample.
    """
    distances = np.array(
        [compute_distance(hypothesis, ground_truth) for hypothesis in hypotheses])  # (20, 17, 3) -> (20, 17)

    mean_error = np.mean(distances, axis=0)  # Mean error across hypotheses
    median_error = np.median(distances, axis=0)  # Median error across hypotheses
    std_error = np.std(distances, axis=0)  # Std deviation across hypotheses

    # Optionally return mean, median, and standard deviation
    return mean_error, median_error, std_error


if __name__ == '__main__':

    model_path = "../model/cartesian/"
    model_name = "AE_cartesian_hd128x64_ld32_att_med_ref.pth"

    ground_truth, predictions = load_diffusion_data()

    mean_errors, median_errors, std_errors = [], [], []

    # Compute noise level metrics for each test sample
    for gt_sample, hyp_sample in zip(ground_truth, predictions):
        # (1, 17, 3)(20, 17, 3)
        gt_tensor = torch.tensor(gt_sample)
        pred_tensor = torch.tensor(hyp_sample)

        gt_rel = preprocess_batch(gt_tensor)
        pred_rel = preprocess_batch(pred_tensor)

        mean_error, median_error, std_error = compute_noise_level(gt_rel, pred_rel)

        mean_errors.append(mean_error)
        median_errors.append(median_error)
        std_errors.append(std_error)

        # Compute global metrics
    global_mean = np.mean(mean_errors)
    global_median = np.median(median_errors)
    global_std = np.mean(std_errors)

    print(f"Global Mean Error: {global_mean:.4f}")
    print(f"Global Median Error: {global_median:.4f}")
    print(f"Global Std Deviation Error: {global_std:.4f}")

    print(" *** " + 3)

    batch_size = 128
    total_batches = len(ground_truth) // batch_size + (1 if len(ground_truth) % batch_size != 0 else 0)

    mpjpe = []

    # Iterate over the array in chunks of [batch_size] elements
    for i in tqdm(range(0, len(ground_truth), batch_size), total=total_batches, desc="Processing", unit="batch"):

        # Select a batch of [batch_size] elements
        gt_batch = ground_truth[i:i + batch_size]
        pred_batch = predictions[i:i + batch_size]

        for batch_idx in range(len(gt_batch)):
            gt_sample = gt_batch[batch_idx]  # (1, 17, 3)
            hyp_for_curr_sample = pred_batch[batch_idx]  # (20, 17, 3)

            # Flatten hypotheses and ground truth for distance computation
            gt_sample_flat = gt_sample.flatten()  # (51,) assuming 17 joints, 3 coordinates
            hyp_for_curr_sample_flat = hyp_for_curr_sample.reshape(hyp_for_curr_sample.shape[0], -1)
            hyp_tensor = torch.from_numpy(hyp_for_curr_sample_flat).float()

            # Compute distances to the median hypothesis
            median_hypothesis = torch.median(hyp_tensor, dim=0).values  # (51,)
            distances = torch.linalg.norm(hyp_tensor - median_hypothesis.unsqueeze(0), dim=-1)  # (20,)

            # Store statistics
            distance_stats["mean"].append(distances.mean().item())
            distance_stats["std"].append(distances.std().item())
            distance_stats["median"].append(distances.median().item())

    # Summarize statistics over all samples
    overall_mean = torch.tensor(distance_stats["mean"]).mean().item()
    overall_std = torch.tensor(distance_stats["std"]).mean().item()
    overall_median = torch.tensor(distance_stats["median"]).mean().item()

    print(f"Overall Mean Distance: {overall_mean:.4f}")
    print(f"Overall Std Distance: {overall_std:.4f}")
    print(f"Overall Median Distance: {overall_median:.4f}")
