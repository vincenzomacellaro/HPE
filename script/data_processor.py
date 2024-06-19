import torch
import torch.nn.functional as F

skeleton = [
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11),
    (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)
]


def limb_length_penalty(decoded_coords, expected_lengths):
    penalties = []
    num_coords_per_joint = 3  # Each joint has 3 coordinates (x, y, z)

    for (joint1, joint2), expected_length in expected_lengths.items():
        # Compute start index for each joint's coordinates
        start_idx1 = joint1 * num_coords_per_joint
        start_idx2 = joint2 * num_coords_per_joint

        # Extract the coordinates for both joints
        joint1_coords = decoded_coords[:, start_idx1:start_idx1 + num_coords_per_joint]
        joint2_coords = decoded_coords[:, start_idx2:start_idx2 + num_coords_per_joint]

        # Calculate the Euclidean distance between the two joints
        limb_lengths = torch.norm(joint1_coords - joint2_coords, dim=1)

        # Calculate penalty for length deviation beyond the tolerance
        penalty = torch.abs(limb_lengths - expected_length) - (0.1 * expected_length)
        penalty = torch.clamp(penalty, min=0)  # Only penalize if outside tolerance
        penalties.append(penalty)

    # Sum up penalties across all limbs
    total_penalty = torch.sum(torch.stack(penalties), dim=0)
    return total_penalty.mean()  # Average over the batch


def custom_loss_function(recon_x, x, mu, logvar, expected_lengths, beta=0.1):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    limb_penalty = (limb_length_penalty(recon_x, expected_lengths)) * 10
    total_loss = recon_loss + beta * kl_loss + limb_penalty
    return total_loss, recon_loss, kl_loss, limb_penalty

