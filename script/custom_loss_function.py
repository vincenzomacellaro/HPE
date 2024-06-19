import torch
import torch.nn.functional as F

skeleton = [
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11),
    (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)
]


def limb_length_penalty(decoded_coords, mean_limb_length, std_limb_length):
    global skeleton

    # Ensure the correct reshaping
    if decoded_coords.dim() == 2 and decoded_coords.size(1) == 17 * 3:
        decoded_coords = decoded_coords.view(-1, 17, 3)  # Reshape

    penalty = 0
    count = 0
    for joint1, joint2 in skeleton:
        # Use dim=1 to compute norms across the second dimension, which now corresponds to the 3 coordinates
        pred_length = torch.norm(decoded_coords[:, joint1] - decoded_coords[:, joint2], dim=1)
        expected_length = torch.full_like(pred_length, mean_limb_length)
        penalty += torch.mean((pred_length - expected_length) ** 2) / (std_limb_length ** 2)
        count += 1
    return penalty / count


def custom_loss_function(recon_x, x, mu, logvar, mean_limb_length, std_limb_length, beta=0.1, gamma=0.1):
    # Check if reshape is needed:
    if recon_x.size(1) == 17 * 3:
        recon_x = recon_x.view(-1, 17, 3)
    recon_loss = F.mse_loss(recon_x.view(-1), x.view(-1), reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    limb_penalty = limb_length_penalty(recon_x, mean_limb_length, std_limb_length) * gamma
    total_loss = recon_loss + beta * kl_loss + limb_penalty
    return total_loss, recon_loss, kl_loss, limb_penalty



