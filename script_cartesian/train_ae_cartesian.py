import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from data_utils import load_pkl_cartesian
from model import *


class SyntheticHypothesisDataset(Dataset):
    def __init__(self, ground_truth_data, num_hypotheses=20, noise_std_range=(0.1, 0.4), temperature=0.1):
        self.ground_truth_data = torch.tensor(np.array(ground_truth_data), dtype=torch.float32)
        self.num_hypotheses = num_hypotheses
        self.noise_std_range = noise_std_range
        self.temperature = temperature

    def __len__(self):
        return len(self.ground_truth_data)

    def __getitem__(self, idx):
        ground_truth = self.ground_truth_data[idx, :-16]

        # Generate synthetic hypotheses
        hypotheses = [
            ground_truth + torch.randn_like(ground_truth) * np.random.uniform(*self.noise_std_range) * self.temperature
            for _ in range(self.num_hypotheses)
        ]
        hypotheses = torch.stack(hypotheses)
        median_hypothesis = torch.median(hypotheses, dim=0).values

        return (hypotheses, median_hypothesis), ground_truth
        # return median_hypothesis, ground_truth


def gen_model_name(hidden_dims, latent_dim):
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]
    hidden_dims_str = 'x'.join(map(str, hidden_dims))
    return f"AE_cartesian_hd{hidden_dims_str}_ld{latent_dim}_test.pth"


def train_model(model, train_loader, val_loader, optimizer, num_epochs, patience):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for (hypotheses, aggr_hypothesis), ground_truth in train_loader:
            optimizer.zero_grad()

            _rec = model(aggr_hypothesis)
            _gt = ground_truth
            mean_loss = loss_function(_rec, _gt)

            # Combine mean loss and weighted individual loss
            total_loss = mean_loss
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() / hypotheses.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for (hypotheses, aggr_hypothesis), ground_truth in val_loader:
                _rec = model(aggr_hypothesis)
                _gt = ground_truth
                loss = loss_function(_rec, _gt)
                val_loss += loss.item() / hypotheses.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), model_path + gen_model_name(hidden_dim, latent_dim))
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    torch.save({
        'state_dict': model.state_dict(),
        'coord_dim': coord_dim,
        'angle_dim': angle_dim,
        'hidden_dim': hidden_dim,
        'latent_dim': latent_dim,
        'attn_layers': attn_layers
    }, model_path + gen_model_name(hidden_dim, latent_dim))

    return train_losses, val_losses


if __name__ == '__main__':
    model_path = "../model/cartesian/"
    plot_path = "../plots/losses/"

    angle_train = load_pkl_cartesian('train')
    angle_val = load_pkl_cartesian('val')

    tr_noise_range = (0.1, 0.3)
    val_noise_range = (0.05, 0.1)  # no noise for validation hypotheses
    temperature = 0.2735
    num_hypotheses = 20

    train_dataset = SyntheticHypothesisDataset(angle_train, num_hypotheses, tr_noise_range, temperature)
    val_dataset = SyntheticHypothesisDataset(angle_val, num_hypotheses, val_noise_range, temperature)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    coord_dim = 48
    angle_dim = 15
    hidden_dim = [128, 64]
    latent_dim = 32
    learning_rate = 1e-4
    attn_layers = 2

    # from model.py
    model = AE(coord_dim, angle_dim, hidden_dim, latent_dim, attn_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    num_epochs = 100
    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, num_epochs, 3)

    # Plotting the training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    out_pth = "../plots/cartesian_ae/"
    _model_name = gen_model_name(hidden_dim, latent_dim)
    out_file = _model_name + '_train_val.png'
    plt.savefig(out_pth + out_file, dpi=300, bbox_inches='tight')  # Save with high resolution
    print(f"Plot saved to {out_pth + out_file}")
    plt.show()
