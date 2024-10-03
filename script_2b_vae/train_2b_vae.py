import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from script_angles.load_data_utils import load_data_for_train


class CustomDataset(Dataset):
    def __init__(self, pose_data, physique_data):
        self.pose_data = pose_data
        self.physique_data = physique_data

    def __len__(self):
        return len(self.pose_data)

    def __getitem__(self, idx):
        pose_sample = torch.tensor(self.pose_data[idx], dtype=torch.float32)
        physique_sample = torch.tensor(self.physique_data[idx], dtype=torch.float32)
        return pose_sample, physique_sample


class VAE(nn.Module):
    def __init__(self, pose_input_dim, physique_input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()

        # Pose Encoder
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[1], latent_dim)
        )

        # Physique Encoder
        self.physique_encoder = nn.Sequential(
            nn.Linear(physique_input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[1], latent_dim)
        )

        # Latent space parameters for both
        self.pose_mu = nn.Linear(latent_dim, latent_dim)
        self.pose_log_var = nn.Linear(latent_dim, latent_dim)

        self.physique_mu = nn.Linear(latent_dim, latent_dim)
        self.physique_log_var = nn.Linear(latent_dim, latent_dim)

        # Pose Decoder
        self.pose_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[0], pose_input_dim),
            nn.Tanh()
        )

        # Physique Decoder
        self.physique_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[0], physique_input_dim),
            nn.Tanh()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, pose_input, physique_input):
        # Pose latent space
        pose_encoded = self.pose_encoder(pose_input)
        pose_mu = self.pose_mu(pose_encoded)
        pose_log_var = self.pose_log_var(pose_encoded)
        pose_z = self.reparameterize(pose_mu, pose_log_var)
        pose_reconstructed = self.pose_decoder(pose_z)

        # Physique latent space
        physique_encoded = self.physique_encoder(physique_input)
        physique_mu = self.physique_mu(physique_encoded)
        physique_log_var = self.physique_log_var(physique_encoded)
        physique_z = self.reparameterize(physique_mu, physique_log_var)
        physique_reconstructed = self.physique_decoder(physique_z)

        return pose_reconstructed, physique_reconstructed, pose_mu, pose_log_var, physique_mu, physique_log_var


def loss_function(pose_recon, pose_input, physique_recon, physique_input,
                  pose_mu, pose_log_var, physique_mu, physique_log_var, gamma=0.0001):

    pose_recon_loss = F.mse_loss(pose_recon, pose_input, reduction="sum")
    physique_recon_loss = F.mse_loss(physique_recon, physique_input, reduction="sum")

    pose_KLD = -0.5 * torch.sum(1 + pose_log_var - pose_mu.pow(2) - pose_log_var.exp())
    physique_KLD = -0.5 * torch.sum(1 + physique_log_var - physique_mu.pow(2) - physique_log_var.exp())

    KLD_total = (pose_KLD + physique_KLD) * gamma
    recon_loss = pose_recon_loss + physique_recon_loss
    total_loss = recon_loss + KLD_total

    return total_loss, recon_loss, KLD_total


def gen_model_name(hidden_dims, latent_dim):
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]  # Convert to list if not already
    hidden_dims_str = 'x'.join(map(str, hidden_dims))  # Joins the list elements to form a string
    return f"2b_vae_hd{hidden_dims_str}_ld{latent_dim}.pth"


def train_model(vae, train_loader, val_loader, optimizer, num_epochs, patience=5):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0.0

        for pose_batch, physique_batch in train_loader:
            optimizer.zero_grad()
            pose_batch = pose_batch.float()
            physique_batch = physique_batch.float()
            pose_recon, physique_recon, pose_mu, pose_log_var, physique_mu, physique_log_var = vae(pose_batch,
                                                                                                   physique_batch)
            loss, recon_loss, KLD_total = loss_function(pose_recon, pose_batch, physique_recon, physique_batch,
                                                        pose_mu, pose_log_var, physique_mu, physique_log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        vae.eval()
        val_loss = 0.0

        with torch.no_grad():
            for pose_batch, physique_batch in val_loader:
                pose_batch = pose_batch.float()
                physique_batch = physique_batch.float()
                pose_recon, physique_recon, pose_mu, pose_log_var, physique_mu, physique_log_var = vae(pose_batch, physique_batch)
                loss, recon_loss, KLD_total = loss_function(pose_recon, pose_batch, physique_recon, physique_batch,
                                                            pose_mu, pose_log_var, physique_mu, physique_log_var)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(vae.state_dict(), model_path + gen_model_name(hidden_dims, latent_dim))
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    torch.save({
        'pose_input_dim': pose_input_dim,
        'physique_input_dim': physique_input_dim,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dims,
        'state_dict': vae.state_dict()
    }, model_path + gen_model_name(hidden_dims, latent_dim))

    return train_losses, val_losses


if __name__ == '__main__':
    model_path = "../model/angles/"
    plot_path = "../plots/losses/"

    pose_train_data, physique_train_data = load_data_for_train('train', True)
    pose_val_data, physique_val_data = load_data_for_train('val', False)

    print(f"Pose train data[0]: {len(pose_train_data[0])}")
    print(f"Physique train data[0]: {len(physique_train_data[0])}")

    # Create dataset with both pose and physique data
    train_dataset = CustomDataset(pose_train_data, physique_train_data)
    val_dataset = CustomDataset(pose_val_data, physique_val_data)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Set input dimensions for both pose and physique data
    pose_input_dim = len(pose_train_data[0])  # Input dimension of the pose data
    physique_input_dim = len(physique_train_data[0])  # Input dimension of the physique data
    hidden_dims = [256, 128]
    latent_dim = 16

    # Initialize the modified VAE
    vae = VAE(pose_input_dim, physique_input_dim, hidden_dims, latent_dim).float()

    num_epochs = 100
    learning_rate = 1e-3
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Train the model
    train_losses, val_losses = train_model(vae, train_loader, val_loader, optimizer, num_epochs)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(plot_path + gen_model_name(hidden_dims, latent_dim)[:-4] + '_loss.png')
    plt.show()
