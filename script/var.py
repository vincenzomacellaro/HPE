import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
from load_dataset import load_data
from torch.utils.data import Dataset, DataLoader


class PoseDataset(Dataset):
    def __init__(self, poses):
        self.poses = torch.tensor(poses, dtype=torch.float32)  # Ensure data is converted to float32

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        return self.poses[idx]


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VAE(AutoEncoder):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__(input_dim, hidden_dim, latent_dim)
        # Add mu and log_var layers for reparameterization
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Ensure input is flattened
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return encoded, decoded.view(decoded.size(0), -1), mu, log_var  # Flatten decoded output

    def sample(self, num_samples):
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(num_samples, self.latent_dim)
            # Pass the noise through the decoder to generate samples
            samples = self.decoder(z)
        # Return the generated samples
        return samples.view(samples.size(0), -1, 3)


def loss_function(recon_x, x, mu, logvar):
    # Compute the binary cross-entropy loss between the reconstructed output and the input data
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 51), reduction="sum")
    # Compute the Kullback-Leibler divergence between the learned latent variable distribution and a standard Gaussian distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())*0.0001
    # Combine the two losses by adding them together and return the result
    return BCE + KLD


def gen_model_name(hdim, ldim):
    return "vae_hd" + str(hdim) + "_ld" + str(ldim) + ".pth"


def train_vae(input_dim, hidden_dim, latent_dim, train_loader, val_loader, learning_rate=1e-4, num_epochs=100, patience=5):
    model = VAE(input_dim, hidden_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction="sum")

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()  # Ensure the model is in training mode
        total_train_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            data = data.float()  # Ensure data is float32
            encoded, decoded, mu, log_var = model(data)
            # Compute the loss and perform backpropagation
            KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * 0.0001
            loss = criterion(decoded, data) + 3 * KLD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * data.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()  # Ensure the model is in evaluation mode
        total_val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.float()
                _, decoded, mu, log_var = model(data)
                KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * 0.0001
                val_loss = criterion(decoded, data) + 3 * KLD
                total_val_loss += val_loss.item() * data.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}: Train loss={avg_train_loss:.4f} Validation loss={avg_val_loss:.4f}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            # Save the best model
            torch.save({
                'input_dim': input_dim,
                'latent_dim': latent_dim,
                'hidden_dim': hidden_dim,
                'state_dict': model.state_dict()
            }, model_path + gen_model_name(256, 16))
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")

                torch.save({
                    'input_dim': input_dim,
                    'latent_dim': latent_dim,
                    'hidden_dim': hidden_dim,
                    'state_dict': model.state_dict()
                }, model_path + gen_model_name(256, 16))

                break

    return


if __name__ == '__main__':
    train_samples, train_ml_len, train_sl_len = load_data('train')
    val_samples, val_ml_len, val_sl_len = load_data('val')

    train_dataset = PoseDataset(np.asarray(train_samples))
    val_dataset = PoseDataset(np.asarray(val_samples))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    input_dim = 51
    hidden_dim = 256
    latent_dim = 16

    model_path = "../model/"
    train_vae(input_dim, hidden_dim, latent_dim, train_loader, val_loader)

    skeleton = [
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11),
        (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)
    ]
