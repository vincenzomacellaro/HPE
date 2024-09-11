import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from script_angles.load_data_utils import load_data_for_train


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()

        # Encoder part
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[1], latent_dim)
        )

        # Latent space parameters
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

        # Decoder part
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Tanh()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return encoded, decoded, mu, log_var


# def loss_function(recon_x, x, mu, logvar):
#     # Compute the mean squared error loss between the reconstructed output and the input data
#     BCE = F.mse_loss(recon_x, x, reduction="sum")
#     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return BCE * KLD

def loss_function(recon_x, x, mu, logvar, gamma=0.0001):
    # Gamma parameter is used to perform the so-called KL-annealing, in which we weight BCE and KLD differently
    # 0.0001 works better than 0.001 which works better than 0.01...
    BCE = F.mse_loss(recon_x, x, reduction="sum")
    KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) * gamma
    total_loss = BCE + KLD
    return total_loss, BCE, KLD


def gen_model_name(hidden_dims, latent_dim):
    if isinstance(hidden_dims, int):
        hidden_dims = [hidden_dims]  # Convert to list if not already
    hidden_dims_str = 'x'.join(map(str, hidden_dims))  # Joins the list elements to form a string
    return f"vae_angle_hd{hidden_dims_str}_ld{latent_dim}.pth"


def train_model(vae, train_loader, val_loader, optimizer, num_epochs, patience=5):
    train_losses = []
    bce_losses = []
    kld_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0.0
        total_bce = 0.0
        total_kld = 0.0

        for batch in train_loader:
            inputs = batch.float()
            optimizer.zero_grad()
            x, recon_x, mu, log_var = vae(inputs)
            loss, bce, kld = loss_function(recon_x, inputs, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_bce = total_bce / len(train_loader.dataset)
        avg_kld = total_kld / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        bce_losses.append(avg_bce)
        kld_losses.append(avg_kld)
        train_losses.append(avg_train_loss)

        vae.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch.float()
                # decoded, mu, log_var, z
                x, recon_x, mu, log_var = vae(inputs)
                loss, bce, kld = loss_function(recon_x, inputs, mu, log_var)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(vae.state_dict(), model_path + gen_model_name(hidden_dims, latent_dim))
        else:
            counter += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], [BCE]: {avg_bce}, [KLD]: {avg_kld}")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} consecutive epochs.")
            break

    torch.save({
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dims,
        'state_dict': vae.state_dict()
    }, model_path + gen_model_name(hidden_dims, latent_dim))

    return train_losses, val_losses


if __name__ == '__main__':
    model_path = "../model/angles/"
    plot_path = "../plots/losses/"

    train_data = load_data_for_train('train', True)
    val_data = load_data_for_train('val', False)

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)

    print(f"[{len(train_dataset)}] Train Samples")
    print(f"[{len(val_dataset)}] Val Samples")

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    input_dim = 47
    hidden_dims = [128, 64]
    latent_dim = 32
    vae = VAE(input_dim, hidden_dims, latent_dim).float()

    num_epochs = 100
    learning_rate = 1e-3
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=1e-4)

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
