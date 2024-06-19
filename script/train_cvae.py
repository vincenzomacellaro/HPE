import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from load_dataset import load_data

input_length = 67

# Define a dummy dataset for illustration
class PoseDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = np.expand_dims(sample, axis=0)  # Add a channel dimension
        return torch.tensor(sample, dtype=torch.float32)


# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(32, 64)
        self.res2 = ResidualBlock(64, 128)
        self.fc_input_size = 128 * input_length

        self.fc_mu = nn.Linear(self.fc_input_size, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_input_size, latent_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * input_length)
        self.res1 = ResidualBlock(128, 64)
        self.res2 = ResidualBlock(64, 32)
        self.deconv3 = nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, input_length)  # Reshape to match deconvolutional input shape
        x = self.res1(x)
        x = self.res2(x)
        x = torch.sigmoid(self.deconv3(x))  # Use sigmoid activation for output
        return x


# Define the VAE class combining Encoder and Decoder
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total VAE loss
    total_loss = recon_loss + kl_loss

    return total_loss

def gen_model_name(ldim):
    return "cvae_ld" + str(ldim) + "_max.pth"

def train_model(vae, train_loader, val_loader, optimizer, num_epochs, patience=5):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0.0

        for batch in train_loader:
            inputs = batch
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(inputs)
            loss = loss_function(recon_x, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        vae.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch
                recon_x, mu, logvar = vae(inputs)
                loss = loss_function(recon_x, inputs, mu, logvar)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(vae.state_dict(), model_path + gen_model_name(latent_dim))
        else:
            counter += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} consecutive epochs.")
            break

    torch.save({
        'input_channels': input_channels,
        'latent_dim': latent_dim,
        'state_dict': model.state_dict()
    }, model_path + gen_model_name(latent_dim))

    return train_losses, val_losses

if __name__ == '__main__':
    # Set paths and other parameters
    model_path = "../model/"
    plot_path = "../plots/losses/"

    # Define hyperparameters
    input_channels = 1  # Since we're treating the input as (1, 67)
    latent_dim = 32

    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001

    # Load data
    train_samples = load_data('train')
    val_samples = load_data('val')

    train_dataset = PoseDataset(train_samples)  # (1, 67)
    val_dataset = PoseDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = VAE(latent_dim)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, num_epochs)

    # Plotting the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(plot_path + gen_model_name(latent_dim)[:-4] + '_loss.png')
    plt.show()
