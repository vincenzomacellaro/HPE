import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from load_dataset import load_data
from custom_loss_function import custom_loss_function


# Custom Dataset Definition
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
         sample = self.data[idx]
         return torch.tensor(sample, dtype=torch.float32)


# Encoder Definition
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x), self.fc3(x)  # Returns mu and log_var


# Decoder Definition
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(z))  # Using sigmoid to ensure the output is between 0 and 1


# VAE definition
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# not currently in use, refer to the "custom_loss_function.py" for the updated one
def loss_function(recon_x, x, mu, logvar, beta=0.1):  # Adjust beta as needed
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


def gen_model_name(hdim, ldim):
    return "vae_hd" + str(hdim) + "_ld" + str(ldim) + ".pth"


def train_model(vae, train_loader, val_loader, optimizer, num_epochs,
                train_ml_len, train_sl_len, val_ml_len, val_sl_len, patience=5):

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    counter = 0

    beta = 0.1
    gamma = 100

    for epoch in range(num_epochs):
        vae.train()
        train_loss = 0.0

        for batch in train_loader:
            inputs = batch
            optimizer.zero_grad()
            recon_x, mu, logvar = vae(inputs)

            # beta parameter constrols the influence of the KL-divergence on the total loss
            total_loss, recon_loss, kl_loss, limb_penalty = custom_loss_function(
                recon_x, inputs, mu, logvar, train_ml_len, train_sl_len, beta=beta, gamma=gamma
            )

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        vae.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch
                recon_x, mu, logvar = vae(inputs)
                total_loss, recon_loss, kl_loss, limb_penalty = custom_loss_function(
                    recon_x, inputs, mu, logvar, val_ml_len, val_sl_len, beta=beta, gamma=gamma
                )
                val_loss += total_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(vae.state_dict(), model_path + gen_model_name(hidden_dim, latent_dim))
        else:
            counter += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")
        print(f"\t [{epoch + 1}/{num_epochs}], Total Loss: {total_loss} "
              f"Reconstruction Loss: {recon_loss} "
              f"KL Loss: {kl_loss} "
              f"Limb Penalty: {limb_penalty}")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} consecutive epochs.")
            break

    torch.save({
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'state_dict': vae.state_dict()
    }, model_path + gen_model_name(hidden_dim, latent_dim))

    return train_losses, val_losses


if __name__ == '__main__':

    model_path = "../model/"
    plot_path = "../plots/losses/"

    # [...]_ml_len = mean_limb_length
    # [...]_sl_len = std_limb_length
    train_samples, train_ml_len, train_sl_len = load_data('train')
    val_samples, val_ml_len, val_sl_len = load_data('val')

    train_dataset = CustomDataset(np.asarray(train_samples))
    val_dataset = CustomDataset(np.asarray(val_samples))

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # input_dim = 67
    input_dim = 51
    hidden_dim = 256
    latent_dim = 16
    vae = VAE(input_dim, hidden_dim, latent_dim)

    num_epochs = 100
    learning_rate = 0.001
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_losses, val_losses = train_model(
        vae, train_loader, val_loader, optimizer, num_epochs,
        train_ml_len, train_sl_len, val_ml_len, val_sl_len
    )

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(plot_path + gen_model_name(hidden_dim, latent_dim)[:-4] + '_loss.png')
    plt.show()
