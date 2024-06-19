import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from load_dataset import load_data
from torch.utils.data import Dataset, DataLoader

# save_path
model_path = "../model/"
plot_path = "../plots/losses/"


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Extract the values (coordinates) from the dictionary
        values = torch.tensor([v for v in sample.values()], dtype=torch.float32)
        return values


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Tanh()  # Assuming input poses are normalized between [-1, 1]
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(decoded.size(0), -1, 3)


input_dim = 3 * 17
latent_dim = 16
autoencoder = Autoencoder(input_dim, latent_dim)

# Define the reconstruction loss criterion (MSE)
criterion = nn.MSELoss()

# Define the optimizer (Adam)
learning_rate = 0.001  # Adjust as needed
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# choices for the "load_data" function: 'train', 'val', 'test'
train_samples = load_data('train')
val_samples = load_data('val')

print(" Train samples: " + str(len(train_samples)))
print(" Val samples: " + str(len(val_samples)))

# Assuming you have loaded your data into a variable called 'data'
train_dataset = CustomDataset(train_samples)
val_dataset = CustomDataset(val_samples)

batch_size = 32

# Define dataloaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation data

# List to store the loss values
losses = []
num_epochs = 100


def gen_model_name(ldim):
    return "model_ld" + str(ldim) + "_v3.pth"


def train_model(autoencoder, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(num_epochs):

        autoencoder.train()
        train_loss = 0.0

        for batch in train_loader:
            inputs = batch
            reconstructed = autoencoder(inputs)
            loss = criterion(reconstructed, inputs.view(inputs.size(0), -1, 3))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        autoencoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch
                reconstructed = autoencoder(inputs)
                loss = criterion(reconstructed, inputs.view(inputs.size(0), -1, 3))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(autoencoder.state_dict(), model_path + gen_model_name(latent_dim))
        else:
            counter += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        if counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1} as validation loss did not improve for {patience} consecutive epochs.")
            break

    torch.save({
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'state_dict': autoencoder.state_dict()
    }, model_path + gen_model_name(latent_dim))

    return train_losses, val_losses


# ...
# Once trained, you can sample from the latent space to generate new poses
# latent_samples = torch.randn(num_samples, latent_dim)
# generated_poses = autoencoder.decoder(latent_samples)

if __name__ == '__main__':
    train_losses, val_losses = train_model(autoencoder, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Plotting the training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(plot_path + gen_model_name(latent_dim)[:-4] + '_mse_loss.png')
    plt.show()
