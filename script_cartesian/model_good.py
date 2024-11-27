import torch.nn as nn
import torch.nn.functional as F


def loss_function(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum') / x.size(0)


class StackedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2):
        super(StackedSelfAttention, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True),
                nn.LayerNorm(embed_dim)
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            attn, _ = layer[0](x, x, x)  # Multihead attention
            x = layer[1](attn + x)  # Add & Normalize (residual connection)
        return x


class AE(nn.Module):
    def __init__(self, relative_input_dim, angle_input_dim, hidden_dims, latent_dim, num_attn_layers=2):
        super(AE, self).__init__()
        pose_input_dim = relative_input_dim + angle_input_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(pose_input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dims[1], latent_dim)
        )

        # Adding self-attention after encoding
        self.attention = StackedSelfAttention(embed_dim=latent_dim, num_heads=4, num_layers=num_attn_layers)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dims[0], pose_input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)

        # Reshape latent space to [batch_size, 1, latent_dim] for attention
        encoded = encoded.unsqueeze(1)  # Adding sequence dimension
        attended = self.attention(encoded)  # Apply attention to the encoded representation
        attended = attended.squeeze(1)  # Remove sequence dimension

        decoded = self.decoder(attended)
        return decoded + x