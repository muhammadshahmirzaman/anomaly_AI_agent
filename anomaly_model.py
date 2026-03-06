"""
anomaly_model.py — Convolutional Autoencoder for self-supervised anomaly detection.

The autoencoder is trained on cropped person regions from NORMAL showroom videos.
During inference, the reconstruction error (MSE) between the input and the
reconstructed output serves as the anomaly score:
  - Low error → person appearance is similar to normal training data → normal
  - High error → person appearance differs from normal patterns → anomalous

Architecture:
  Encoder: Conv2d layers progressively downsample the input.
  Bottleneck: Compressed latent representation.
  Decoder: ConvTranspose2d layers reconstruct the original input.
"""

import torch
import torch.nn as nn
import config


class AnomalyAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for anomaly detection on person crops.

    Input shape:  (batch, 3, 128, 64)  — RGB person crops resized to 128x64
    Output shape: (batch, 3, 128, 64)  — Reconstructed person crops
    """

    def __init__(self):
        super(AnomalyAutoencoder, self).__init__()

        # =====================================================================
        # ENCODER — Compresses the input image into a latent representation
        # Each block: Conv2d → BatchNorm → LeakyReLU
        # Spatial dimensions are halved at each stage via stride=2
        # =====================================================================
        self.encoder = nn.Sequential(
            # Input: (3, 128, 64) → Output: (32, 64, 32)
            nn.Conv2d(config.INPUT_CHANNELS, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # (32, 64, 32) → (64, 32, 16)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # (64, 32, 16) → (128, 16, 8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # (128, 16, 8) → (256, 8, 4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # =====================================================================
        # DECODER — Reconstructs the image from the latent representation
        # Each block: ConvTranspose2d → BatchNorm → ReLU (final layer uses Sigmoid)
        # Spatial dimensions are doubled at each stage
        # =====================================================================
        self.decoder = nn.Sequential(
            # (256, 8, 4) → (128, 16, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (128, 16, 8) → (64, 32, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (64, 32, 16) → (32, 64, 32)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # (32, 64, 32) → (3, 128, 64)
            nn.ConvTranspose2d(32, config.INPUT_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, x):
        """
        Forward pass: encode then decode.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, 3, 128, 64).

        Returns:
            torch.Tensor: Reconstructed tensor of same shape as input.
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def compute_anomaly_score(model, crop_tensor, device=None):
    """
    Compute the anomaly score for a single cropped person region.

    The anomaly score is the Mean Squared Error (MSE) between the input
    and the autoencoder's reconstruction. Higher MSE = more anomalous.

    Args:
        model (AnomalyAutoencoder): Trained autoencoder model.
        crop_tensor (torch.Tensor): Preprocessed person crop tensor (C, H, W).
        device (torch.device): Device to run computation on. Defaults to config.DEVICE.

    Returns:
        float: Anomaly score (MSE reconstruction error).
    """
    if device is None:
        device = config.DEVICE

    model.eval()
    with torch.no_grad():
        # Add batch dimension: (C, H, W) → (1, C, H, W)
        input_tensor = crop_tensor.unsqueeze(0).to(device)

        # Forward pass through the autoencoder
        reconstructed = model(input_tensor)

        # Compute MSE between input and reconstruction
        mse = torch.mean((input_tensor - reconstructed) ** 2).item()

    return mse


def load_model(model_path=None):
    """
    Load a trained anomaly autoencoder model from disk.

    Args:
        model_path (str): Path to the saved model weights.
                          Defaults to config.MODEL_SAVE_PATH.

    Returns:
        AnomalyAutoencoder: The loaded model in eval mode on the configured device.
    """
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH

    model = AnomalyAutoencoder().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    print(f"[AnomalyModel] Loaded model from: {model_path}")
    return model
