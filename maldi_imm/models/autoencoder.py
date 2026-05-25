import torch
import torch.nn as nn
import torch.nn.functional as F


class MaldiAutoencoder(nn.Module):
    """
    Lightweight convolutional autoencoder that maps MALDI-TOF spectra to a
    784-d latent space and reconstructs the input.

    Input tensors must be shaped as (B, 1, L) where L is typically 6000 bins.
    Downsampling is done with 4 stride-2 convolutions, so L must be divisible
    by 16 (e.g. 6000). The decoder mirrors the encoder with transposed
    convolutions so the reconstruction has the same length as the input.
    """

    def __init__(
        self,
        input_length: int = 6000,
        latent_dim: int = 784,
        in_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if input_length % 16 != 0:
            raise ValueError("input_length must be divisible by 16 to match the 4 stride-2 blocks.")

        self.input_length = int(input_length)
        self.latent_dim = int(latent_dim)
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)

        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(self.in_channels, base_channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, self.input_length)
            enc_feat = self.encoder_cnn(dummy)
        self._enc_channels = enc_feat.shape[1]
        self._enc_length = enc_feat.shape[2]
        self._enc_flat_dim = self._enc_channels * self._enc_length

        self.encoder_fc = nn.Sequential(
            nn.Linear(self._enc_flat_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, self.latent_dim),
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(self.latent_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, self._enc_flat_dim),
            nn.GELU(),
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(self._enc_channels, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(base_channels, self.in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_cnn(x)
        h = h.view(x.size(0), -1)
        z = self.encoder_fc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(z.size(0), self._enc_channels, self._enc_length)
        recon = self.decoder_cnn(h)
        return recon

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        recon = self.decode(z)
        if return_latent:
            return recon, z
        return recon

    @staticmethod
    def reconstruction_losses(
        recon: torch.Tensor,
        target: torch.Tensor,
        *,
        bce_eps: float = 1e-6,
    ):
        mse = F.mse_loss(recon, target)
        signal_mask = (target > 0).float()
        recon_prob = torch.clamp(recon, min=bce_eps, max=1.0 - bce_eps)
        bce = F.binary_cross_entropy(recon_prob, signal_mask)
        return mse, bce
