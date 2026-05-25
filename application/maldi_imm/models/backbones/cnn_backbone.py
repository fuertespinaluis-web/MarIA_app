import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """
    1D CNN encoder baseline copied from the autoencoder encoder.

    Input:  (B, C, L) where C=1 by default.
    Output: (B, 768) embedding.
    """

    def __init__(
        self,
        input_length: int = 6000,
        embed_dim: int = 768,
        in_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if input_length % 16 != 0:
            raise ValueError("input_length must be divisible by 16 to match the 4 stride-2 blocks.")

        self.input_length = int(input_length)
        self.embed_dim = int(embed_dim)
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
            nn.Linear(2048, self.embed_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_cnn(x)
        h = h.view(x.size(0), -1)
        z = self.encoder_fc(h)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


class CNNDecoder(nn.Module):
    """
    1D CNN decoder that mirrors CNNBackbone in reverse.

    Input:  (B, 768) embedding.
    Output: (B, C, L) reconstruction.
    """

    def __init__(
        self,
        input_length: int = 6000,
        embed_dim: int = 768,
        in_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if input_length % 16 != 0:
            raise ValueError("input_length must be divisible by 16 to match the 4 stride-2 blocks.")

        self.input_length = int(input_length)
        self.embed_dim = int(embed_dim)
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)

        shape_cnn = nn.Sequential(
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
            enc_feat = shape_cnn(dummy)
        self._enc_channels = enc_feat.shape[1]
        self._enc_length = enc_feat.shape[2]
        self._enc_flat_dim = self._enc_channels * self._enc_length

        self.decoder_fc = nn.Sequential(
            nn.Linear(self.embed_dim, 2048),
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
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(z.size(0), self._enc_channels, self._enc_length)
        recon = self.decoder_cnn(h)
        return recon

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)
