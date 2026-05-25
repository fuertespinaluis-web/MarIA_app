# src/models/simple_diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Sinusoidal time embedding (from lucidrains)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

# Simple Residual Block (1D)
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Time embedding add
        t_emb = self.time_emb_proj(t)
        h = h + t_emb[..., None]
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        
        return h + self.residual_conv(x)

# 1D U-Net for diffusion (lucidrains style)
class UNet1D(nn.Module):
    def __init__(
        self, 
        in_channels=1, 
        n_features=64, 
        time_emb_dim=128
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Initial conv
        self.init_conv = nn.Conv1d(in_channels, n_features, 7, padding=3)

        # Down path
        self.resblock1 = ResidualBlock1D(n_features, n_features, time_emb_dim)
        self.resblock2 = ResidualBlock1D(n_features, n_features, time_emb_dim)
        self.downsample1 = nn.Conv1d(n_features, n_features, 4, stride=2, padding=1)

        self.resblock3 = ResidualBlock1D(n_features, n_features*2, time_emb_dim)
        self.resblock4 = ResidualBlock1D(n_features*2, n_features*2, time_emb_dim)
        self.downsample2 = nn.Conv1d(n_features*2, n_features*2, 4, stride=2, padding=1)

        self.resblock5 = ResidualBlock1D(n_features*2, n_features*4, time_emb_dim)
        self.resblock6 = ResidualBlock1D(n_features*4, n_features*4, time_emb_dim)
        self.downsample3 = nn.Conv1d(n_features*4, n_features*4, 4, stride=2, padding=1)

        # Middle
        self.mid1 = ResidualBlock1D(n_features*4, n_features*4, time_emb_dim)
        self.mid2 = ResidualBlock1D(n_features*4, n_features*4, time_emb_dim)

        # Up path
        self.upconv1 = nn.ConvTranspose1d(n_features*4, n_features*4, 4, stride=2, padding=1)
        self.upresblock1 = ResidualBlock1D(n_features*4 + n_features*4, n_features*4, time_emb_dim)

        self.upconv2 = nn.ConvTranspose1d(n_features*4, n_features*2, 4, stride=2, padding=1)
        self.upresblock2 = ResidualBlock1D(n_features*2 + n_features*2, n_features*2, time_emb_dim)

        self.upconv3 = nn.ConvTranspose1d(n_features*2, n_features, 4, stride=2, padding=1)
        self.upresblock3 = ResidualBlock1D(n_features + n_features, n_features, time_emb_dim)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, n_features),
            nn.SiLU(),
            nn.Conv1d(n_features, in_channels, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        # Downsampling
        h = self.init_conv(x)                                  # [B, 64, L]
        h1 = self.resblock1(h, t_emb)                          # [B, 64, L]
        h2 = self.resblock2(h1, t_emb)                         # [B, 64, L]
        skip0 = h2
        h3 = self.downsample1(skip0)                           # [B, 64, L/2]

        h4 = self.resblock3(h3, t_emb)                         # [B, 128, L/2]
        h5 = self.resblock4(h4, t_emb)                         # [B, 128, L/2]
        skip1 = h5
        h6 = self.downsample2(skip1)                           # [B, 128, L/4]

        h7 = self.resblock5(h6, t_emb)                         # [B, 256, L/4]
        h8 = self.resblock6(h7, t_emb)                         # [B, 256, L/4]
        skip2 = h8
        h9 = self.downsample3(skip2)                           # [B, 256, L/8]

        # Middle
        h = self.mid1(h9, t_emb)                               # [B, 256, L/8]
        h = self.mid2(h, t_emb)                                # [B, 256, L/8]

        # Upsampling
        h = self.upconv1(h)                                    # [B, 256, L/4]
        h = torch.cat([h, skip2], dim=1)                       # [B, 512, L/4]
        h = self.upresblock1(h, t_emb)                         # [B, 256, L/4]

        h = self.upconv2(h)                                    # [B, 128, L/2]
        h = torch.cat([h, skip1], dim=1)                       # [B, 256, L/2]
        h = self.upresblock2(h, t_emb)                         # [B, 128, L/2]

        h = self.upconv3(h)                                    # [B, 64, L]
        h = torch.cat([h, skip0], dim=1)                       # [B, 128, L]
        h = self.upresblock3(h, t_emb)                         # [B, 64, L]

        return self.final_conv(h)                              # [B, 1, L]

# --- U-Net-like 1D Denoising Network ---
class Simple1DUNet(nn.Module):
    def __init__(self, n_channels=1, n_features=64):
        super().__init__()
        self.down1 = nn.Conv1d(n_channels, n_features, 7, padding=3)
        self.down2 = nn.Conv1d(n_features, n_features*2, 3, stride=2, padding=1)
        self.mid = nn.Conv1d(n_features*2, n_features*2, 3, padding=1)
        self.up = nn.ConvTranspose1d(n_features*2, n_features, 4, stride=2, padding=1)
        self.final = nn.Conv1d(n_features, n_channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x, t):
        # t: [B] timestep for potential conditioning later
        h = self.act(self.down1(x))
        h = self.act(self.down2(h))
        h = self.act(self.mid(h))
        h = self.act(self.up(h))
        out = self.final(h)
        return out

# --- Noise Scheduler (beta schedule) ---
def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

# --- Diffusion Model Trainer ---
class Diffusion1D:
    def __init__(self, model, T=1000, device='cpu'):
        self.model = model.to(device)
        self.T = T
        self.device = device
        self.betas = make_beta_schedule(T).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        """Diffusion forward process: q(x_t|x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_hat = self.alpha_hat[t].sqrt().view(-1, 1, 1)
        sqrt_one_minus = (1 - self.alpha_hat[t]).sqrt().view(-1, 1, 1)
        return sqrt_alpha_hat * x0 + sqrt_one_minus * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_noisy, t.float())
        return F.mse_loss(noise_pred, noise, reduction='none').sum(dim=2).mean()

    def train(self, dataloader, epochs=10, lr=2e-4):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            total_loss = 0.
            for batch in dataloader:
                x = batch.to(self.device)
                t = torch.randint(0, self.T, (x.size(0),), device=self.device)
                loss = self.p_losses(x, t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader.dataset):.4f}")

    @torch.no_grad()
    def sample(self, n_samples, shape):
        x = torch.randn((n_samples, *shape), device=self.device)
        for t in reversed(range(self.T)):
            t_tensor = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            noise_pred = self.model(x, t_tensor.float())
            alpha = self.alphas[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.betas[t]
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = (1 / alpha.sqrt()) * (
                x - ((1 - alpha) / (1 - alpha_hat).sqrt()) * noise_pred
            ) + beta.sqrt() * noise
        # Clip x between -1 and +1
        x = torch.clamp(x, -1, 1)
        return x

