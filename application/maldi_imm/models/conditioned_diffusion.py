# src/models/simple_diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..SpectrumObject import SpectrumObject
import math
import numpy as np

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


# --- Basic Self-Attention for 1D (channels last) ---
class SelfAttention1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.qkv = nn.Conv1d(dim, dim * 3, 1)
        self.proj = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        # x: [B, C, L]
        x = self.norm(x)
        qkv = self.qkv(x)
        B, C3, L = qkv.shape
        q, k, v = qkv.chunk(3, dim=1)  # Each [B, C, L]
        attn = torch.einsum('bcl,bcm->blm', q, k) / (q.shape[1] ** 0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('blm,bcm->bcl', attn, v)
        return x + self.proj(out)

# --- Cross-Attention Conditioning ---
class CrossAttention1D(nn.Module):
    def __init__(self, query_dim, context_dim):
        super().__init__()
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, query, context):
        # query: [B, D], context: [B, N, D_ctx]
        q = self.to_q(query).unsqueeze(1)   # [B, 1, D]
        k = self.to_k(context)              # [B, N, D]
        v = self.to_v(context)              # [B, N, D]
        attn = torch.softmax((q * k).sum(-1, keepdim=True), dim=1)  # [B, N, 1]
        out = (attn * v).sum(1)  # [B, D]
        return self.out_proj(out)

# --- Embedding Utilities ---
class MetaEmbeddings(nn.Module):
    def __init__(self, genus_vocab, genus_species_vocab, hospital_vocab, country_vocab, quality_vocab, emb_dim):
        super().__init__()
        self.genus_emb = nn.Embedding(genus_vocab, emb_dim)
        self.genus_species_emb = nn.Embedding(genus_species_vocab, emb_dim)
        self.hospital_emb = nn.Embedding(hospital_vocab, emb_dim)
        self.country_emb = nn.Embedding(country_vocab, emb_dim)
        self.quality_emb = nn.Embedding(quality_vocab, emb_dim)
        self.emb_dim = emb_dim
        self.cross_att = CrossAttention1D(emb_dim, emb_dim)

    def forward(self, genus, genus_species, hospital, country, quality=None):
        g = self.genus_emb(genus)              # [B, D]
        gs = self.genus_species_emb(genus_species)  # [B, D]
        h = self.hospital_emb(hospital)        # [B, D]
        c = self.country_emb(country)          # [B, D]
        q = self.quality_emb(quality) if quality is not None else torch.zeros_like(g)
        context = torch.stack([g, h, c, q], dim=1)  # [B, 4, D]
        out = self.cross_att(gs, context)      # [B, D]
        return out

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
    

# --- helper: linear ramp position channel ---
def pos_channel(L, device, B):
    # [B, 1, L] with values linearly spaced 0..1
    pos = torch.linspace(0, 1, L, device=device).view(1, 1, L)
    return pos.expand(B, 1, L)

# --- UNet1D with Attention and Metadata Conditioning ---
class UNet1D(nn.Module):
    def __init__(
                    self, 
                    in_channels=1, 
                    n_features=64, 
                    time_emb_dim=128,
                    meta_emb_dim=256,
                    genus_vocab=20, genus_species_vocab=128, hospital_vocab=20, country_vocab=20, quality_vocab=2,
                    use_pos_channel=True
                ):
        super().__init__()
        self.use_pos_channel = use_pos_channel
        self.data_in_channels = in_channels  # number of data channels to output (1)
        in_ch_total = in_channels + (1 if use_pos_channel else 0)  # add pos channel

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.meta_embs = MetaEmbeddings(
            genus_vocab, genus_species_vocab, hospital_vocab, country_vocab, quality_vocab, meta_emb_dim
        )

        self.n_features = n_features
    
        # Initial conv now uses in_ch_total
        self.init_conv = nn.Conv1d(in_ch_total, n_features, 7, padding=3)

        # Down path
        self.resblock1 = ResidualBlock1D(n_features, n_features, time_emb_dim)
        self.resblock2 = ResidualBlock1D(n_features, n_features, time_emb_dim)
        self.attn1 = SelfAttention1D(n_features)
        self.downsample1 = nn.Conv1d(n_features, n_features, 4, stride=2, padding=1)

        self.resblock3 = ResidualBlock1D(n_features, n_features*2, time_emb_dim)
        self.resblock4 = ResidualBlock1D(n_features*2, n_features*2, time_emb_dim)
        self.attn2 = SelfAttention1D(n_features*2)
        self.downsample2 = nn.Conv1d(n_features*2, n_features*2, 4, stride=2, padding=1)

        self.resblock5 = ResidualBlock1D(n_features*2, n_features*4, time_emb_dim)
        self.resblock6 = ResidualBlock1D(n_features*4, n_features*4, time_emb_dim)
        self.attn3 = SelfAttention1D(n_features*4)
        self.downsample3 = nn.Conv1d(n_features*4, n_features*4, 4, stride=2, padding=1)

        # Middle
        self.mid1 = ResidualBlock1D(n_features*4, n_features*4, time_emb_dim)
        self.mid_attn = SelfAttention1D(n_features*4)
        self.mid2 = ResidualBlock1D(n_features*4, n_features*4, time_emb_dim)

        # Up path
        self.upconv1 = nn.ConvTranspose1d(n_features*4, n_features*4, 4, stride=2, padding=1)
        self.upresblock1 = ResidualBlock1D(n_features*4 + n_features*4, n_features*4, time_emb_dim)
        self.upattn1 = SelfAttention1D(n_features*4)

        self.upconv2 = nn.ConvTranspose1d(n_features*4, n_features*2, 4, stride=2, padding=1)
        self.upresblock2 = ResidualBlock1D(n_features*2 + n_features*2, n_features*2, time_emb_dim)
        self.upattn2 = SelfAttention1D(n_features*2)

        self.upconv3 = nn.ConvTranspose1d(n_features*2, n_features, 4, stride=2, padding=1)
        self.upresblock3 = ResidualBlock1D(n_features + n_features, n_features, time_emb_dim)
        self.upattn3 = SelfAttention1D(n_features)

        # Cross-attention conditioning block for upsampling path
        self.cond_proj = nn.Linear(meta_emb_dim, n_features*4)

        # Final conv outputs the original data channels (NOT including pos)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, n_features),
            nn.SiLU(),
            nn.Conv1d(n_features, self.data_in_channels, 3, padding=1)
        )


    def forward(self, x, t, genus, genus_species, hospital, country, quality=None):
        # concat position channel if enabled
        if self.use_pos_channel:
            B, C, L = x.shape
            pos = pos_channel(L, x.device, B)          # [B,1,L], no grad needed
            x = torch.cat([x, pos], dim=1)             # [B, C+1, L]

        t_emb = self.time_mlp(t)
        cond = self.meta_embs(genus, genus_species, hospital, country, quality)  # [B, D]
        
        # Downsampling
        h = self.init_conv(x)                                   # [B, 64, L]
        h1 = self.resblock1(h, t_emb)
        h2 = self.resblock2(h1, t_emb)
        h2 = self.attn1(h2)
        skip0 = h2
        h3 = self.downsample1(skip0)

        h4 = self.resblock3(h3, t_emb)
        h5 = self.resblock4(h4, t_emb)
        h5 = self.attn2(h5)
        skip1 = h5
        h6 = self.downsample2(skip1)

        h7 = self.resblock5(h6, t_emb)
        h8 = self.resblock6(h7, t_emb)
        h8 = self.attn3(h8)
        skip2 = h8
        h9 = self.downsample3(skip2)

        # Middle
        h = self.mid1(h9, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Upsampling (conditioned)
        h = self.upconv1(h)  # [B, 256, L/4]
        cond1 = self.cond_proj(cond)[:, :, None].expand(-1, -1, h.shape[2])
        h = h + cond1
        h = torch.cat([h, skip2], dim=1)
        h = self.upresblock1(h, t_emb)
        h = self.upattn1(h)

        h = self.upconv2(h)
        cond2 = self.cond_proj(cond)[:, :self.n_features*2, None].expand(-1, -1, h.shape[2])
        h = h + cond2
        h = torch.cat([h, skip1], dim=1)
        h = self.upresblock2(h, t_emb)
        h = self.upattn2(h)

        h = self.upconv3(h)
        cond3 = self.cond_proj(cond)[:, :self.n_features, None].expand(-1, -1, h.shape[2])
        h = h + cond3
        h = torch.cat([h, skip0], dim=1)
        h = self.upresblock3(h, t_emb)
        h = self.upattn3(h)

        return self.final_conv(h)

# --- Noise Schedulers ---
def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02, schedule="cosine", s=0.008):
    """
    Returns betas of length T.
    schedule: "linear" or "cosine"
    """
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        return betas

    if schedule == "cosine":
        # ᾱ_t for t=0..T
        t = torch.linspace(0, 1, T+1, dtype=torch.float64)
        f = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
        f = f / f[0]
        alpha_bar = f
        # betas_t = 1 - ᾱ_t / ᾱ_{t-1}
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = betas.clamp(1e-5, 0.999).to(torch.float32)
        return betas

    raise ValueError(f"Unknown schedule: {schedule}")


# --- Diffusion Model Trainer ---
class Diffusion1D:
    """Supports v-prediction (with optional BCE) or epsilon-prediction."""
    def __init__(self, model, T=1000, device='cpu', background_suppressor=None, loss_mode: str = "v_bce"):
        self.model = model.to(device)
        self.T = T
        self.device = device
        self.betas = make_beta_schedule(T).to(device)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        self.background_suppressor = background_suppressor
        self.loss_mode = loss_mode  # "v_bce", "v", or "eps"

    def _predict_x0_from_v(self, x_t, t_long, v_pred):
        """
        Recover x0 from v-pred (see Sec. 3 of v-pred paper).
        x0 = c1 * x_t - c2 * v
        where c1 = sqrt(alpha_hat_t), c2 = sqrt(1 - alpha_hat_t).
        """
        c1 = self._sqrt_alpha_hat(t_long)            # (B,1,1)
        c2 = self._sqrt_one_minus_alpha_hat(t_long)  # (B,1,1)
        return c1 * x_t - c2 * v_pred

    def _sqrt_alpha_hat(self, t):
        return self.alpha_hat[t].sqrt().view(-1, 1, 1)

    def _sqrt_one_minus_alpha_hat(self, t):
        return (1. - self.alpha_hat[t]).sqrt().view(-1, 1, 1)

    def q_sample(self, x0, t, noise=None):
        """Diffusion forward process: q(x_t|x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_hat = self.alpha_hat[t].sqrt().view(-1, 1, 1)
        sqrt_one_minus = (1 - self.alpha_hat[t]).sqrt().view(-1, 1, 1)
        return sqrt_alpha_hat * x0 + sqrt_one_minus * noise

    def p_losses(
        self, x0, t, genus, genus_species, hospital, country, quality=None,
        *,
        aux_lambda=1,        # how much to mix aux loss with v-loss (only for v_bce)
        return_components=False,
        loss_mode: str | None = None,
    ):
        """
        Train-time loss with selectable mode:
          - "v_bce": v-pred with auxiliary BCE on signal mask (default)
          - "v": v-pred only (MSE)
          - "eps": epsilon-prediction (MSE)
        """
        mode = loss_mode or self.loss_mode
        B = x0.size(0)
        noise   = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)

        if mode == "eps":
            eps_target = noise
            eps_pred = self.model(x_noisy, t.float(), genus, genus_species, hospital, country, quality)
            L_eps = F.mse_loss(eps_pred, eps_target, reduction='none').sum(dim=2).mean()
            if return_components:
                return L_eps, L_eps.detach(), torch.zeros((), device=L_eps.device)
            return L_eps

        # ----- targets for v-pred -----
        c1 = self._sqrt_alpha_hat(t)              # sqrt(alpha_hat_t)
        c2 = self._sqrt_one_minus_alpha_hat(t)    # sqrt(1 - alpha_hat_t)
        v_target = c1 * noise - c2 * x0           # v = sqrt(â)*eps - sqrt(1-â)*x0

        # ----- model predicts v -----
        v_pred = self.model(x_noisy, t.float(), genus, genus_species, hospital, country, quality)

        # ----- primary v-loss-----
        L_v = F.mse_loss(v_pred, v_target, reduction='none').sum(dim=2).mean()

        if mode == "v":
            if return_components:
                return L_v, L_v.detach(), torch.zeros((), device=L_v.device)
            return L_v

        # mode == "v_bce"
        x0_hat = self._predict_x0_from_v(x_noisy, t, v_pred)
        signal_mask = (x0 > 0).float()
        x0_hat_prob = torch.clamp(x0_hat, min=1e-6, max=1 - 1e-6)
        bce_loss = F.binary_cross_entropy(x0_hat_prob, signal_mask)
        L_aux = bce_loss
        total_loss = L_v + aux_lambda * L_aux

        if return_components:
            return total_loss, L_v.detach(), L_aux.detach()
        return total_loss



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
    def sample(
        self,
        n_samples: int,
        shape,  # (C, L) or just L
        genus_idx,
        genus_species_idx,
        hospital_idx,
        country_idx,
        quality_idx=1,
        mode: str = "stochastic",  # "stochastic" (default) or "deterministic"
    ):
        """
        Returns a torch.Tensor of shape (B, C, L) on self.device, values in [0, 1].

        mode:
            - "stochastic": DDPM-style sampling with noise at each step (default)
            - "deterministic": same update, but without per-step noise
        """

        device = self.device

        # Normalize shape → (C, L)
        if isinstance(shape, int):
            C, L = 1, int(shape)
        else:
            if len(shape) == 1:
                C, L = 1, int(shape[0])
            elif len(shape) == 2:
                C, L = int(shape[0]), int(shape[1])
            else:
                raise ValueError(f"shape must be (L) or (C, L); got {shape}")

        # Helper: turn cond into LongTensor[B] on device, broadcasting if needed
        def as_long_vec(x):
            if torch.is_tensor(x):
                v = x.to(device=device, dtype=torch.long).view(-1)
            elif isinstance(x, (list, tuple)):
                v = torch.tensor(x, dtype=torch.long, device=device).view(-1)
            else:  # int/np scalar
                v = torch.tensor([int(x)], dtype=torch.long, device=device)
            # broadcast if length 1
            if v.numel() == 1 and n_samples > 1:
                v = v.expand(n_samples)
            if v.numel() != n_samples:
                raise ValueError(f"Condition length {v.numel()} != n_samples {n_samples}")
            return v

        genus_idx         = as_long_vec(genus_idx)
        genus_species_idx = as_long_vec(genus_species_idx)
        hospital_idx      = as_long_vec(hospital_idx)
        country_idx       = as_long_vec(country_idx)
        quality_idx       = as_long_vec(quality_idx)

        if mode not in ("stochastic", "deterministic"):
            raise ValueError(f"mode must be 'stochastic' or 'deterministic', got {mode}")

        # Reverse diffusion
        x = torch.randn((n_samples, C, L), device=device)
        for t in reversed(range(self.T)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            t_in = t_tensor.float()

            if self.loss_mode == "eps":
                eps_pred = self.model(
                    x, t_in, genus_idx, genus_species_idx,
                    hospital_idx, country_idx, quality_idx
                )
            else:
                v_pred = self.model(
                    x, t_in, genus_idx, genus_species_idx,
                    hospital_idx, country_idx, quality_idx
                )
                # Convert v → eps
                # eps = sqrt(1-â)*x_t + sqrt(â)*v
                c1 = self._sqrt_alpha_hat(t_tensor)
                c2 = self._sqrt_one_minus_alpha_hat(t_tensor)
                eps_pred = c2 * x + c1 * v_pred

            alpha     = self.alphas[t]
            alpha_hat = self.alpha_hat[t]
            beta      = self.betas[t]

            if mode == "stochastic":
                noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            else:  # deterministic
                noise = torch.zeros_like(x)

            # SAME update as before but with eps_pred
            x = (1 / alpha.sqrt()) * (
                    x - ((1 - alpha) / (1 - alpha_hat).sqrt()) * eps_pred
                ) + beta.sqrt() * noise

        return x  # (B, C, L) tensor (no clamping; typically in [-1,1] if trained that way)

    
 # --- Old Diffusion Model Trainer (for reference) ---
class Diffusion1Dold:
        def __init__(self, model, T=1000, device='cpu', background_suppressor=None):
            self.model = model.to(device)
            self.T = T
            self.device = device
            self.betas = make_beta_schedule(T).to(device)
            self.alphas = 1. - self.betas
            self.alpha_hat = torch.cumprod(self.alphas, dim=0)
            self.background_suppressor = background_suppressor

        def q_sample(self, x0, t, noise=None):
            """Diffusion forward process: q(x_t|x_0)"""
            if noise is None:
                noise = torch.randn_like(x0)
            sqrt_alpha_hat = self.alpha_hat[t].sqrt().view(-1, 1, 1)
            sqrt_one_minus = (1 - self.alpha_hat[t]).sqrt().view(-1, 1, 1)
            return sqrt_alpha_hat * x0 + sqrt_one_minus * noise

        def p_losses(self, x0, t, genus, genus_species, hospital, country, quality=None):
            noise = torch.randn_like(x0)
            x_noisy = self.q_sample(x0, t, noise)
            noise_pred = self.model(x_noisy, t.float(), genus, genus_species, hospital, country, quality)
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
        def sample(
                    self,
                    n_samples: int,
                    shape,  # (C, L) or just L
                    genus_idx,
                    genus_species_idx,
                    hospital_idx,
                    country_idx,
                    quality_idx=1,
                ):
            """
            Returns a torch.Tensor of shape (B, C, L) on self.device, values in [-1, 1].
            """

            device = self.device

            # Normalize shape → (C, L)
            if isinstance(shape, int):
                C, L = 1, int(shape)
            else:
                if len(shape) == 1:
                    C, L = 1, int(shape[0])
                elif len(shape) == 2:
                    C, L = int(shape[0]), int(shape[1])
                else:
                    raise ValueError(f"shape must be (L) or (C, L); got {shape}")

            # Helper: turn cond into LongTensor[B] on device, broadcasting if needed
            def as_long_vec(x):
                if torch.is_tensor(x):
                    v = x.to(device=device, dtype=torch.long).view(-1)
                elif isinstance(x, (list, tuple)):
                    v = torch.tensor(x, dtype=torch.long, device=device).view(-1)
                else:  # int/np scalar
                    v = torch.tensor([int(x)], dtype=torch.long, device=device)
                # broadcast if length 1
                if v.numel() == 1 and n_samples > 1:
                    v = v.expand(n_samples)
                if v.numel() != n_samples:
                    raise ValueError(f"Condition length {v.numel()} != n_samples {n_samples}")
                return v

            genus_idx        = as_long_vec(genus_idx)
            genus_species_idx= as_long_vec(genus_species_idx)
            hospital_idx     = as_long_vec(hospital_idx)
            country_idx      = as_long_vec(country_idx)
            quality_idx      = as_long_vec(quality_idx)

            # Reverse diffusion
            x = torch.randn((n_samples, C, L), device=device)
            for t in reversed(range(self.T)):
                t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
                # if your model expects float timesteps, cast here:
                t_in = t_tensor.float()
                noise_pred = self.model(
                    x, t_in, genus_idx, genus_species_idx, hospital_idx, country_idx, quality_idx
                )
                alpha = self.alphas[t]
                alpha_hat = self.alpha_hat[t]
                beta = self.betas[t]
                noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
                x = (1 / alpha.sqrt()) * (x - ((1 - alpha) / (1 - alpha_hat).sqrt()) * noise_pred) + beta.sqrt() * noise

            x = x.clamp(0.0, 1.0)  # (B, C, L)

            return x  # (B, C, L) tensor
