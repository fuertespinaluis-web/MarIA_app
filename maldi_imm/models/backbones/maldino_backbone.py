import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Positional features (Fourier)
# ----------------------------
def fourier_posenc_mz(
    mz: torch.Tensor,
    num_frequencies: int = 32,
    use_log_mz: bool = True,
    min_mz: Optional[float] = None,
    max_mz: Optional[float] = None,
    include_raw: bool = True,
) -> torch.Tensor:
    """
    Compute Fourier positional encoding for m/z.

    Args:
        mz: (B, L) m/z in Daltons.
        num_frequencies: number of frequency bands (sin+cos => 2*num_frequencies dims).
        use_log_mz: use log(mz) before scaling (often stabilizes across range).
        min_mz, max_mz: global dataset bounds for stable scaling. If None, infer per-batch.
        include_raw: append the raw normalized coordinate as an extra 1-dim channel.

    Returns:
        pos: (B, L, 2*num_frequencies + (1 if include_raw else 0)) positional features.
    """
    assert mz.ndim == 2, "mz should be (B, L)"
    eps = 1e-8
    x = torch.log(mz + eps) if use_log_mz else mz

    # Normalize to [0,1] using provided bounds if given (recommended to pass dataset-level min/max)
    if min_mz is None or max_mz is None:
        x_min = x.amin(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
    else:
        x_min = torch.tensor(math.log(min_mz + eps) if use_log_mz else min_mz, device=mz.device)
        x_max = torch.tensor(math.log(max_mz + eps) if use_log_mz else max_mz, device=mz.device)
        x_min = x_min.view(1, 1).expand_as(x)
        x_max = x_max.view(1, 1).expand_as(x)

    x01 = (x - x_min) / (x_max - x_min + eps)  # (B, L) in [0,1]

    # Frequencies (log-spaced works well)
    freqs = torch.logspace(0, math.log10(1024.0), steps=num_frequencies, device=mz.device)  # 1..1024
    angles = 2 * math.pi * x01.unsqueeze(-1) * freqs  # (B, L, F)

    pos_sin = torch.sin(angles)
    pos_cos = torch.cos(angles)
    feats = torch.cat([pos_sin, pos_cos], dim=-1)  # (B, L, 2F)

    if include_raw:
        feats = torch.cat([feats, x01.unsqueeze(-1)], dim=-1)  # add 1 dim

    return feats


# ----------------------------
# Transformer building blocks
# ----------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) with True=PAD to ignore
        attn_mask: Optional[torch.Tensor] = None,         # (T, T) or (B, T, T), optional
    ) -> torch.Tensor:
        # Self-attention (Pre-LN)
        h = self.norm1(x)
        h, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,  # True entries are ignored, take care of this! contra-intuitive
            attn_mask=attn_mask
        )
        x = x + self.dropout1(h)

        # FFN (Pre-LN)
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.dropout2(h)
        return x


# ----------------------------
# MALDINO Backbone (encoder)
# ----------------------------
class MaldinoBackbone(nn.Module):
    """
    Minimal MALDI Transformer backbone for self-supervised pretraining.

    - Inputs: intensity (B, L, 1) and m/z (B, L)
    - Positional encoding: Fourier features of m/z
    - Returns: (cls_emb, tok_emb)
        cls_emb: (B, d_model)  spectrum-level embedding (for DINO [CLS] head)
        tok_emb: (B, L, d_model) per-peak embeddings (for iBOT token head)

    Notes:
    - Provide attention_mask with True for VALID tokens, False for padding.
    - Padding positions are ignored in attention via key_padding_mask.
    """

    def __init__(
        self,
        d_model: int = 512,
        depth: int = 12,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pos_fourier_dim: int = 65,  # 2*32 + 1 raw
        use_log1p_intensity: bool = True,
        max_tokens: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.use_log1p_intensity = use_log1p_intensity

        # Learned [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Project [ intensity (1) + pos_enc (pos_fourier_dim) ] -> d_model
        self.input_proj = nn.Linear(1 + pos_fourier_dim, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    @torch.no_grad()
    def _prep_inputs(
        self,
        intensity: torch.Tensor,  # (B, L) or (B, L, 1)
        mz: torch.Tensor,         # (B, L)
        pos_feats: Optional[torch.Tensor] = None,  # (B, L, P) if precomputed
        **pos_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (features, pos_feats) with shapes (B, L, 1) and (B, L, P)."""
        if intensity.ndim == 2:
            intensity = intensity.unsqueeze(-1)
        assert intensity.ndim == 3 and intensity.shape[:2] == mz.shape, "shape mismatch"

        x = intensity
        if self.use_log1p_intensity:
            x = torch.log1p(torch.clamp(x, min=0))

        if pos_feats is None:
            pos_feats = fourier_posenc_mz(mz, **pos_kwargs)  # (B, L, P)
        return x, pos_feats

    def forward(
        self,
        intensity: torch.Tensor,              # (B, L) or (B, L, 1)
        mz: torch.Tensor,                     # (B, L)
        attention_mask: Optional[torch.Tensor] = None,  # (B, L) True=valid, False=pad
        pos_feats: Optional[torch.Tensor] = None,       # optional precomputed pos enc
        **pos_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            intensity: (B, L) or (B, L, 1) intensities.
            mz: (B, L) m/z values (Da).
            attention_mask: (B, L) bool, True for VALID tokens (not padding).
            pos_feats: (B, L, P) optional Fourier features; else computed on the fly.
            pos_kwargs: passed to fourier_posenc_mz (e.g., num_frequencies, use_log_mz, min_mz, max_mz)

        Returns:
            cls_emb: (B, d_model)
            tok_emb: (B, L, d_model)
        """
        B, L = mz.shape
        assert L <= self.max_tokens, f"L={L} exceeds max_tokens={self.max_tokens}"

        x_feat, x_pos = self._prep_inputs(intensity, mz, pos_feats, **pos_kwargs)  # (B,L,1),(B,L,P)
        x = torch.cat([x_feat, x_pos], dim=-1)  # (B, L, 1+P)
        x = self.input_proj(x)                  # (B, L, d)

        # prepend [CLS]
        cls = self.cls_token.expand(B, 1, self.d_model)   # (B,1,d)
        x = torch.cat([cls, x], dim=1)                    # (B, 1+L, d)

        # key_padding_mask expects True=PAD -> we invert if attention_mask uses True=valid
        key_padding_mask = None
        if attention_mask is not None:
            assert attention_mask.shape == (B, L)
            pad = ~attention_mask  # True where padding
            # add False for the [CLS] token (never pad)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], dim=1)
            key_padding_mask = pad  # (B, 1+L)

        # Transformer
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        cls_emb = x[:, 0]      # (B, d)
        tok_emb = x[:, 1:]     # (B, L, d)
        return cls_emb, tok_emb
