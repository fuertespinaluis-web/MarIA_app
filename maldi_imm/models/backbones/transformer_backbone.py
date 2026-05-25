from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.glu(x, dim=-1)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = GLUFeedForward(d_model, d_ff, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        x = x + self.dropout1(h)

        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.dropout2(h)
        return x


class TransformerBackbone(nn.Module):
    """
    Transformer baseline with GLU feedforward blocks.

    - 6 layers, 8 heads, dropout 0.2 by default.
    - Input: (B, L) or (B, L, D_in)
    - Output: (B, 768) embedding (CLS token).
    """

    def __init__(
        self,
        input_dim: int = 1,
        embed_dim: int = 768,
        depth: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.2,
        max_tokens: int = 512,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)
        self.max_tokens = int(max_tokens)
        self.use_cls_token = bool(use_cls_token)

        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)
        self.input_dropout = nn.Dropout(self.dropout)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim) * 0.02)
            pos_len = self.max_tokens + 1
        else:
            pos_len = self.max_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, pos_len, self.embed_dim) * 0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.embed_dim, self.n_heads, self.d_ff, dropout=self.dropout) for _ in range(self.depth)]
        )
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # (B, L) True=valid, False=pad
    ) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError("x must be (B, L) or (B, L, D)")

        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_tokens:
            raise ValueError(f"seq_len={seq_len} exceeds max_tokens={self.max_tokens}")

        x = self.input_proj(x)

        if self.use_cls_token:
            cls = self.cls_token.expand(batch_size, 1, self.embed_dim)
            x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.input_dropout(x)

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError("attention_mask must be (B, L)")
            pad = ~attention_mask
            if self.use_cls_token:
                cls_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=pad.device)
                pad = torch.cat([cls_pad, pad], dim=1)
            key_padding_mask = pad

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        if self.use_cls_token:
            return x[:, 0]
        if attention_mask is None:
            return x.mean(dim=1)
        mask = attention_mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (x * mask).sum(dim=1) / denom


class TransformerDecoder(nn.Module):
    """
    Transformer decoder that mirrors TransformerBackbone blocks.

    - Input: (B, D) latent or (B, L, D) token embeddings
    - Output: (B, L, output_dim) reconstruction
    """

    def __init__(
        self,
        output_dim: int = 1,
        embed_dim: int = 768,
        depth: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.dropout = float(dropout)
        self.max_tokens = int(max_tokens)

        self.latent_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.input_dropout = nn.Dropout(self.dropout)

        self.pos_embed = nn.Parameter(torch.randn(1, self.max_tokens, self.embed_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.embed_dim, self.n_heads, self.d_ff, dropout=self.dropout) for _ in range(self.depth)]
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.output_dim)

    def forward(
        self,
        z: torch.Tensor,
        seq_len: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,  # (B, L) True=valid, False=pad
    ) -> torch.Tensor:
        if z.ndim == 2:
            if seq_len is None:
                raise ValueError("seq_len is required when z is (B, D).")
            if seq_len > self.max_tokens:
                raise ValueError(f"seq_len={seq_len} exceeds max_tokens={self.max_tokens}")
            x = self.latent_proj(z)
            x = x.unsqueeze(1).expand(z.size(0), seq_len, self.embed_dim)
        elif z.ndim == 3:
            x = self.latent_proj(z)
            seq_len = x.size(1)
            if seq_len > self.max_tokens:
                raise ValueError(f"seq_len={seq_len} exceeds max_tokens={self.max_tokens}")
        else:
            raise ValueError("z must be (B, D) or (B, L, D)")

        x = x + self.pos_embed[:, :seq_len, :]
        x = self.input_dropout(x)

        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (x.size(0), seq_len):
                raise ValueError("attention_mask must be (B, L)")
            key_padding_mask = ~attention_mask

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        x = self.out_proj(x)
        return x
