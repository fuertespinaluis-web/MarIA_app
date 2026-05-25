import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    """2-layer MLP head for both CLS (DINO) and TOK (iBOT)."""
    def __init__(self, in_dim, out_dim, hidden=2048, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)  # logits
