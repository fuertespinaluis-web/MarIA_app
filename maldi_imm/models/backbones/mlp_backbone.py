import torch
import torch.nn as nn


class MLPBackbone(nn.Module):
    """
    Simple 4-layer MLP baseline for fixed-length vectors.

    - Input: (B, X) or any tensor that can be flattened to (B, X)
    - Output: (B, 768) embedding
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 768,
        hidden_dims=None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 768]
        if len(hidden_dims) != 3:
            raise ValueError("hidden_dims must have 3 values to make 4 layers total.")

        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.hidden_dims = [int(d) for d in hidden_dims]
        self.dropout = float(dropout)

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(self.hidden_dims[0])
        self.fc2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(self.hidden_dims[1])
        self.fc3 = nn.Linear(self.hidden_dims[1], self.hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(self.hidden_dims[2])
        self.fc4 = nn.Linear(self.hidden_dims[2], self.embed_dim)
        self.bn4 = nn.BatchNorm1d(self.embed_dim)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.drop(x)

        return x


class MLPDecoder(nn.Module):
    """
    4-layer MLP decoder that mirrors MLPBackbone in reverse.

    - Input: (B, 768) embedding
    - Output: (B, input_dim) reconstruction
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 768,
        hidden_dims=None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 768]
        if len(hidden_dims) != 3:
            raise ValueError("hidden_dims must have 3 values to make 4 layers total.")

        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.hidden_dims = [int(d) for d in hidden_dims]
        self.dropout = float(dropout)

        self.fc1 = nn.Linear(self.embed_dim, self.hidden_dims[2])
        self.bn1 = nn.BatchNorm1d(self.hidden_dims[2])
        self.fc2 = nn.Linear(self.hidden_dims[2], self.hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(self.hidden_dims[1])
        self.fc3 = nn.Linear(self.hidden_dims[1], self.hidden_dims[0])
        self.bn3 = nn.BatchNorm1d(self.hidden_dims[0])
        self.fc4 = nn.Linear(self.hidden_dims[0], self.input_dim)
        self.bn4 = nn.BatchNorm1d(self.input_dim)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.act(x)
        x = self.drop(x)

        return x
