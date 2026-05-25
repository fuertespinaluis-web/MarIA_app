import torch
import torch.nn as nn
from tqdm import tqdm


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        Generic one-layer feed-forward network that embeds input_dim to emb_dim.
        """
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        if self.input_dim == 0:
            batch = x.shape[0] if hasattr(x, "shape") and len(x.shape) > 0 else 1
            emb_dim = self.model[-1].out_features if hasattr(self.model[-1], "out_features") else self.model[-1].in_features
            return x.new_zeros((batch, emb_dim))
        try:
            x = x.view(x.size(0), -1)
        except Exception:
            pass
        return self.model(x)


class ContextUnet1D(nn.Module):
    """
    Asymmetric 1D U-Net for diffusion, configurable to match the original MALDIGen baseline.
    """
    def __init__(
        self,
        in_channels=1,
        n_feat=64,
        n_cfeat=6,
        length=6000,
        n_blocks=2,
        norm_groups=8,
        kernel_size=4,
        logger=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.length = length
        self.n_blocks = n_blocks
        self.logger = logger

        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, n_feat, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, n_feat),
            nn.ReLU(),
        )

        self.down_blocks = nn.ModuleList()
        in_ch = n_feat
        down_channels = [n_feat]
        for i in range(n_blocks):
            out_ch = in_ch if i < n_blocks - 1 else 2 * in_ch
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=2, padding=1),
                    nn.GroupNorm(norm_groups, out_ch),
                    nn.ReLU(),
                )
            )
            down_channels.append(out_ch)
            in_ch = out_ch

        self.to_vec = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.GELU(),
        )

        self.timeembeds = nn.ModuleList()
        self.contextembeds = nn.ModuleList()
        for i in range(self.n_blocks):
            out_dim = 2 * self.n_feat if i == 0 else self.n_feat
            self.timeembeds.append(EmbedFC(1, out_dim))
            self.contextembeds.append(EmbedFC(self.n_cfeat, out_dim))

        self.up0 = nn.Sequential(
            nn.ConvTranspose1d(
                2 * n_feat,
                2 * n_feat,
                kernel_size=length // (2 ** n_blocks),
                stride=length // (2 ** n_blocks),
            ),
            nn.GroupNorm(norm_groups, 2 * n_feat),
            nn.ReLU(),
        )

        self.up_blocks = nn.ModuleList()
        in_ch = 2 * n_feat
        for i in range(n_blocks):
            skip_ch = down_channels[-(i + 1)]
            out_ch = n_feat
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(in_ch + skip_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(norm_groups, out_ch),
                    nn.ReLU(),
                )
            )
            in_ch = out_ch

        self.out = nn.Sequential(
            nn.Conv1d(2 * n_feat, n_feat, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, n_feat),
            nn.ReLU(),
            nn.Conv1d(n_feat, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t, c=None):
        x = self.init_conv(x)
        downs = [x]
        for down in self.down_blocks:
            downs.append(down(downs[-1]))

        hiddenvec = self.to_vec(downs[-1])

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat, device=x.device, dtype=x.dtype)

        up = self.up0(hiddenvec)

        for i, up_block in enumerate(self.up_blocks):
            skip = downs[-(i + 1)]
            cemb = self.contextembeds[i](c).view(c.size(0), -1, 1)
            temb = self.timeembeds[i](t).view(t.size(0), -1, 1)
            up = cemb * up + temb

            if up.shape[-1] != skip.shape[-1]:
                min_len = min(up.shape[-1], skip.shape[-1])
                up = up[..., :min_len]
                skip = skip[..., :min_len]

            up = up_block(torch.cat([up, skip], dim=1))

        out = self.out(torch.cat([up, downs[0]], dim=1))
        return out


def generate_spectra_per_label_ddpm(
    model,
    label_correspondence,
    n_samples,
    timesteps,
    a_t,
    b_t,
    ab_t,
    logger,
    device,
):
    """
    Generate n_samples per label using the trained diffusion model.
    """
    model.eval()
    results = {}
    num_classes = len(label_correspondence)

    for label_id, label_name in label_correspondence.items():
        if logger:
            logger.info(f"Generating diffusion samples for label: {label_name}")

        c = torch.zeros(n_samples, num_classes, device=device)
        c[:, label_id] = 1.0

        L = model.length
        x = torch.randn(n_samples, model.in_channels, L, device=device)

        with torch.no_grad():
            for t_inv in tqdm(range(timesteps, 0, -1), desc=f"Sampling {label_name}"):
                t = torch.full((n_samples,), t_inv, device=device, dtype=torch.long)
                t_norm = (t.float() / float(timesteps)).view(-1, 1)
                eps = model(x, t_norm, c)
                ab = ab_t[t].view(n_samples, 1, 1)
                a = a_t[t].view(n_samples, 1, 1)
                b = b_t[t].view(n_samples, 1, 1)
                x = (x - (b / (1 - ab).sqrt()) * eps) / a.sqrt()
                if t_inv > 1:
                    x += b.sqrt() * torch.randn_like(x)

        results[label_name] = x.detach().cpu()

    return results
