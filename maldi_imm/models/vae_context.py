import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextConditionalEncoder(nn.Module):
    def __init__(
        self,
        encoder_net,
        y_species_dim: int,
        y_context_dim: int,
        species_embed_dim: int,
        context_embed_dim: int,
        y_country_dim: int = 0,
        logvar_clip=None,
    ):
        super().__init__()
        self.encoder = encoder_net
        self.species_embed = nn.Embedding(y_species_dim, species_embed_dim)
        self.context_embed = nn.Embedding(y_context_dim, context_embed_dim)
        self.country_embed = nn.Embedding(y_country_dim, context_embed_dim) if y_country_dim > 0 else None
        self.logvar_clip = logvar_clip

    def _context_embedding(self, y_context, y_country=None):
        if y_context is None:
            raise ValueError("y_context must be provided.")
        ctx = self.context_embed(y_context)
        if self.country_embed is not None and y_country is not None:
            ctx = ctx + self.country_embed(y_country)
        return ctx

    def get_cond(self, y_species, y_context, y_country=None):
        sp = self.species_embed(y_species)
        ctx = self._context_embedding(y_context, y_country=y_country)
        return torch.cat([sp, ctx], dim=1)

    def forward(self, x, y_species, y_context, y_country=None):
        cond = self.get_cond(y_species, y_context, y_country=y_country)
        h_e = self.encoder(x, cond)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        if self.logvar_clip is not None and self.logvar_clip > 0:
            log_var_e = torch.clamp(log_var_e, min=-self.logvar_clip, max=self.logvar_clip)
        return mu_e, log_var_e

    def sample(self, x=None, y_species=None, y_context=None, y_country=None, mu_e=None, log_var_e=None):
        if (mu_e is None) or (log_var_e is None):
            mu_e, log_var_e = self.forward(x, y_species, y_context, y_country=y_country)
        std = torch.exp(0.5 * log_var_e)
        eps = torch.randn_like(std)
        return mu_e + std * eps


class ContextConditionalDecoder(nn.Module):
    def __init__(
        self,
        decoder_net,
        y_species_dim: int,
        y_context_dim: int,
        species_embed_dim: int,
        context_embed_dim: int,
        y_country_dim: int = 0,
        likelihood: str = "bernoulli",
        fixed_var: float = 1.0,
        bern_loss: str = "bce",
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma_pos: float = 1.0,
        focal_gamma_neg: float = 2.0,
        focal_alpha_pos: float = 1.0,
        focal_alpha_neg: float = 1.0,
    ):
        super().__init__()
        self.decoder = decoder_net
        self.species_embed = nn.Embedding(y_species_dim, species_embed_dim)
        self.context_embed = nn.Embedding(y_context_dim, context_embed_dim)
        self.country_embed = nn.Embedding(y_country_dim, context_embed_dim) if y_country_dim > 0 else None
        self.likelihood = likelihood
        self.var = fixed_var
        self.bern_loss = bern_loss
        self.bce_weight = float(bce_weight)
        self.focal_weight = float(focal_weight)
        self.focal_gamma_pos = float(focal_gamma_pos)
        self.focal_gamma_neg = float(focal_gamma_neg)
        self.focal_alpha_pos = float(focal_alpha_pos)
        self.focal_alpha_neg = float(focal_alpha_neg)

    def _context_embedding(self, y_context, y_country=None):
        if y_context is None:
            raise ValueError("y_context must be provided.")
        ctx = self.context_embed(y_context)
        if self.country_embed is not None and y_country is not None:
            ctx = ctx + self.country_embed(y_country)
        return ctx

    def get_cond(self, y_species, y_context, y_country=None):
        sp = self.species_embed(y_species)
        ctx = self._context_embedding(y_context, y_country=y_country)
        return torch.cat([sp, ctx], dim=1)

    def decode(self, z, cond):
        return self.decoder(z, cond)

    def forward(self, z, y_species, y_context, y_country=None):
        cond = self.get_cond(y_species, y_context, y_country=y_country)
        logits = self.decode(z, cond)
        if self.likelihood == "gaussian":
            return logits
        return torch.sigmoid(logits)

    def sample(self, z, y_species, y_context, y_country=None):
        cond = self.get_cond(y_species, y_context, y_country=y_country)
        out = self.decode(z, cond)
        if self.likelihood == "gaussian":
            std = torch.sqrt(torch.tensor(self.var)).to(out.device)
            eps = torch.randn_like(out)
            return out + eps * std
        return torch.sigmoid(out)

    def _focal_loss_with_logits(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pt = probs * target + (1.0 - probs) * (1.0 - target)
        gamma = self.focal_gamma_pos * target + self.focal_gamma_neg * (1.0 - target)
        alpha = self.focal_alpha_pos * target + self.focal_alpha_neg * (1.0 - target)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return alpha * torch.pow(1.0 - pt, gamma) * bce

    def log_prob(self, x, z, cond):
        x_flat = x.view(x.size(0), -1)
        out = self.decode(z, cond)

        if self.likelihood == "gaussian":
            log_prob = -0.5 * ((x_flat - out) ** 2) / self.var - 0.5 * torch.log(
                2 * torch.pi * torch.tensor(self.var)
            )
            return log_prob.sum(dim=-1)
        if self.bern_loss == "bce_logits":
            recon = F.binary_cross_entropy_with_logits(out, x_flat, reduction="none")
        elif self.bern_loss == "focal":
            recon = self._focal_loss_with_logits(out, x_flat)
        elif self.bern_loss == "bce_focal":
            bce = F.binary_cross_entropy_with_logits(out, x_flat, reduction="none")
            focal = self._focal_loss_with_logits(out, x_flat)
            recon = self.bce_weight * bce + self.focal_weight * focal
        else:
            probs = torch.sigmoid(out)
            recon = F.binary_cross_entropy(probs, x_flat, reduction="none")
        return -recon.sum(dim=1)


class ContextConditionalPrior(nn.Module):
    def __init__(
        self,
        y_species_dim: int,
        y_context_dim: int,
        species_embed_dim: int,
        context_embed_dim: int,
        latent_dim: int,
        y_country_dim: int = 0,
        prior_hidden1: int = 128,
        prior_hidden2: int = 64,
        logvar_clip=None,
    ):
        super().__init__()
        self.species_embed = nn.Embedding(y_species_dim, species_embed_dim)
        self.context_embed = nn.Embedding(y_context_dim, context_embed_dim)
        self.country_embed = nn.Embedding(y_country_dim, context_embed_dim) if y_country_dim > 0 else None
        self.logvar_clip = logvar_clip

        cond_dim = species_embed_dim + context_embed_dim
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, prior_hidden1),
            nn.LeakyReLU(),
            nn.Linear(prior_hidden1, prior_hidden2),
            nn.LeakyReLU(),
            nn.Linear(prior_hidden2, 2 * latent_dim),
        )

    def _context_embedding(self, y_context, y_country=None):
        if y_context is None:
            raise ValueError("y_context must be provided.")
        ctx = self.context_embed(y_context)
        if self.country_embed is not None and y_country is not None:
            ctx = ctx + self.country_embed(y_country)
        return ctx

    def get_cond(self, y_species, y_context, y_country=None):
        sp = self.species_embed(y_species)
        ctx = self._context_embedding(y_context, y_country=y_country)
        return torch.cat([sp, ctx], dim=1)

    def forward(self, y_species, y_context, y_country=None):
        cond = self.get_cond(y_species, y_context, y_country=y_country)
        h = self.fc(cond)
        mu_p, log_var_p = torch.chunk(h, 2, dim=1)
        if self.logvar_clip is not None and self.logvar_clip > 0:
            log_var_p = torch.clamp(log_var_p, min=-self.logvar_clip, max=self.logvar_clip)
        return mu_p, log_var_p


class ContextConditionalVAE(nn.Module):
    def __init__(
        self,
        encoder_net,
        decoder_net,
        y_species_dim: int,
        y_context_dim: int,
        species_embed_dim: int,
        context_embed_dim: int,
        latent_dim: int,
        y_country_dim: int = 0,
        likelihood: str = "bernoulli",
        fixed_var: float = 1.0,
        beta: float = 1.0,
        logvar_clip=None,
        bern_loss: str = "bce",
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma_pos: float = 1.0,
        focal_gamma_neg: float = 2.0,
        focal_alpha_pos: float = 1.0,
        focal_alpha_neg: float = 1.0,
    ):
        super().__init__()
        self.encoder = ContextConditionalEncoder(
            encoder_net,
            y_species_dim,
            y_context_dim,
            species_embed_dim,
            context_embed_dim,
            y_country_dim=y_country_dim,
            logvar_clip=logvar_clip,
        )
        self.decoder = ContextConditionalDecoder(
            decoder_net,
            y_species_dim,
            y_context_dim,
            species_embed_dim,
            context_embed_dim,
            y_country_dim=y_country_dim,
            likelihood=likelihood,
            fixed_var=fixed_var,
            bern_loss=bern_loss,
            bce_weight=bce_weight,
            focal_weight=focal_weight,
            focal_gamma_pos=focal_gamma_pos,
            focal_gamma_neg=focal_gamma_neg,
            focal_alpha_pos=focal_alpha_pos,
            focal_alpha_neg=focal_alpha_neg,
        )
        self.prior = ContextConditionalPrior(
            y_species_dim,
            y_context_dim,
            species_embed_dim,
            context_embed_dim,
            latent_dim,
            y_country_dim=y_country_dim,
            logvar_clip=logvar_clip,
        )
        self.beta = beta

    def forward(self, x, y_species, y_context, y_country=None):
        mu_e, log_var_e = self.encoder(x, y_species, y_context, y_country=y_country)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        cond = self.decoder.get_cond(y_species, y_context, y_country=y_country)
        re = self.decoder.log_prob(x, z, cond)
        mu_p, log_var_p = self.prior(y_species, y_context, y_country=y_country)
        kl = -0.5 * torch.sum(
            1 + (log_var_e - log_var_p) - ((mu_e - mu_p).pow(2) + log_var_e.exp()) / log_var_p.exp(),
            dim=1,
        )
        nll = -(re - self.beta * kl).mean()
        return nll, kl.mean()

    def sample(self, y_species, y_context, y_country=None):
        mu_p, log_var_p = self.prior(y_species, y_context, y_country=y_country)
        std = torch.exp(0.5 * log_var_p)
        eps = torch.randn_like(std)
        z = mu_p + std * eps
        return self.decoder.sample(z, y_species, y_context, y_country=y_country)
