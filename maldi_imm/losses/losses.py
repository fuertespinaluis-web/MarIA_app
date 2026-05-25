import torch
import torch.nn.functional as F

def dino_cls_loss(student_logp_list, teacher_p_list):
    """
    Cross-view DINO loss on [CLS].
    student_logp_list: list[ (B, K) log-softmax ] ; teacher_p_list: list[ (B, K) probs ]
    """
    losses = []
    for i, logpS in enumerate(student_logp_list):
        for j, pT in enumerate(teacher_p_list):
            if i == j: continue
            losses.append(-(pT.detach() * logpS).sum(dim=-1).mean())
    return torch.stack(losses).mean() if losses else torch.tensor(0., device=student_logp_list[0].device)

def ibot_token_loss(logpS_tok, pT_tok, mask_idx, match_idx):
    """
    iBOT token loss for one view.
      logpS_tok: (L, K) log-softmax of student token head
      pT_tok   : (L, K) soft targets from teacher token head (same view)
      mask_idx : (L,) bool student mask (True = masked token)
      match_idx: (L,) int64 teacher alignment indices (-1 if no match)
    """
    sel = mask_idx & (match_idx >= 0)
    if not sel.any():
        return torch.tensor(0., device=logpS_tok.device)
    j = match_idx[sel].long()
    return -(pT_tok[j] * logpS_tok[sel]).sum(dim=-1).mean()

def koleo_loss(x_normed):
    """
    KoLeo on L2-normalized embeddings x_normed: (B, D).
    """
    with torch.no_grad():
        d = torch.cdist(x_normed, x_normed, p=2)
        d += torch.eye(len(d), device=d.device) * 1e9
    nn_d, _ = d.min(dim=1)
    return -torch.log(nn_d + 1e-12).mean()


def peak_presence_intensity_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    peak_threshold: float = 0.05,
    presence_weight: float = 1.0,
    intensity_weight: float = 1.0,
    bce_weight: float = 0.5,
    focal_weight: float = 0.5,
    focal_gamma_pos: float = 1.0,
    focal_gamma_neg: float = 2.0,
    focal_alpha_pos: float = 1.0,
    focal_alpha_neg: float = 1.0,
) -> torch.Tensor:
    """
    Hybrid loss:
      - Peak presence: BCE + focal on a binary mask (target >= threshold).
      - Peak intensity: MSE on sigmoid(logits) only where mask == 1.
    Returns per-sample loss (no reduction across batch).
    """
    target = target.detach()
    mask = (target >= float(peak_threshold)).float()
    probs = torch.sigmoid(logits)
    # presence loss (BCE + focal) per element
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    pt = probs * mask + (1.0 - probs) * (1.0 - mask)
    gamma = focal_gamma_pos * mask + focal_gamma_neg * (1.0 - mask)
    alpha = focal_alpha_pos * mask + focal_alpha_neg * (1.0 - mask)
    focal = alpha * torch.pow(1.0 - pt, gamma) * bce
    presence = bce_weight * bce + focal_weight * focal
    # intensity loss only where mask == 1
    if mask.sum() > 0:
        mse = (probs - target) ** 2
        intensity = (mse * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
    else:
        intensity = torch.zeros(logits.size(0), device=logits.device)
    presence = presence.sum(dim=1)
    return presence_weight * presence + intensity_weight * intensity
