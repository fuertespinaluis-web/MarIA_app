import torch

class EMAHelper:
    def __init__(self, mu0=0.996, mu_final=0.9995):
        self.mu0, self.mu_final = mu0, mu_final
    def momentum(self, step, max_steps):
        t = min(step / max_steps, 1.0)
        return self.mu0 + (self.mu_final - self.mu0) * t
    @torch.no_grad()
    def step(self, teacher, student, step, max_steps):
        m = self.momentum(step, max_steps)
        for tp, sp in zip(teacher.parameters(), student.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=(1.0 - m))

def sinkhorn_knopp(logits, iters=3, eps=1e-12):
    """
    Balanced probabilities ala SwAV/DINO for a batch.
    logits: (B, K) → returns (B, K) probs.
    """
    Q = torch.softmax(logits, dim=-1).T  # (K,B)
    K, B = Q.shape
    Q = Q / (Q.sum() + eps)
    r = torch.ones(K, device=Q.device) / K
    c = torch.ones(B, device=Q.device) / B
    for _ in range(iters):
        r = r / (Q @ c + eps)
        c = c / (Q.T @ r + eps)
    Q = torch.diag(r) @ Q @ torch.diag(c)
    return (Q / Q.sum(dim=0, keepdim=True)).T  # (B,K)
