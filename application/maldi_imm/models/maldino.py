# maldi_imm/models/maldino.py
import torch, torch.nn as nn
from .backbones.maldino_backbone import MaldinoBackbone
from ..heads.head import ProjectionHead
from ..utils.utils import EMAHelper

class StudentTeacher(nn.Module):
    def __init__(self, d_model=512, depth=12, n_heads=8, d_ff=2048, dropout=0.1,
                 pos_fourier_dim=65, max_tokens=512, k_cls=4096, k_tok=8192,
                 ema_mu0=0.996, ema_mu_final=0.9995):
        super().__init__()
        # student/teacher backbones
        self.student = MaldinoBackbone(d_model, depth, n_heads, d_ff, dropout, pos_fourier_dim, max_tokens)
        self.teacher = MaldinoBackbone(d_model, depth, n_heads, d_ff, 0.0,     pos_fourier_dim, max_tokens)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters(): p.requires_grad_(False)
        self.teacher.eval()
        # heads (unshared)
        self.headS_cls = ProjectionHead(d_model, k_cls)
        self.headT_cls = ProjectionHead(d_model, k_cls)
        self.headS_tok = ProjectionHead(d_model, k_tok)
        self.headT_tok = ProjectionHead(d_model, k_tok)
        # EMA
        self.ema = EMAHelper(mu0=ema_mu0, mu_final=ema_mu_final)

    @torch.no_grad()
    def ema_update(self, step, max_steps):
        self.ema.step(self.teacher, self.student, step, max_steps)

def build_maldino(**kwargs):
    return StudentTeacher(**kwargs)
