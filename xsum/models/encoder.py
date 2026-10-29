"""Permutation-invariant set encoder over particle tokens with a global context token."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..constants import N_CLASSES

MOM_SCALE = 1000.0  # MeV -> GeV-ish before asinh


def momentum_features(mom):
    """(px,py,pz) in MeV -> 7 scale-friendly features: asinh(p/scale) x3, log|p|, unit direction x3."""
    p = mom.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    return torch.cat([torch.asinh(mom / MOM_SCALE), torch.log(p / MOM_SCALE), mom / p], -1)


class SetEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, d_ff=256, dropout=0.0, n_ctx=5, n_classes=N_CLASSES):
        super().__init__()
        self.cls_emb = nn.Embedding(n_classes, d_model)
        self.mom_mlp = nn.Sequential(nn.Linear(7, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.ctx_mlp = nn.Sequential(nn.Linear(n_ctx, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.global_tok = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.ctx_loc = nn.Parameter(torch.tensor([0.0, 0.0, 6500.0, 20.0, 40.0]), requires_grad=False)
        self.ctx_scale = nn.Parameter(torch.tensor([600.0, 600.0, 1500.0, 30.0, 70.0]), requires_grad=False)

    def forward(self, cls, mom, mask, ctx):
        """Returns event embedding z [B,d] and token states h [B,n,d]."""
        tok = self.cls_emb(cls) + self.mom_mlp(momentum_features(mom))
        g = self.global_tok + self.ctx_mlp((ctx - self.ctx_loc) / self.ctx_scale)[:, None]
        x = torch.cat([g, tok], 1)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)
        h = self.norm(self.tf(x, src_key_padding_mask=pad))
        return h[:, 0], h[:, 1:]
