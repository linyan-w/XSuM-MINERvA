"""Masked set flow matching for the prong set: permutation-equivariant velocity net over prong tokens,
conditioned on the event embedding, the Tier 1 vector, and cross-attention to the particle token states."""
from __future__ import annotations

import torch
import torch.nn as nn

from .flow import TimeEmbedding


class ProngVelocity(nn.Module):
    def __init__(self, dim, d_model=128, cond_dim=128 + 9, n_layers=3, n_heads=4, t_dim=64):
        super().__init__()
        self.inp = nn.Linear(dim + t_dim + cond_dim, d_model)
        self.temb = TimeEmbedding(t_dim)
        self.self_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, 0.0, activation="gelu", batch_first=True, norm_first=True) for _ in range(n_layers)])
        self.cross = nn.ModuleList([nn.MultiheadAttention(d_model, n_heads, batch_first=True) for _ in range(n_layers)])
        self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dim))

    def forward(self, x, t, cond, h, hmask, pmask):
        """x [B,N,dim] prong states; cond [B,cond_dim]; h [B,n,d] particle states with mask hmask; pmask [B,N] valid prongs."""
        B, N, _ = x.shape
        c = torch.cat([self.temb(t), cond], -1)[:, None].expand(B, N, -1)
        s = self.inp(torch.cat([x, c], -1))
        pad = ~pmask
        for sl, ca, ln in zip(self.self_layers, self.cross, self.cross_norm):
            s = sl(s, src_key_padding_mask=pad)
            a, _ = ca(ln(s), h, h, key_padding_mask=~hmask)
            s = s + a
        return self.out(s)


class ProngFlow(nn.Module):
    def __init__(self, dim=10, d_model=128, cond_dim=128 + 9, n_layers=3, sigma_min=1e-3):
        super().__init__()
        self.dim, self.sigma_min = dim, sigma_min
        self.v = ProngVelocity(dim, d_model, cond_dim, n_layers)

    def loss(self, x1, cond, h, hmask, pmask):
        """Flow-matching loss averaged over valid prongs. x1 [B,N,dim]."""
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=x1.device)
        tt = t[:, None, None]
        xt = (1 - (1 - self.sigma_min) * tt) * x0 + tt * x1
        target = x1 - (1 - self.sigma_min) * x0
        err = ((self.v(xt, t, cond, h, hmask, pmask) - target) ** 2).mean(-1)
        return (err * pmask).sum() / pmask.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, cond, h, hmask, pmask, n_steps=64, bound=20.0):
        B, N = pmask.shape
        x = torch.randn(B, N, self.dim, device=cond.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=cond.device)
            k1 = self.v(x, t, cond, h, hmask, pmask)
            k2 = self.v(x + 0.5 * dt * k1, t + 0.5 * dt, cond, h, hmask, pmask)
            x = (x + dt * k2).clamp(-bound, bound)
        return torch.nan_to_num(x, nan=0.0, posinf=bound, neginf=-bound)
