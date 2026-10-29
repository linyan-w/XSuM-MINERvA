"""Tier 2 prong records <-> model space (10 columns per prong), and the vertex-plane table.

Prong model columns:
  0 has_kin (dequantised flag)   1-3 unit direction (beam frame)   4 log pi_P   5 has_p (flag)
  6 log p_P   7 logit proton score   8 is_exiting (flag)   9 is_primary (flag)
Fields that do not exist for a prong (no kinematics / no proton fit) are filled with standard-normal noise so
that the flow learns "flag low => the rest is noise"; the decoder ignores them.
"""
from __future__ import annotations

import numpy as np

PRONG_DIM = 10
FLAG_COLS = [0, 5, 8, 9]
CONT_COLS = [4, 6, 7]


def _dequant(flag, rng):
    """{0,1} -> uniform in [-1,-0.2) or (0.2,1]: a gap around zero keeps decoding unambiguous."""
    u = 0.2 + 0.8 * rng.random(flag.shape)
    return np.where(flag > 0.5, u, -u)


class ProngTransform:
    def __init__(self):
        self.loc = np.zeros(PRONG_DIM); self.scale = np.ones(PRONG_DIM)

    def _unscaled(self, P, rng):
        theta, phi, has_kin, piP, has_p, pP, sc, prim, ex = (P[:, i] for i in range(9))
        n = len(P); out = np.empty((n, PRONG_DIM))
        kin = has_kin > 0.5; pf = has_p > 0.5
        out[:, 0] = _dequant(has_kin, rng)
        out[:, 1] = np.where(kin, np.sin(theta) * np.cos(phi), rng.standard_normal(n))
        out[:, 2] = np.where(kin, np.sin(theta) * np.sin(phi), rng.standard_normal(n))
        out[:, 3] = np.where(kin, np.cos(theta), rng.standard_normal(n))
        out[:, 4] = np.where(kin & (piP > 0), np.log(np.clip(piP, 1, None)), np.nan)
        out[:, 5] = _dequant(has_p, rng)
        out[:, 6] = np.where(pf, np.log(np.clip(pP, 1, None)), np.nan)
        s = np.clip(sc, 1e-4, 1 - 1e-4)
        s = np.where(sc >= 1 - 1e-6, 1 - 1e-4 * (1 + 9 * rng.random(n)), s)   # dequantise the score==1 edge
        out[:, 7] = np.where(pf, np.log(s / (1 - s)), np.nan)
        out[:, 8] = _dequant(ex, rng); out[:, 9] = _dequant(prim, rng)
        return out

    def fit(self, P, seed=0):
        u = self._unscaled(P, np.random.default_rng(seed))
        for c in CONT_COLS:
            v = u[:, c][np.isfinite(u[:, c])]
            self.loc[c] = np.median(v); self.scale[c] = 1.4826 * np.median(np.abs(v - self.loc[c])) + 1e-6
        return self

    def forward(self, P, rng):
        u = self._unscaled(P, rng)
        x = (u - self.loc) / self.scale
        x = np.where(np.isfinite(x), x, rng.standard_normal(x.shape))  # missing fields -> noise
        return np.clip(x, -8, 8).astype(np.float32)

    def inverse(self, x):
        """Model space -> prong record columns [theta, phi, has_kin, pi_P, has_p, p_P, p_score1, is_primary, is_exiting]."""
        u = x.astype(np.float64) * self.scale + self.loc
        kin = u[:, 0] > 0; pf = kin & (u[:, 5] > 0)          # proton fit requires kinematics
        d = u[:, 1:4]; d = d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-9, None)
        theta = np.where(kin, np.arccos(np.clip(d[:, 2], -1, 1)), -9.0)
        phi = np.where(kin, np.arctan2(d[:, 1], d[:, 0]), -9.0)
        piP = np.where(kin, np.exp(np.clip(u[:, 4], 0, 12)), -1.0)
        pP = np.where(pf, np.exp(np.clip(u[:, 6], 0, 12)), -1.0)
        sc = np.where(pf, 1 / (1 + np.exp(-np.clip(u[:, 7], -30, 30))), -1.0)
        ex = (u[:, 8] > 0).astype(np.int32)
        prim_score = np.where(pf, u[:, 9], -np.inf)
        return np.stack([theta, phi, kin, piP, pf, pP, sc, prim_score, ex], 1)

    def state(self):
        return {"loc": self.loc, "scale": self.scale}

    @classmethod
    def from_state(cls, s):
        t = cls(); t.loc, t.scale = np.asarray(s["loc"]), np.asarray(s["scale"]); return t


class VertexPlaneTable:
    """Plane positions at which MasterAnaDev snaps the reconstructed vertex z (69% of events).

    Fitted from the reconstructed z values that occur far more often than a continuous density allows. The
    vertex class is: 0 = not snapped; 1..7 = snapped to plane nearest(true z) + delta, delta in [-3, 3];
    8 = snapped to a plane further away (14% of events), decoded as the plane nearest to the flow's continuous z.
    """
    N_CLASSES = 9
    FAR = 8
    TOL = 0.05  # mm

    def __init__(self):
        self.z = None

    def fit(self, reco_z, min_count=200):
        v, c = np.unique(np.round(reco_z, 1), return_counts=True)
        z = np.sort(v[c >= min_count])
        merged = [z[0]] if len(z) else []
        for zz in z[1:]:
            if zz - merged[-1] > 1.0: merged.append(zz)
        self.z = np.array(merged); return self

    def nearest(self, z):
        i = np.clip(np.searchsorted(self.z, z), 1, len(self.z) - 1)
        return np.where(np.abs(self.z[i - 1] - z) <= np.abs(self.z[i] - z), i - 1, i)

    def classify(self, reco_z, true_z):
        i_r = self.nearest(reco_z); snapped = np.abs(self.z[i_r] - reco_z) <= self.TOL
        delta = i_r - self.nearest(true_z)
        cls = np.where(snapped, np.where(np.abs(delta) <= 3, delta + 4, self.FAR), 0)
        return cls.astype(np.int64)

    def z_for_class(self, cls, true_z, z_cont=None):
        """Snapped z for classes 1..7 (nearest(true z) + delta) and 8 (plane nearest to z_cont); NaN for class 0."""
        i = np.clip(self.nearest(true_z) + (cls - 4), 0, len(self.z) - 1)
        out = np.where((cls > 0) & (cls < self.FAR), self.z[i], np.nan)
        if z_cont is not None:
            out = np.where(cls == self.FAR, self.z[self.nearest(z_cont)], out)
        return out

    def state(self):
        return {"z": self.z}

    @classmethod
    def from_state(cls, s):
        t = cls(); t.z = np.asarray(s["z"]); return t
