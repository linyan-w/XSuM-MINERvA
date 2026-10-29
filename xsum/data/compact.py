"""Compact multi-file dataset for the surrogate (M2+).

Per file the population (CC nu_mu or all events, z >= z_min) is selected, particles filtered (13.1) and stored ragged:
flat class (int8) and momentum (float32 [n,3]) arrays with offsets. Targets: Tier 0 (3), N prongs, Tier 1 raw
(11: mu pxyz, reco vtx xyz, 5 calorimetric energies), the reference momentum the reco muon is expressed against
(`mu_true`: the true muon, or for events without a mu- the reference of prep.particles.reference_momentum), context
(5), subrun. Cached as .npz.
"""
from __future__ import annotations

import pathlib

import awkward as ak
import numpy as np
import torch
from torch.utils.data import Dataset

from ..prep.features import muon_true
from ..prep.particles import context_from_tuple, in_population, reference_momentum, select_from_tuple
from ..prep.prongs import encode as encode_prongs
from .prongs_tf import PRONG_DIM, ProngTransform, VertexPlaneTable

CACHE_VERSION = "v2"  # v2: + prong records, mc_intType, n_charged_true
PRONG_FIELDS = ["theta", "phi", "has_kin", "pi_P", "has_p", "p_P", "p_score1", "is_primary", "is_exiting"]
N_PRONG_CAP = 8

TIER1_NAMES = ["mu_px", "mu_py", "mu_pz", "rvtx_x", "rvtx_y", "rvtx_z",
               "recoil_E", "recoil_passivecorrected", "hadron_recoil", "recoil_nonvtx100", "nonvtx_iso_blobs_E"]
# Columns the flow generates. recoil_passivecorrected and hadron_recoil are deterministic MasterAnaDev
# calibrations of recoil_E (point masses at fixed ratios: 0.6669 and 1.385) and are derived at decode time.
MODEL_COLS = [0, 1, 2, 3, 4, 5, 6, 9, 10]
DERIVED_COLS = [7, 8]
MODEL_NAMES = ["log_P_ratio", "dtheta_x", "dtheta_y", "dvtx_x", "dvtx_y", "dvtx_z",
               "log_recoil_E", "log_recoil_nonvtx100", "log_nonvtx_iso_blobs_E"]
_TIER1_BRANCHES = {"recoil_E": "MasterAnaDev_recoil_E", "recoil_passivecorrected": "MasterAnaDev_recoil_passivecorrected",
                   "hadron_recoil": "MasterAnaDev_hadron_recoil", "recoil_nonvtx100": "recoil_energy_nonmuon_nonvtx100mm",
                   "nonvtx_iso_blobs_E": "nonvtx_iso_blobs_energy"}
N_PRONG_MAX = 8


def _event_key(arr):
    run = ak.to_numpy(arr["mc_run"]).astype(np.int64); sub = ak.to_numpy(arr["mc_subrun"]).astype(np.int64)
    nth = ak.to_numpy(arr["mc_nthEvtInFile"]).astype(np.int64)
    return (run * 100_000 + sub) * 1_000_000 + nth


def selection_tag(ke_cut_mev: float = 10.0, keep_neutrons: bool = True, population: str = "ccnumu") -> str:
    """Cache-directory suffix for an input selection, e.g. 'ke50', 'ke10n', 'ke10n_all'."""
    return f"ke{ke_cut_mev:g}" + ("n" if keep_neutrons else "") + ("" if population == "ccnumu" else f"_{population}")


def selection_from_config(c: dict) -> dict:
    """load_compact keyword arguments of a model checkpoint's config (keys absent from older checkpoints default to
    the selection those were trained with)."""
    return {"ke_cut_mev": c.get("ke_cut_mev", 50.0), "keep_neutrons": c.get("keep_neutrons", False), "population": c.get("population", "ccnumu")}


SUBSETS = ("ccnumu", "nc", "other")


def subset_mask(d: dict, name: str) -> np.ndarray:
    """Evaluation subsets of an all-events compact dataset: CC nu_mu, NC (all flavours), other CC (nu_mu-bar, nu_e)."""
    cur, inc = d["current"], d["incoming"]; cc = (cur == 1) & (inc == 14)
    return {"ccnumu": cc, "nc": cur == 2, "other": ~cc & (cur != 2)}[name]


def build_compact(stem: str, ke_cut_mev: float = 10.0, z_min: float = 4000.0, keep_neutrons: bool = True, population: str = "ccnumu") -> dict:
    reco = ak.from_parquet(stem + ".reco.parquet")
    truth = ak.from_parquet(stem + ".truth.parquet")
    pop = in_population(truth, population, z_min=z_min)
    truth = truth[pop]
    kr, kt = _event_key(reco), _event_key(truth)
    pos = {k: i for i, k in enumerate(kr)}
    ridx = np.array([pos.get(k, -1) for k in kt]); ok = ridx >= 0
    parts = select_from_tuple(truth, ke_cut_mev, keep_neutrons, population)
    n = ak.to_numpy(ak.num(parts["cls"])).astype(np.int32)
    out = {
        "offsets": np.concatenate([[0], np.cumsum(n)]).astype(np.int64),
        "cls": ak.to_numpy(ak.flatten(parts["cls"])).astype(np.int8),
        "mom": np.stack([ak.to_numpy(ak.flatten(parts[c])) for c in ("px", "py", "pz")], 1).astype(np.float32),
        "ctx": context_from_tuple(truth).astype(np.float32),
        "subrun": ak.to_numpy(truth["mc_subrun"]).astype(np.int32),
        "reco_exists": ok,
    }
    # reference momentum (true muon for CC nu_mu) from the *unfiltered* FS list
    out["mu_true"] = reference_momentum(truth, population, ke_cut_mev)
    if population != "ccnumu":  # evaluation subsets (never model inputs)
        out["current"] = ak.to_numpy(truth["mc_current"]).astype(np.int8); out["incoming"] = ak.to_numpy(truth["mc_incoming"]).astype(np.int8)
    t0 = np.zeros((len(truth), 3), np.float32); t0[:, 0] = ok
    minos = ak.to_numpy(reco["MasterAnaDev_minos_trk_is_ok"]); qp = ak.to_numpy(reco["MasterAnaDev_muon_qp"])
    # charge flag is defined only for MINOS-matched muons (the tuple stores qp = -9999.9 when unmatched, which
    # would otherwise read as "negative"); store it as (matched AND negative) to match the sampler's convention
    t0[ok, 1] = minos[ridx[ok]]; t0[ok, 2] = minos[ridx[ok]] & (qp[ridx[ok]] < 0)
    out["tier0"] = t0
    npr = np.full(len(truth), -1, np.int8); npr[ok] = np.minimum(ak.to_numpy(reco["MasterAnaDev_hadron_number"])[ridx[ok]], N_PRONG_MAX)
    out["nprong"] = npr
    t1 = np.full((len(truth), len(TIER1_NAMES)), np.nan, np.float32)
    for j, c in enumerate("xyz"):
        t1[ok, j] = ak.to_numpy(reco[f"MasterAnaDev_muon_P{c}"])[ridx[ok]]
    t1[ok, 3:6] = ak.to_numpy(reco["MasterAnaDev_vtx"])[ridx[ok], :3]
    for j, nm in enumerate(TIER1_NAMES[6:]):
        t1[ok, 6 + j] = ak.to_numpy(reco[_TIER1_BRANCHES[nm]])[ridx[ok]]
    out["tier1"] = t1
    # holdout labels for M4 (never model inputs)
    out["intType"] = ak.to_numpy(truth["mc_intType"]).astype(np.int8)
    pdgc = truth["mc_FSPartPDG"]
    out["n_charged_true"] = ak.to_numpy(ak.sum((abs(pdgc) == 211) | (pdgc == 2212) | (abs(pdgc) == 321), axis=1)).astype(np.int8)
    # Tier 2: prong records of the reconstructed events, ragged over events (0 prongs for unreconstructed)
    pr = encode_prongs(reco)
    pr = pr[:, :N_PRONG_CAP]
    npr = np.zeros(len(truth), np.int64); npr[ok] = ak.to_numpy(ak.num(pr))[ridx[ok]]
    out["p_offsets"] = np.concatenate([[0], np.cumsum(npr)]).astype(np.int64)
    sel = pr[ridx[ok]]  # prong lists in truth order for reconstructed events
    def flat(name): return ak.to_numpy(ak.flatten(sel[name]))
    has_kin = flat("theta") > -8
    out["prongs"] = np.stack([flat("theta"), flat("phi"), has_kin, np.where(flat("has_pion_fit"), flat("pi_P"), -1.0),
                              flat("has_proton_fit"), flat("p_P"), flat("p_score1"), flat("is_primary_proton"), flat("is_exiting")], 1).astype(np.float32)
    return out


def load_compact(stems, cache_dir=None, ke_cut_mev: float = 10.0, keep_neutrons: bool = True, population: str = "ccnumu", **kw) -> dict:
    """Build (or load cached) compact arrays for each stem and concatenate. The cache directory is keyed by the
    input selection unless given explicitly (the historical 50 MeV / no-neutron selection maps to data/cache_v2)."""
    if cache_dir is None:
        tag = selection_tag(ke_cut_mev, keep_neutrons, population)
        cache_dir = f"data/cache_{CACHE_VERSION}" if tag == "ke50" else f"data/cache_{CACHE_VERSION}_{tag}"
    kw = dict(kw, ke_cut_mev=ke_cut_mev, keep_neutrons=keep_neutrons, population=population)
    parts = []
    for stem in stems:
        cache = pathlib.Path(cache_dir) / (pathlib.Path(stem).name + ".npz")
        if cache.exists():
            d = dict(np.load(cache))
        else:
            d = build_compact(stem, **kw)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, **d)
        parts.append(d)
    out = {}
    for k in parts[0]:
        if k == "tier0":  # older caches stored charge<0 for unmatched muons (sentinel qp); enforce matched & negative
            t = np.concatenate([d[k] for d in parts]); t[:, 2] = t[:, 1] * t[:, 2]; out[k] = t; continue
        if k in ("offsets", "p_offsets"):
            offs, base = [parts[0][k]], parts[0][k][-1]
            for d in parts[1:]:
                offs.append(d[k][1:] + base); base += d[k][-1]
            out[k] = np.concatenate(offs)
        else:
            out[k] = np.concatenate([d[k] for d in parts])
    return out


class Tier1Transform:
    """Raw Tier 1 (11 columns) <-> model space (9 columns, MODEL_COLS).

    Muon: log P ratio and wrapped differences of the projected angles w.r.t. the true muon; vertex: residual;
    energies: log(E+1) with exact zeros dequantised into (-dq, 0). Muon and vertex columns are compressed with
    asinh(u / s0) so broad tails become logarithmic instead of being clipped. Then robust standardisation.
    The two derived columns (recoil_passivecorrected, hadron_recoil) are reproduced at inverse() as
    recoil_E times the median ratio in bins of true vertex z, fitted on the training data.
    """
    ASINH_COLS = [0, 1, 2, 3, 4, 5]
    Z_EDGES = np.arange(4000.0, 9001.0, 250.0)

    def __init__(self, clip: float = 8.0, dequant_width: float = 0.5):
        self.clip, self.dq = clip, dequant_width
        self.s0 = self.loc = self.scale = None
        self.ratio_tables = None  # {col: median ratio per z bin}
        self.planes = None        # VertexPlaneTable (M3): if set, snapped reco z is dequantised in forward()

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _raw_to_unscaled(self, y, mu_true, ctx, rng=None):
        """11-column raw -> 9-column unscaled model variables."""
        out = np.empty((len(y), len(MODEL_COLS)), dtype=np.float64)
        y = y.copy()
        if self.planes is not None and rng is not None:
            # reco z snapped to a plane -> spread uniformly over +-half pitch so the flow sees a smooth target;
            # the discrete vertex class decides at decode time whether to snap back
            snapped = self.planes.classify(y[:, 5], ctx[:, 2]) > 0
            y[:, 5] = np.where(snapped, y[:, 5] + (rng.random(len(y)) - 0.5) * 22.0, y[:, 5])
        Pt = np.linalg.norm(mu_true, axis=1); Pr = np.linalg.norm(y[:, :3], axis=1)
        out[:, 0] = np.log(Pr / Pt)
        out[:, 1] = self._wrap(np.arctan2(y[:, 0], y[:, 2]) - np.arctan2(mu_true[:, 0], mu_true[:, 2]))
        out[:, 2] = self._wrap(np.arctan2(y[:, 1], y[:, 2]) - np.arctan2(mu_true[:, 1], mu_true[:, 2]))
        out[:, 3:6] = y[:, 3:6] - ctx[:, :3]
        E = y[:, [6, 9, 10]]
        e = np.log(np.clip(E, 0, None) + 1.0)
        if rng is not None:  # dequantise exact zeros into (-dq, 0)
            e = np.where(E <= 0, -rng.random(e.shape) * self.dq, e)
        out[:, 6:] = e
        return out

    def _compress(self, u):
        w = u.copy(); k = self.ASINH_COLS
        w[:, k] = np.arcsinh(u[:, k] / self.s0)
        return w

    def _decompress(self, w):
        u = w.copy(); k = self.ASINH_COLS
        u[:, k] = np.sinh(w[:, k]) * self.s0
        return u

    @staticmethod
    def _robust(a):
        loc = np.median(a, 0); return loc, 1.4826 * np.median(np.abs(a - loc), 0) + 1e-6

    def fit(self, y, mu_true, ctx, seed=0):
        u = self._raw_to_unscaled(y, mu_true, ctx, np.random.default_rng(seed))
        good = np.isfinite(u).all(1); u = u[good]
        self.s0 = self._robust(u[:, self.ASINH_COLS])[1]
        self.loc, self.scale = self._robust(self._compress(u))
        # derived-column ratio tables vs true vertex z
        z = ctx[good, 2]; rec = np.clip(y[good, 6], 1e-3, None); bins = np.digitize(z, self.Z_EDGES) - 1
        self.ratio_tables = {}
        for c in DERIVED_COLS:
            ratio = y[good, c] / rec; glob = np.median(ratio[np.isfinite(ratio)])
            tab = np.full(len(self.Z_EDGES) - 1, glob)
            for b in range(len(tab)):
                sel = (bins == b) & np.isfinite(ratio)
                if sel.sum() > 100: tab[b] = np.median(ratio[sel])
            self.ratio_tables[c] = tab
        return self

    def forward(self, y, mu_true, ctx, rng):
        """Returns (x [n,9] float32, valid [n] bool). Non-finite rows (a handful of corrupt tuple entries) are
        zeroed and flagged invalid; callers must exclude them from the flow loss."""
        u = self._raw_to_unscaled(y, mu_true, ctx, rng)
        valid = np.isfinite(u).all(1)
        w = self._compress(np.nan_to_num(u))
        x = np.clip((w - self.loc) / self.scale, -self.clip, self.clip).astype(np.float32)
        x[~valid] = 0.0
        return x, valid

    def derived(self, rec, ctx):
        """recoil_passivecorrected and hadron_recoil from recoil_E and the true vertex z."""
        bins = np.clip(np.digitize(ctx[:, 2], self.Z_EDGES) - 1, 0, len(self.Z_EDGES) - 2)
        return {c: rec * self.ratio_tables[c][bins] for c in DERIVED_COLS}

    def inverse(self, x, mu_true, ctx):
        """9-column model space -> 11-column raw Tier 1 (derived columns filled from the ratio tables)."""
        u = self._decompress(x.astype(np.float64) * self.scale + self.loc)
        y = np.empty((len(u), len(TIER1_NAMES)))
        Pt = np.linalg.norm(mu_true, axis=1); P = Pt * np.exp(u[:, 0])
        thx = self._wrap(np.arctan2(mu_true[:, 0], mu_true[:, 2]) + u[:, 1])
        thy = self._wrap(np.arctan2(mu_true[:, 1], mu_true[:, 2]) + u[:, 2])
        tx, ty = np.tan(thx), np.tan(thy)
        pz = P / np.sqrt(1 + tx**2 + ty**2) * np.where(np.cos(thx) >= 0, 1.0, -1.0)
        y[:, 0], y[:, 1], y[:, 2] = tx * pz, ty * pz, pz
        y[:, 3:6] = u[:, 3:6] + ctx[:, :3]
        E = np.where(u[:, 6:] < 0, 0.0, np.exp(u[:, 6:]) - 1.0)
        y[:, 6], y[:, 9], y[:, 10] = E[:, 0], E[:, 1], E[:, 2]
        for c, v in self.derived(y[:, 6], ctx).items():
            y[:, c] = v
        return y

    ZERO_MODEL_COLS = [7, 8]  # log_recoil_nonvtx100, log_nonvtx_iso_blobs_E: energies with a spike of exact zeros

    def zero_band(self):
        """Model-space interval [lo, hi] per zero-spike column over which exact zeros are dequantised in forward()
        (u in (-dq, 0)); the sampler fills zeros uniformly over it so model-space comparisons see the same band."""
        return np.array([[(-self.dq - self.loc[c]) / self.scale[c], (0.0 - self.loc[c]) / self.scale[c]] for c in self.ZERO_MODEL_COLS], np.float32)

    def zero_fill(self):
        """Model-space value standing for an exact zero of the two zero-spike energies: the centre of the
        dequantisation band, which inverse() maps back to E = 0."""
        u = -0.5 * self.dq
        return np.array([(u - self.loc[c]) / self.scale[c] for c in self.ZERO_MODEL_COLS], np.float32)

    def state(self):
        return {"s0": self.s0, "loc": self.loc, "scale": self.scale, "clip": self.clip, "dq": self.dq,
                "ratio_tables": {str(k): v for k, v in self.ratio_tables.items()},
                "planes": None if self.planes is None else self.planes.state()}

    @classmethod
    def from_state(cls, s):
        t = cls(float(s["clip"]), float(s["dq"]))
        t.s0, t.loc, t.scale = np.asarray(s["s0"]), np.asarray(s["loc"]), np.asarray(s["scale"])
        t.ratio_tables = {int(k): np.asarray(v) for k, v in s["ratio_tables"].items()}
        if s.get("planes") is not None: t.planes = VertexPlaneTable.from_state(s["planes"])
        return t


class CompactDataset(Dataset):
    def __init__(self, d: dict, idx: np.ndarray, transform: Tier1Transform, seed: int = 0,
                 prong_tf: ProngTransform | None = None):
        self.d, self.idx, self.tf = d, idx, transform
        self.rng = np.random.default_rng(seed)
        ok = d["reco_exists"][idx]
        self.x1 = np.zeros((len(idx), len(MODEL_COLS)), np.float32); self.valid = np.zeros(len(idx), bool)
        # zero flags of the two zero-spike energies (raw Tier 1 columns 9, 10); 1 = exactly zero
        self.zflags = np.zeros((len(idx), 2), np.float32)
        if ok.any():
            x, v = transform.forward(d["tier1"][idx[ok]], d["mu_true"][idx[ok]], d["ctx"][idx[ok]], self.rng)
            self.x1[ok] = x; self.valid[ok] = v
            t1 = d["tier1"][idx[ok]]; self.zflags[ok] = np.stack([t1[:, 9] <= 0, t1[:, 10] <= 0], 1).astype(np.float32)
        # Tier 2 (optional): vertex class and prong model-space tensors
        self.tier2 = prong_tf is not None and "prongs" in d
        self.vclass = np.zeros(len(idx), np.int64); self.vphase = np.zeros(len(idx), np.float32)
        if self.tier2:
            if transform.planes is not None:
                # phase of the true z within the plane cycle: (z_true - nearest plane) / half pitch, in [-1, 1];
                # the snapping class depends on it at a frequency the encoder cannot resolve from raw z
                zt = d["ctx"][idx, 2]; self.vphase = ((zt - transform.planes.z[transform.planes.nearest(zt)]) / 11.05).astype(np.float32)
                if ok.any(): self.vclass[ok] = transform.planes.classify(d["tier1"][idx[ok], 5], zt[ok])
            self.ptf = prong_tf
            self.px = prong_tf.forward(d["prongs"], self.rng) if len(d["prongs"]) else np.zeros((0, PRONG_DIM), np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        e = self.idx[i]; a, b = self.d["offsets"][e], self.d["offsets"][e + 1]
        item = [self.d["cls"][a:b], self.d["mom"][a:b], self.d["ctx"][e], self.d["tier0"][e], self.d["nprong"][e],
                self.x1[i], self.d["mu_true"][e], np.float32(self.valid[i]), self.zflags[i]]
        if self.tier2:
            pa, pb = self.d["p_offsets"][e], self.d["p_offsets"][e + 1]
            item += [self.px[pa:pb], self.vclass[i], self.vphase[i]]
        return tuple(item)


def collate(batch):
    n = max(len(b[0]) for b in batch); B = len(batch)
    cls = np.zeros((B, n), np.int64); mom = np.zeros((B, n, 3), np.float32)
    for i, b in enumerate(batch):
        k = len(b[0]); cls[i, :k] = b[0]; mom[i, :k] = b[1]
    stack = lambda j, dt: torch.tensor(np.stack([b[j] for b in batch]), dtype=dt)
    out = {"cls": torch.from_numpy(cls), "mom": torch.from_numpy(mom), "mask": torch.from_numpy(cls > 0),
           "ctx": stack(2, torch.float32), "tier0": stack(3, torch.float32), "nprong": stack(4, torch.long),
           "x1": stack(5, torch.float32), "mu_true": stack(6, torch.float32), "valid": stack(7, torch.float32), "zflags": stack(8, torch.float32)}
    if len(batch[0]) > 9:
        npr = max(1, max(len(b[9]) for b in batch))
        pr = np.zeros((B, npr, PRONG_DIM), np.float32); pm = np.zeros((B, npr), bool)
        for i, b in enumerate(batch):
            k = len(b[9]); pr[i, :k] = b[9]; pm[i, :k] = True
        out["prongs"] = torch.from_numpy(pr); out["pmask"] = torch.from_numpy(pm); out["vclass"] = stack(10, torch.long)
        out["vphase"] = stack(11, torch.float32) if len(batch[0]) > 11 else torch.zeros(B)
    return out


class BucketBatchSampler(torch.utils.data.Sampler):
    """Shuffled, length-bucketed batches: events are shuffled, cut into chunks of `bucket_batches` batches, each
    chunk sorted by token count and split into batches, and the batches shuffled. Padding then follows the
    typical event instead of the longest in a random batch (step time 0.44 of random batching at 10 MeV with
    neutrons); incomplete batches are dropped."""
    def __init__(self, lengths, bs, seed=0, bucket_batches=50):
        self.lengths = np.asarray(lengths); self.bs = bs; self.rng = np.random.default_rng(seed); self.chunk = bs * bucket_batches
        self.n_batches = sum(len(self.lengths[i:i + self.chunk]) // bs for i in range(0, len(self.lengths), self.chunk))

    def __iter__(self):
        perm = self.rng.permutation(len(self.lengths)); batches = []
        for i in range(0, len(perm), self.chunk):
            c = perm[i:i + self.chunk]; c = c[np.argsort(self.lengths[c], kind="stable")]
            batches += [c[j:j + self.bs] for j in range(0, len(c) - self.bs + 1, self.bs)]
        for j in self.rng.permutation(len(batches)): yield batches[j].tolist()

    def __len__(self): return self.n_batches
