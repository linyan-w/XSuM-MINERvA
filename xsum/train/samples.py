"""One sampling pass of the surrogate over an event set, kept as plain arrays and optionally cached, so that the
Tier 0/1 evaluation (m2.evaluate), the Tier 2 evaluation (m3.evaluate_tier2) and any subset evaluation (2p2h, control,
population subsets) reuse one draw instead of each re-sampling the same events. A cache holds the event indices with
the samples; a subset evaluation picks its rows by index."""
from __future__ import annotations

import pathlib

import numpy as np
import torch

from xsum.models.surrogate import PRONG_DIM_CAP
from xsum.train.event_stage import to_dev

FIELDS = ("exist", "minos", "charge", "nprong", "x1", "p0", "pn", "vclass", "pv", "prongs", "pmask")


def sample_split(model, ld, n_steps=64, device="cuda", seed=0, prongs=True):
    """Sample every batch of `ld` once (in loader order); returns {field: array over events}."""
    torch.manual_seed(seed); acc = {k: [] for k in FIELDS}
    with torch.no_grad():
        for b in ld:
            s = model.sample(to_dev(b, device), n_steps, prongs=prongs)
            for k in ("exist", "minos", "charge", "nprong", "x1", "p0", "pn"): acc[k].append(s[k].cpu())
            if model.tier2:
                acc["vclass"].append(s["vclass"].cpu()); acc["pv"].append(s["pv"].cpu())
                pr, pm = s["prongs"].cpu(), s["pmask"].cpu()
                acc["prongs"].append(torch.nn.functional.pad(pr, (0, 0, 0, PRONG_DIM_CAP - pr.shape[1])))  # fixed pad: the cache is one array
                acc["pmask"].append(torch.nn.functional.pad(pm, (0, PRONG_DIM_CAP - pm.shape[1])))
    out = {k: torch.cat(v).numpy() for k, v in acc.items() if v}
    for k in ("exist", "minos", "charge", "pmask"):
        if k in out: out[k] = out[k].astype(bool)
    return out


def get_samples(model, ld, idx, n_steps=64, cache=None, from_cache=None, device="cuda", seed=0, prongs=True):
    """Samples for the events `idx` (the loader's order): read from `from_cache` (rows picked by event index; the cache
    must cover every event of `idx`), else sampled now and written to `cache` if given."""
    if from_cache:
        z = np.load(from_cache); ci = z["idx"]; order = np.argsort(ci); pos = np.searchsorted(ci[order], idx)
        assert pos.max() < len(ci) and np.array_equal(ci[order][pos], idx), f"{from_cache} does not cover the requested events"
        rows = order[pos]; S = {k: z[k][rows] for k in FIELDS if k in z.files}
        print(f"samples of {len(idx):,} events read from {from_cache} ({len(ci):,} cached)", flush=True); return S
    S = sample_split(model, ld, n_steps, device, seed, prongs)
    if cache:
        pathlib.Path(cache).parent.mkdir(parents=True, exist_ok=True); np.savez(cache, idx=np.asarray(idx), **S); print(f"samples of {len(idx):,} events written to {cache}", flush=True)
    return S
