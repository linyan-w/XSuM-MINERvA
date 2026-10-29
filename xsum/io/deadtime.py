"""The dead-time cut variable of the open dataset, `phys_n_dead_discr_pair_upstream_prim_track_proj`, for the surrogate.

It counts front-end discriminator pairs that were dead along the upstream projection of the primary track; it comes
from the data overlay (spill pile-up), not from the interaction, and on the open dataset its distribution is the same
to half a point in the pass rate of the standard cut (<= 1) in every bin of the reconstructed and true quantities
(scripts/curate/deadtime_table.py, reports/deadtime). The surrogate therefore draws it from the empirical distribution,
in bins of the reconstructed vertex z (the only visible dependence), independently of everything else.
"""
from __future__ import annotations

import json

import numpy as np

VARIABLE = "phys_n_dead_discr_pair_upstream_prim_track_proj"
DEFAULT_TABLE = "data/deadtime_marginal.json"


def sample_deadtime(vtx_z, rng, table: str = DEFAULT_TABLE):
    """Draw the variable (int32) for events with reconstructed vertex z [mm] `vtx_z`; outside the table's z range the overall distribution is used."""
    T = json.load(open(table)); z = np.asarray(vtx_z, float); out = np.empty(len(z), np.int32)
    edges = np.asarray(T["by_vtx_z"]["edges_mm"], float); b = np.searchsorted(edges, z, side="right") - 1
    for i in range(-1, len(edges)):
        m = (b == i) if 0 <= i < len(edges) - 1 else ((b < 0) | (b >= len(edges) - 1)) if i == -1 else None
        if m is None or not m.any(): continue
        t = T["by_vtx_z"]["tables"][i] if i >= 0 else T["all"]
        out[m] = rng.choice(np.asarray(t["values"], np.int32), size=int(m.sum()), p=np.asarray(t["probabilities"]) / np.sum(t["probabilities"]))
    return out
