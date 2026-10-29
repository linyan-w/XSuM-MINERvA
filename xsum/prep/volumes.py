"""Detector volumes and open-dataset vertex pools for placing generator events that come without a vertex.

`fiducial`: the tracker fiducial volume (hexagon of apothem 850 mm, 5990 < z < 8340 mm).
`full`: the surrogate's training volume (true vertex 4000 <= z <= 8700 mm, no transverse cut), which includes the
nuclear targets, the outer detector and the calorimeters, i.e. every region whose events can be reconstructed
inside the fiducial volume."""
from __future__ import annotations

import glob

import awkward as ak
import numpy as np

APOTHEM, Z_FID = 850.0, (5990.0, 8340.0)
Z_FULL = (4000.0, 8700.0)


def in_hexagon(x, y, apothem=APOTHEM):
    return (np.abs(x) <= apothem) & (np.abs(y) <= 2 * apothem / np.sqrt(3) - np.abs(x) / np.sqrt(3))


def in_volume(v, volume):
    """v [n, >=3] vertices in mm; volume 'fiducial' or 'full'."""
    if volume == "fiducial": return in_hexagon(v[:, 0], v[:, 1]) & (v[:, 2] > Z_FID[0]) & (v[:, 2] < Z_FID[1])
    if volume == "full": return (v[:, 2] >= Z_FULL[0]) & (v[:, 2] <= Z_FULL[1])
    raise ValueError(volume)


def vertex_pool(slim_dir="data/slim_1A", volume="fiducial", current: int = 1, incoming=14):
    """{Z: array [n, 5] (x, y, z, t, module)} of open-dataset nu_mu true vertices in the volume, CC (current=1) or NC
    (current=2), of one beam flavour (`incoming` pdg, None for all), and the per-Z event fraction in that volume (the open dataset's nucleus composition weighted by flux x
    cross section, which differs between CC and NC)."""
    pool = {}
    for f in sorted(glob.glob(f"{slim_dir}/*.truth.parquet")):
        t = ak.from_parquet(f, columns=["mc_vtx", "mc_targetZ", "truth_vtx_module", "mc_current", "mc_incoming"])
        v = ak.to_numpy(t["mc_vtx"]); Z = ak.to_numpy(t["mc_targetZ"]); mod = ak.to_numpy(t["truth_vtx_module"])
        sel = (ak.to_numpy(t["mc_current"]) == current) & ((ak.to_numpy(t["mc_incoming"]) == incoming) if incoming is not None else True) & in_volume(v, volume)
        for z in np.unique(Z[sel]): m = sel & (Z == z); pool.setdefault(int(z), []).append(np.column_stack([v[m], mod[m]]))
    pool = {z: np.concatenate(p) for z, p in pool.items()}; n = sum(len(p) for p in pool.values())
    return pool, {z: len(p) / n for z, p in pool.items()}
