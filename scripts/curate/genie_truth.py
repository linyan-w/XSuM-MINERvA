#!/usr/bin/env python
"""GENIE reference sample for generator comparisons: open-dataset Truth events (CC nu_mu) with the true vertex in the
tracker fiducial volume (hexagon of apothem 850 mm, 5990 < z < 8340 mm), subsampled to N events, with the open
dataset's own reconstruction of each event attached as `ref_*` columns (reco_exists, muon P and beam-frame angle,
recoil_E, recoil_nonvtx100, nonvtx_iso_blobs_E, prong count). --population ncnumu gives the NC nu_mu events instead,
for the NC generator comparison, and numu the CC and NC nu_mu events together (inclusive comparison), and all every event of every flavour and current. Usage: scripts/curate/genie_truth.py OUT.parquet [--n 2000000] [--population ccnumu|ncnumu|numu|all]"""
import argparse, glob, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import awkward as ak, numpy as np
from xsum.prep.particles import in_training_population, in_population
from xsum.data.compact import load_compact, TIER1_NAMES
from xsum.prep.frames import theta_phi_beam
from xsum.prep.volumes import in_volume
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=2000000); ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--volume", choices=["fiducial", "full"], default="fiducial"); ap.add_argument("--population", choices=["ccnumu", "ncnumu", "numu", "all"], default="ccnumu"); a = ap.parse_args()
stems = sorted(p[:-len(".truth.parquet")] for p in glob.glob(f"{a.slim_dir}/*.truth.parquet")); d = load_compact(stems, population="ccnumu" if a.population == "ccnumu" else "all")  # compact rows follow the truth files (population-selected) in order
def fiducial(v):
    return in_volume(v, a.volume)
def selected(t):  # population and volume masks of one truth file (rows of the compact cache follow the population mask)
    pop = in_training_population(t) if a.population == "ccnumu" else in_population(t, "all"); tp = t[pop]
    if a.population in ("ncnumu", "numu"):
        cur = ak.to_numpy(tp["mc_current"]); keep = ((cur == 2) | ((cur == 1) & (a.population == "numu"))) & (ak.to_numpy(tp["mc_incoming"]) == 14)
    else:
        keep = np.ones(len(tp), bool)
    return pop, keep & fiducial(ak.to_numpy(tp["mc_vtx"]))
# pass 1 (light columns): number of selected events per file, then a global random subsample of N; pass 2 keeps only those
sel_n = [int(selected(ak.from_parquet(s + ".truth.parquet", columns=["mc_vtx", "mc_current", "mc_incoming"]))[1].sum()) for s in stems]
tot = sum(sel_n); pick = np.zeros(tot, bool); pick[np.random.default_rng(a.seed).permutation(tot)[:a.n]] = True; pick_by_file = np.split(pick, np.cumsum(sel_n)[:-1])
parts, off = [], 0
for i_s, s in enumerate(stems):
    t = ak.from_parquet(s + ".truth.parquet"); pop = in_training_population(t) if a.population == "ccnumu" else in_population(t, "all"); t = t[pop]; n = len(t)
    if a.population != "ccnumu":
        cur = ak.to_numpy(t["mc_current"]); keep = np.ones(len(cur), bool) if a.population == "all" else ((cur == 2) | ((cur == 1) & (a.population == "numu"))) & (ak.to_numpy(t["mc_incoming"]) == 14); t = t[keep]; rows_all = np.arange(off, off + n)[keep]; off += n; n = len(t)
    else:
        rows_all = None
    v = ak.to_numpy(t["mc_vtx"]); fid = fiducial(v)
    if rows_all is None: rows = np.arange(off, off + n); off += n
    else: rows = rows_all
    assert np.allclose(ak.to_numpy(t["mc_vtx"])[:, 2], d["ctx"][rows, 2]), "compact/truth row mismatch"
    t = t[fid]; r = rows[fid]; pk = pick_by_file[i_s]; assert len(pk) == len(t); t = t[pk]; r = r[pk]; t1 = d["tier1"][r]; ex = d["reco_exists"][r]
    th, _ = theta_phi_beam(t1[:, 0], t1[:, 1], t1[:, 2])
    extra = {"ref_reco_exists": ex, "ref_muon_P": np.where(ex, np.linalg.norm(t1[:, :3], axis=1), np.nan), "ref_muon_theta": np.where(ex, th, np.nan),
             "ref_recoil_E": np.where(ex, t1[:, 6], np.nan), "ref_recoil_nonvtx100": np.where(ex, t1[:, 9], np.nan), "ref_nonvtx_iso_blobs_E": np.where(ex, t1[:, 10], np.nan),
             "ref_n_prongs": np.where(ex, d["nprong"][r], -1).astype(np.int32), "ref_vtx": np.where(ex[:, None], t1[:, 3:6], np.nan)}
    parts.append(ak.Array({**{k: t[k] for k in t.fields}, **extra})); print(f"  {pathlib.Path(s).name}: {fid.sum():,} {a.volume} events of {n:,}, {pk.sum():,} kept", flush=True)
allt = ak.concatenate(parts); print(f"{tot:,} {a.volume} {a.population} events, {len(allt):,} kept", flush=True)
ak.to_parquet(allt, a.out); print(f"wrote {len(allt):,} -> {a.out}")
