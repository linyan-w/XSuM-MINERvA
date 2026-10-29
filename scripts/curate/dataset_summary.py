#!/usr/bin/env python
"""Build the training dataset from slimmed Parquet files and print population statistics.
Usage: scripts/curate/dataset_summary.py data/slim/<stem>"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import awkward as ak
import numpy as np
from xsum.data.dataset import EventDataset, split_by_subrun
from xsum.constants import CLASS_NAMES

if __name__ == "__main__":
    stem = sys.argv[1]
    t0 = time.time()
    reco = ak.from_parquet(stem + ".reco.parquet")
    truth = ak.from_parquet(stem + ".truth.parquet")
    ds = EventDataset(reco, truth, n_max=64)
    print(f"loaded {len(reco)} reco / {len(truth)} truth entries in {time.time()-t0:.0f} s")
    n = len(ds)
    print(f"training population (CC numu, z>=4000): {n} events; reco_exists {ds.reco_exists.mean():.3f}")
    npart = ds.mask.sum(1)
    print(f"particles per event after cuts: mean {npart.mean():.2f}, p50/90/99/max {np.percentile(npart,[50,90,99]).astype(int).tolist()}/{npart.max()}")
    cnt = np.bincount(ds.cls[ds.mask], minlength=len(CLASS_NAMES))
    print("class counts:", {CLASS_NAMES[i]: int(c) for i, c in enumerate(cnt) if c})
    sub = ds.subrun
    sp = split_by_subrun(sub)
    print(f"subruns: {len(np.unique(sub))}; split sizes train/val/test: {np.bincount(sp, minlength=3).tolist()}")
    ok = ds.reco_exists
    print(f"tier0 among reconstructed: minos_ok {ds.tier0[ok,1].mean():.3f}, charge_neg {ds.tier0[ok,2].mean():.3f}")
    print(f"tier1 (reconstructed) medians: {dict(zip(['mu_px','mu_py','mu_pz','recoil_E','recoil_pc','had_recoil','nonvtx100'], np.nanmedian(ds.tier1[ok],0).round(1).tolist()))}")
    print(f"total {time.time()-t0:.0f} s")
