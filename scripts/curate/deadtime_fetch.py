#!/usr/bin/env python
"""Fetch the dead-time variables of the open dataset (not in the slim files) with the reconstructed quantities they
could depend on, streamed from the xrootd door, to study whether `phys_n_dead_discr_pair_upstream_prim_track_proj`
can be generated independently of the event (docs/PROJECT.md 13.2) and to store its distribution for the surrogate.
Usage: scripts/curate/deadtime_fetch.py OUT_DIR [--files 4] [--playlist configs/MediumEnergy_FHC_StandardMC_Playlist1M.txt]"""
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import awkward as ak, numpy as np, uproot
from xsum.io.branches import ID_BRANCHES
ap = argparse.ArgumentParser(); ap.add_argument("out_dir"); ap.add_argument("--files", type=int, default=4); ap.add_argument("--playlist", default="configs/MediumEnergy_FHC_StandardMC_Playlist1M.txt"); a = ap.parse_args()
out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
RECO = ["MasterAnaDev_minos_trk_is_ok", "MasterAnaDev_muon_P", "MasterAnaDev_muon_theta", "MasterAnaDev_vtx", "MasterAnaDev_recoil_E", "n_prongs", "multiplicity",
        "numi_pot", "ev_run", "ev_subrun", "ev_gate", "batch_structure", "n_slices", "slice_n_hits"]
urls = [l.strip() for l in open(a.playlist) if l.strip()][: a.files]
for url in urls:
    stem = pathlib.Path(url).stem; t0 = time.time()
    if (out / f"{stem}.deadtime.parquet").exists(): print("skip", stem); continue
    f = uproot.open(url); tree = f["MasterAnaDev"]
    dead = [b for b in tree.keys() if "dead" in b.lower()]; have = [b for b in RECO if b in tree.keys()]
    print(stem, "dead-time branches:", dead, "| missing:", [b for b in RECO if b not in tree.keys()], flush=True)
    arr = tree.arrays(ID_BRANCHES + dead + have, library="ak")
    ak.to_parquet(arr, out / f"{stem}.deadtime.parquet"); print(f"  {len(arr):,} entries, {time.time() - t0:.0f} s", flush=True)
