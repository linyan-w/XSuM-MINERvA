#!/usr/bin/env python
"""Convert NuWro output trees to the open-dataset Truth-tree layout the surrogate reads (docs/PROJECT.md 13).

Final-state particles are NuWro's post-FSI list (`post`), rotated from the beam frame (NuWro: beam along +z)
into the detector frame. NuWro has no detector geometry, so each event gets a vertex drawn from the open
dataset's fiducial tracker events with the same target nucleus, which preserves the position-nucleus
correlation of the real detector. CC and NC campaigns are both handled (NuWro dyn: even codes CC, odd codes NC);
for NC the primary lepton is the outgoing neutrino and mc_current = 2. Any beam flavour (nu_mu, nu_mu-bar, nu_e, nu_e-bar). Extra branches `nuwro_dyn`, `nuwro_weight`, `nuwro_job` are carried through.

Usage: scripts/curate/nuwro_to_truth.py NUWRO_DIR OUT.truth.parquet [--slim-dir data/slim_1A] [--seed 0]
"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import awkward as ak, numpy as np, uproot
from xsum.prep.frames import from_beam
from xsum.prep.volumes import vertex_pool, in_volume

M_P = 938.272; DYN_TO_INTTYPE = {0: 1, 2: 2, 4: 3, 6: 4, 8: 8, 10: 0, 1: 1, 3: 2, 5: 3, 7: 4, 9: 8}  # NuWro dyn (CC even, NC odd) -> GENIE intType; hyperon production -> 0
ap = argparse.ArgumentParser(); ap.add_argument("nuwro_dir"); ap.add_argument("out"); ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--volume", choices=["fiducial", "full"], default="fiducial", help="volume the vertices are drawn from (fiducial tracker, or the full training volume for the out-of-FV validation)"); ap.add_argument("--pool", choices=["cc", "nc"], default="cc", help="open-dataset nu_mu events the vertices are drawn from: CC, or NC for NC campaigns"); ap.add_argument("--run-offset", type=int, default=900000, help="mc_run = offset + job; distinct per campaign so event keys do not collide")
a = ap.parse_args(); rng = np.random.default_rng(a.seed)

# ---- vertex pool from the open dataset: nu_mu events (CC or NC, --pool) in the chosen volume, keyed by target Z
pool, _ = vertex_pool(a.slim_dir, a.volume, {"cc": 1, "nc": 2}[a.pool]); print(f"vertex pool ({a.volume}, {a.pool}) per Z:", {z: len(p) for z, p in pool.items()}, flush=True)

cols = {"pdg": "e/post/post.pdg", "px": "e/post/post.x", "py": "e/post/post.y", "pz": "e/post/post.z", "E": "e/post/post.t",
        "in_pdg": "e/in/in.pdg", "in_E": "e/in/in.t", "dyn": "e/dyn", "weight": "e/weight", "Z": "e/par/par.nucleus_p", "N": "e/par/par.nucleus_n"}
parts = []
for f in sorted(glob.glob(f"{a.nuwro_dir}/**/nuwro_*.root", recursive=True)):
    job = int(pathlib.Path(f).stem.split("_")[1]); arr = uproot.open(f)["treeout"].arrays(list(cols.values()), library="ak"); arr = {k: arr[v] for k, v in cols.items()}
    n = len(arr["dyn"]); px, py, pz = from_beam(arr["px"], arr["py"], arr["pz"])  # beam frame -> detector frame
    isnu = (abs(arr["in_pdg"]) == 12) | (abs(arr["in_pdg"]) == 14); beam = ak.to_numpy(ak.firsts(arr["in_pdg"][isnu])).astype(np.int32)  # any flavour
    Enu = ak.to_numpy(ak.fill_none(ak.firsts(arr["in_E"][isnu]), np.nan))
    dyn = ak.to_numpy(arr["dyn"]); nc = dyn % 2 == 1
    lep = np.where(nc, beam, beam - np.sign(beam)).astype(np.int32)  # CC: charged partner (14->13, -14->-13, 12->11, -12->-11); NC: the neutrino
    mu = arr["pdg"] == ak.Array(lep); assert ak.all(ak.sum(mu, axis=1) >= 1), f"{f}: events without the primary lepton"
    lead = ak.argmax(ak.where(mu, arr["E"], -1.0), axis=1, keepdims=True)  # primary lepton = most energetic charged partner (CC; rare DIS events carry an extra lepton pair) or neutrino (NC)
    mu4 = np.column_stack([ak.to_numpy(ak.fill_none(ak.flatten(c[lead]), np.nan)) for c in (px, py, pz, arr["E"])])  # plain float columns, no option type
    # Q2 and lepton-defined W: neutrino along the beam axis, detector frame
    bx, by, bz = from_beam(np.zeros(n), np.zeros(n), Enu); q = np.column_stack([bx - mu4[:, 0], by - mu4[:, 1], bz - mu4[:, 2], Enu - mu4[:, 3]])
    Q2 = q[:, 0] ** 2 + q[:, 1] ** 2 + q[:, 2] ** 2 - q[:, 3] ** 2; W = np.sqrt(np.clip(M_P ** 2 + 2 * M_P * q[:, 3] - Q2, 0, None))
    Z = ak.to_numpy(arr["Z"]).astype(np.int32); A = (Z + ak.to_numpy(arr["N"])).astype(np.int32)
    vtx = np.empty((n, 4)); mod = np.empty(n, np.int32)
    for z in np.unique(Z):
        m = Z == z; pick = pool[int(z)][rng.integers(0, len(pool[int(z)]), m.sum())]; vtx[m] = pick[:, :4]; mod[m] = pick[:, 4].astype(np.int32)
    parts.append(ak.Array({
        "mc_run": np.full(n, a.run_offset + job, np.int32), "mc_subrun": np.full(n, job, np.int32), "mc_nthEvtInFile": np.arange(n, dtype=np.int32), "eventID": (a.run_offset + job) * 1e7 + np.arange(n, dtype=np.float64),
        "mc_vtx": vtx, "mc_targetZ": Z, "mc_targetA": A, "truth_targetID": np.zeros(n, np.int32), "truth_vtx_module": mod, "truth_is_fiducial": np.ones(n, bool) if a.volume == "fiducial" else in_volume(vtx, "fiducial"),
        "mc_nFSPart": ak.to_numpy(ak.num(arr["pdg"])).astype(np.int32), "mc_FSPartPDG": ak.values_astype(arr["pdg"], np.int32), "mc_FSPartPx": px, "mc_FSPartPy": py, "mc_FSPartPz": pz, "mc_FSPartE": arr["E"],
        "mc_primFSLepton": mu4, "mc_incoming": beam, "mc_incomingE": Enu, "mc_current": np.where(nc, 2, 1).astype(np.int32),
        "mc_intType": np.array([DYN_TO_INTTYPE.get(int(d), 0) for d in ak.to_numpy(arr["dyn"])], np.int32), "mc_Q2": Q2, "mc_w": W,
        "nuwro_dyn": ak.to_numpy(arr["dyn"]).astype(np.int32), "nuwro_weight": ak.to_numpy(arr["weight"]), "nuwro_job": np.full(n, job, np.int32)}))
    print(f"  {pathlib.Path(f).name}: {n:,} events, Z={Z[0]} A={A[0]}", flush=True)
out = ak.concatenate(parts); ak.to_parquet(out, a.out); print(f"wrote {len(out):,} events -> {a.out}")
