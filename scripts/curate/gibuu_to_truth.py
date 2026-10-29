#!/usr/bin/env python
"""Convert a GiBUU campaign (jobs/NNNN/FinalEvents.dat.gz, one nucleus per job) to the open-dataset Truth-tree layout
the surrogate reads (docs/PROJECT.md 13), like scripts/curate/nuwro_to_truth.py for NuWro.

FinalEvents.dat columns: run, event, ID, charge, perweight, position(3), momentum(E, px, py, pz) [GeV], history,
production_ID, enu [GeV]. The campaigns are equal-weight (MC rejection against a ceiling); an event above the
ceiling is kept with its own, larger weight, carried as `gen_relweight` = perweight / (the file's equal weight, its most
common value), which is 1 for all other events. Rows with perweight 0 (the target-nucleon placeholder) are dropped. Final-state particles
are rotated from the beam frame into the detector frame; the vertex is drawn from the open dataset's events of the
same nucleus (--volume, --pool). GiBUU particle IDs map to PDG codes by ID and charge; production_ID maps to GENIE
intType as QE 1 -> 1, resonances 2-31 -> 2, DIS 34 -> 3, 2p2h 35/36 -> 8, non-resonant 1pi/2pi backgrounds 32/33/37
-> 3 (GENIE's DIS model covers them); the raw code is carried as `gibuu_prod`.

Usage: scripts/curate/gibuu_to_truth.py GIBUU_DIR OUT.truth.parquet [--volume full] [--pool cc|nc] [--run-offset N] [--seed 0]"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import awkward as ak, numpy as np, pandas as pd
from xsum.prep.frames import from_beam
from xsum.prep.volumes import vertex_pool, in_volume

M_P = 938.272
ap = argparse.ArgumentParser(); ap.add_argument("gibuu_dir"); ap.add_argument("out"); ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--volume", choices=["fiducial", "full"], default="full"); ap.add_argument("--pool", choices=["cc", "nc"], default="cc"); ap.add_argument("--run-offset", type=int, default=960000)
a = ap.parse_args(); rng = np.random.default_rng(a.seed)
run = json.load(open(pathlib.Path(a.gibuu_dir) / "gibuu_run.json")); beam = int(run["pdg"]); nc = run["current"].lower() == "nc"
lep = beam if nc else beam - int(np.sign(beam))  # CC: charged partner; NC: the neutrino
pool, _ = vertex_pool(a.slim_dir, a.volume, {"cc": 1, "nc": 2}[a.pool]); print(f"vertex pool ({a.volume}, {a.pool}) per Z:", {z: len(p) for z, p in pool.items()}, flush=True)


def intType(prod):
    p = np.asarray(prod); t = np.zeros(len(p), np.int32)
    t[p == 1] = 1; t[(p >= 2) & (p <= 31)] = 2; t[np.isin(p, [32, 33, 34, 37])] = 3; t[np.isin(p, [35, 36])] = 8
    return t


def pdg_of(gid, chg):
    """GiBUU ID and charge -> PDG; negative baryon/neutrino IDs are antiparticles. Unknown IDs -> 0 (reported)."""
    g, c = np.asarray(gid), np.asarray(chg); out = np.zeros(len(g), np.int32); ag = np.abs(g); anti = np.where(g < 0, -1, 1)
    def put(m, v): out[m] = v
    put(ag == 1, anti[ag == 1] * np.where(c[ag == 1] != 0, 2212, 2112))
    d = {2: 2224, 1: 2214, 0: 2114, -1: 1114}; m = (ag >= 2) & (ag <= 31); put(m, anti[m] * np.vectorize(lambda q: d.get(int(q), 2114))(c[m]) if m.any() else 0)  # undecayed resonances (rare)
    put(ag == 101, np.where(c[ag == 101] == 0, 111, np.sign(c[ag == 101]) * 211)); put(ag == 102, 221); put(ag == 103, np.where(c[ag == 103] == 0, 113, np.sign(c[ag == 103]) * 213))
    put(ag == 104, 9000221); put(ag == 105, 223); put(ag == 106, 331); put(ag == 107, 333)
    put(ag == 110, np.where(c[ag == 110] == 0, 311, 321)); put(ag == 111, np.where(c[ag == 111] == 0, -311, -321)); put(ag == 112, np.where(c[ag == 112] == 0, 313, 323)); put(ag == 113, np.where(c[ag == 113] == 0, -313, -323))
    put(ag == 114, np.where(c[ag == 114] == 0, 421, 411)); put(ag == 115, np.where(c[ag == 115] == 0, -421, -411)); put(ag == 118, 431); put(ag == 119, -431)
    put(ag == 32, anti[ag == 32] * 3122); d = {1: 3222, 0: 3212, -1: 3112}; m = ag == 33; put(m, anti[m] * np.vectorize(lambda q: d[int(q)])(c[m]) if m.any() else 0)
    m = ag == 53; put(m, anti[m] * np.where(c[m] == 0, 3322, 3312)); put(ag == 55, anti[ag == 55] * 3334)
    for gi, p in ((901, 11), (902, 13), (903, 15)): m = ag == gi; put(m, np.where(c[m] < 0, p, -p))
    for gi, p in ((911, 12), (912, 14), (913, 16)): m = ag == gi; put(m, anti[m] * p)
    put(ag == 999, 22)
    return out


COLS = ["run", "ev", "id", "chg", "w", "x", "y", "z", "E", "px", "py", "pz", "hist", "prod", "enu"]
parts = []; unknown = {}; nrel = 0
for jd in sorted(glob.glob(f"{a.gibuu_dir}/jobs/*/")):
    job = int(pathlib.Path(jd).name); mf = json.load(open(glob.glob(f"{jd}/manifest_*.json")[0])); Z, A = int(mf["Z"]), int(mf["A"])
    df = pd.read_csv(f"{jd}/FinalEvents.dat.gz", sep=r"\s+", comment="#", header=None, names=COLS, dtype={"run": np.int32, "ev": np.int32, "id": np.int32, "chg": np.int32, "hist": np.int64, "prod": np.int32})
    df = df[df["w"] > 0]; uv, uc = np.unique(df["w"].to_numpy(), return_counts=True); unit = uv[uc.argmax()]  # the equal weight of the file (per-nucleon units differ by nucleus)
    key = df["run"].to_numpy().astype(np.int64) * 10_000_000 + df["ev"].to_numpy(); new = np.r_[True, key[1:] != key[:-1]]; starts = np.flatnonzero(new); counts = np.diff(np.r_[starts, len(df)]); n = len(starts)
    pdg = pdg_of(df["id"].to_numpy(), df["chg"].to_numpy()); bad = pdg == 0
    for gi, cnt in zip(*np.unique(df["id"].to_numpy()[bad], return_counts=True)): unknown[int(gi)] = unknown.get(int(gi), 0) + int(cnt)
    E, px, py, pz = (df[c].to_numpy() * 1e3 for c in ("E", "px", "py", "pz")); px, py, pz = from_beam(px, py, pz)
    Enu = df["enu"].to_numpy()[starts] * 1e3; prod = df["prod"].to_numpy()[starts]; rel = df["w"].to_numpy()[starts] / unit; nrel += int((rel > 1.001).sum())
    seg = np.repeat(np.arange(n), counts); ismu = pdg == lep; Emu = np.where(ismu, E, -1.0); lead = np.full(n, -1.0); np.maximum.at(lead, seg, Emu)
    assert (lead >= 0).all(), f"{jd}: events without the primary lepton"
    li = np.full(n, -1); cand = np.flatnonzero(ismu & (E == lead[seg])); li[seg[cand]] = cand  # row of the leading lepton per event
    mu4 = np.column_stack([px[li], py[li], pz[li], E[li]])
    bx, by, bz = from_beam(np.zeros(n), np.zeros(n), Enu); q = np.column_stack([bx - mu4[:, 0], by - mu4[:, 1], bz - mu4[:, 2], Enu - mu4[:, 3]])
    Q2 = q[:, 0] ** 2 + q[:, 1] ** 2 + q[:, 2] ** 2 - q[:, 3] ** 2; W = np.sqrt(np.clip(M_P ** 2 + 2 * M_P * q[:, 3] - Q2, 0, None))
    pick = pool[Z][rng.integers(0, len(pool[Z]), n)]; vtx = pick[:, :4]; mod = pick[:, 4].astype(np.int32)
    parts.append(ak.Array({
        "mc_run": np.full(n, a.run_offset + job, np.int32), "mc_subrun": np.full(n, job, np.int32), "mc_nthEvtInFile": np.arange(n, dtype=np.int32), "eventID": (a.run_offset + job) * 1e7 + np.arange(n, dtype=np.float64),
        "mc_vtx": vtx, "mc_targetZ": np.full(n, Z, np.int32), "mc_targetA": np.full(n, A, np.int32), "truth_targetID": np.zeros(n, np.int32), "truth_vtx_module": mod, "truth_is_fiducial": np.ones(n, bool) if a.volume == "fiducial" else in_volume(vtx, "fiducial"),
        "mc_nFSPart": counts.astype(np.int32), "mc_FSPartPDG": ak.unflatten(pdg, counts), "mc_FSPartPx": ak.unflatten(px, counts), "mc_FSPartPy": ak.unflatten(py, counts), "mc_FSPartPz": ak.unflatten(pz, counts), "mc_FSPartE": ak.unflatten(E, counts),
        "mc_primFSLepton": mu4, "mc_incoming": np.full(n, beam, np.int32), "mc_incomingE": Enu, "mc_current": np.full(n, 2 if nc else 1, np.int32), "mc_intType": intType(prod), "mc_Q2": Q2, "mc_w": W,
        "gibuu_prod": prod.astype(np.int32), "gibuu_job": np.full(n, job, np.int32), "gen_relweight": rel}))
    print(f"  job {job:04d} {mf['label']}: {n:,} events, {len(df):,} particles, {(rel > 1.001).sum()} above the ceiling", flush=True)
out = ak.concatenate(parts); ak.to_parquet(out, a.out)
print(f"wrote {len(out):,} events -> {a.out}; {nrel:,} above the ceiling (relative weight > 1); unmapped GiBUU IDs (dropped as PDG 0): {unknown}")
