#!/usr/bin/env python
"""Does `phys_n_dead_discr_pair_upstream_prim_track_proj` (the MINERvA dead-time cut variable, standard cut <= 1)
depend on the event?  On the open dataset's reconstructed events (scripts/curate/deadtime_fetch.py), its distribution and
the pass rate of the cut are measured overall and in bins of the reconstructed quantities (vertex z, muon angle and
momentum, recoil energy, prong count, MINOS match), of the truth (neutrino energy, interaction type, final-state
multiplicity, nucleus) and of the beam conditions (spill POT), with the reference fraction and its statistical error.
Writes reports/deadtime/{summary.json, macros.tex, figures/deadtime.png} and the empirical distribution for the
surrogate, data/deadtime_marginal.json.
Usage: scripts/curate/deadtime_table.py [--in data/deadtime] [--slim-dir data/slim_1A] [--out reports/deadtime]"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import awkward as ak, numpy as np
from xsum.eval import plots
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", default="data/deadtime"); ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--out", default="reports/deadtime"); a = ap.parse_args()
out = pathlib.Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True); V = "phys_n_dead_discr_pair_upstream_prim_track_proj"
R = ak.concatenate([ak.from_parquet(f) for f in sorted(glob.glob(f"{a.inp}/*.deadtime.parquet"))])
stems = [pathlib.Path(f).name.replace(".deadtime.parquet", "") for f in sorted(glob.glob(f"{a.inp}/*.deadtime.parquet"))]
T = ak.concatenate([ak.from_parquet(f"{a.slim_dir}/{s}.truth.parquet", columns=["eventID", "mc_incomingE", "mc_intType", "mc_nFSPart", "mc_targetZ", "mc_current", "mc_incoming", "mc_vtx"]) for s in stems])
# join reco -> truth on eventID (the reco tree holds one entry per reconstructed event; several can share a truth event)
te = ak.to_numpy(T["eventID"]); order = np.argsort(te); pos = np.searchsorted(te[order], ak.to_numpy(R["eventID"])); ok = pos < len(te); pos = np.clip(pos, 0, len(te) - 1); ok &= te[order][pos] == ak.to_numpy(R["eventID"])
R = R[ok]; ti = order[pos[ok]]; print(f"{len(R):,} reconstructed events matched to truth ({ok.mean():.4f})", flush=True)
d = ak.to_numpy(R[V]).astype(float); passed = d <= 1
S = {"n": int(len(d)), "pass_le1": float(passed.mean()), "mean": float(d.mean()), "quantiles": {q: float(np.percentile(d, q)) for q in (50, 90, 99)}, "frac_zero": float((d == 0).mean()), "frac_one": float((d == 1).mean())}
S["other_variables"] = {k: {"mean": float(ak.to_numpy(R[k]).mean()), "pass_le1": float((ak.to_numpy(R[k]) <= 1).mean())} for k in R.fields if k.startswith("phys_n_dead")}
print(f"{V}: pass(<=1) {100 * passed.mean():.2f}%, mean {d.mean():.2f}, zero {100 * (d == 0).mean():.1f}%, p50/p90/p99 {S['quantiles']}", flush=True)
vtx = ak.to_numpy(R["MasterAnaDev_vtx"]); mu = ak.to_numpy(R["MasterAnaDev_muon_P"]); th = np.degrees(ak.to_numpy(R["MasterAnaDev_muon_theta"])); rec = ak.to_numpy(R["MasterAnaDev_recoil_E"]); npr = ak.to_numpy(R["n_prongs"]); minos = ak.to_numpy(R["MasterAnaDev_minos_trk_is_ok"]); pot = ak.to_numpy(R["numi_pot"])
Enu = ak.to_numpy(T["mc_incomingE"])[ti]; it = ak.to_numpy(T["mc_intType"])[ti]; nfs = ak.to_numpy(T["mc_nFSPart"])[ti]; Z = ak.to_numpy(T["mc_targetZ"])[ti]; cur = ak.to_numpy(T["mc_current"])[ti]; tz = ak.to_numpy(T["mc_vtx"])[ti][:, 2]
def binned(x, edges, name, labels=None):
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (x >= lo) & (x < hi)
        if m.sum() < 200: continue
        p = passed[m].mean(); rows.append({"bin": labels[i] if labels else f"[{lo:g}, {hi:g})", "n": int(m.sum()), "pass": float(p), "err": float(np.sqrt(p * (1 - p) / m.sum())), "mean": float(d[m].mean())})
    sp = max(abs(r["pass"] - S["pass_le1"]) for r in rows); S.setdefault("binned", {})[name] = {"rows": rows, "max_abs_dev": float(sp), "max_pull": float(max(abs(r["pass"] - S["pass_le1"]) / max(r["err"], 1e-9) for r in rows))}
    print(f"  {name:22s} pass by bin: " + " ".join(f"{100 * r['pass']:.1f}" for r in rows) + f"  (max |dev| {100 * sp:.2f} points)", flush=True); return rows
B = {}
B["reco vertex z [mm]"] = binned(vtx[:, 2], np.array([4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000]), "reco_vtx_z")
B["true vertex z [mm]"] = binned(tz, np.array([4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000]), "true_vtx_z")
B["muon angle [deg]"] = binned(th, np.array([0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 60, 90]), "muon_theta")
B["muon P [GeV]"] = binned(mu / 1e3, np.array([0, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 40]), "muon_P")
B["recoil E [GeV]"] = binned(rec / 1e3, np.array([0, 0.05, 0.1, 0.2, 0.4, 0.7, 1, 2, 4, 8, 20]), "recoil_E")
B["prongs"] = binned(npr, np.arange(1, 8), "n_prongs", [str(i) for i in range(1, 7)])
B["MINOS match"] = binned(minos, np.array([0, 1, 2]), "minos", ["no", "yes"])
B["true E_nu [GeV]"] = binned(Enu / 1e3, np.array([0, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 60]), "Enu")
B["interaction type"] = binned(it, np.array([1, 2, 3, 4, 8, 9]), "intType", ["QE", "RES", "DIS", "COH/other", "2p2h"])
B["final-state particles"] = binned(nfs, np.array([1, 2, 3, 4, 6, 8, 12, 20, 60]), "nFSPart")
B["nucleus Z"] = binned(Z, np.array([1, 2, 6, 7, 8, 9, 26, 27, 82, 83]), "targetZ", ["H", "2-5", "C", "N", "O", "9-25", "Fe", "27-81", "Pb"])
B["current"] = binned(cur, np.array([1, 2, 3]), "current", ["CC", "NC"])
B["spill POT [1e12]"] = binned(pot, np.array([0, 18, 20, 22, 23, 24, 25, 26, 27, 30]), "numi_pot")
# correlation of the variable itself with the reconstructed quantities
fin = lambda v: np.isfinite(v) & (v > -999)
S["corr"] = {k: float(np.corrcoef(d[fin(v)], v[fin(v)])[0, 1]) for k, v in (("vtx_z", vtx[:, 2]), ("muon_theta", th), ("muon_P", mu), ("recoil_E", rec), ("n_prongs", npr), ("Enu", Enu), ("nFSPart", nfs), ("numi_pot", pot))}
print("correlations:", {k: round(v, 3) for k, v in S["corr"].items()}, flush=True)
# empirical distribution for the surrogate: value counts (capped at 200) overall and in bins of the reconstructed vertex z (the only
# dependence seen, ~0.5 points in the pass rate along the detector); the surrogate draws from the table of its reconstructed z
ZE = [4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000]; tab = {}
def vc(x): vals, cnts = np.unique(np.minimum(x, 200).astype(int), return_counts=True); return {"values": vals.tolist(), "probabilities": (cnts / cnts.sum()).tolist(), "n": int(len(x))}
tab["all"] = vc(d); tab["by_vtx_z"] = {"edges_mm": ZE, "tables": [vc(d[(vtx[:, 2] >= lo) & (vtx[:, 2] < hi)]) for lo, hi in zip(ZE[:-1], ZE[1:])]}
json.dump({"variable": V, "n": int(len(d)), "pass_le1": float(passed.mean()), "note": "empirical distribution over reconstructed open-dataset events (overlay dead time, Playlist 1A), overall and by reconstructed vertex z; events outside the z range use 'all'", **tab}, open(pathlib.Path(a.inp) / "deadtime_marginal.json", "w"), indent=1)
# figure: distribution and the pass rate in bins of the quantities with the largest spread
fig, axs = plots.plt.subplots(2, 4, figsize=(15, 6.4)); axs = axs.ravel()
ax = axs[0]; b = np.arange(0, 41) - 0.5; ax.hist(np.minimum(d, 40), b, color=plots.PALETTE["real"]); ax.set_yscale("log"); ax.set_xlabel(V.replace("phys_n_dead_discr_pair_", "n dead discr. pairs, ")); ax.set_ylabel("events"); ax.axvline(1.5, color="k", ls=":", lw=1); ax.set_title(f"pass(<=1) {100 * passed.mean():.1f}%", fontsize=9, loc="left")
for ax, name in zip(axs[1:], ["reco vertex z [mm]", "muon angle [deg]", "muon P [GeV]", "recoil E [GeV]", "prongs", "true E_nu [GeV]", "spill POT [1e12]"]):
    rows = B[name]; x = np.arange(len(rows)); ax.errorbar(x, [100 * r["pass"] for r in rows], [100 * r["err"] for r in rows], fmt="o", ms=3.5, color=plots.PALETTE["real"]); ax.axhline(100 * passed.mean(), color="0.5", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([r["bin"] for r in rows], rotation=60, fontsize=6.5); ax.set_xlabel(name); ax.set_ylabel("pass rate [%]")
plots.save(fig, out / "figures" / "deadtime.png")
json.dump(S, open(out / "summary.json", "w"), indent=1)
mac = {"DeadPass": f"{100 * S['pass_le1']:.1f}\\%", "DeadMean": f"{S['mean']:.1f}", "DeadZero": f"{100 * S['frac_zero']:.0f}\\%", "DeadN": f"{len(d):,}".replace(",", "{,}"),
       "DeadMaxDevVtxZ": f"{100 * S['binned']['reco_vtx_z']['max_abs_dev']:.1f}", "DeadMaxDevTheta": f"{100 * S['binned']['muon_theta']['max_abs_dev']:.1f}", "DeadMaxDevRecoil": f"{100 * S['binned']['recoil_E']['max_abs_dev']:.1f}",
       "DeadMaxDevAll": f"{100 * max(v['max_abs_dev'] for v in S['binned'].values()):.1f}", "DeadMaxDevPOT": f"{100 * S['binned']['numi_pot']['max_abs_dev']:.1f}"}
open(out / "macros.tex", "w").write("".join(f"\\newcommand{{\\n{k}}}{{{v}}}\n" for k, v in mac.items())); print("written", out)
