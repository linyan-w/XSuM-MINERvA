#!/usr/bin/env python
"""Does the spread between ensemble members track the surrogate's residual, and does it grow where the training data are
sparse?  The members are sampled on the same test events with common noise; in bins of the reconstructed quantities (the
calibration's marginals) and of the true muon pT and pL (the MINERvA 101 bins, for the reconstruction and selection
probabilities of signal-like events) the between-member spread of the bin fraction is compared with the ensemble mean's
residual against the open dataset and with the statistical and sampling errors; pulls with and without the ensemble term.
Per event, the between-member std of the muon momentum, the recoil and the vertex z is binned in true pT and angle.
Usage: scripts/evaluate/ensemble_spread.py --members weights/full weights/full_s1 ... [--subset ccnumu] [--n 300000] [--out reports/ensemble]"""
import argparse, glob, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np, torch
from torch.utils.data import DataLoader
from xsum.data.compact import load_compact, CompactDataset, collate, selection_from_config, subset_mask
from xsum.data.dataset import split_by_subrun
from xsum.train.prong_stage import load_model
from xsum.train.event_stage import to_dev
from xsum.prep.frames import theta_phi_beam
from xsum.prep.volumes import in_volume
from xsum.eval import plots
ap = argparse.ArgumentParser(); ap.add_argument("--members", nargs="+", required=True); ap.add_argument("--subset", default="ccnumu"); ap.add_argument("--n", type=int, default=300000); ap.add_argument("--out", default="reports/ensemble"); ap.add_argument("--steps", type=int, default=64); a = ap.parse_args()
out = pathlib.Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True)
cfg = torch.load(f"{a.members[0]}/model.pt", map_location="cpu", weights_only=False)["config"]
d = load_compact(sorted(p[:-len(".truth.parquet")] for p in glob.glob("data/slim_1A/*.truth.parquet")), **selection_from_config(cfg)); split = split_by_subrun(d["subrun"], seed=0)
te = np.where(split == 2)[0]
if a.subset: te = te[subset_mask(d, a.subset)[te]]
idx = np.random.default_rng(0).permutation(te)[:a.n]; K = len(a.members)
# ---- one common-noise pass per member, and a second-noise pass of the nominal member (per-event resolution)
def one_pass(md, seed):
    model, tf, ptf = load_model(f"{md}/model.pt"); ds = CompactDataset(d, idx, tf, 0, prong_tf=ptf); torch.manual_seed(seed); X1, ex, mi, ch, VC = [], [], [], [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=4096, collate_fn=collate, num_workers=4):
            s = model.sample(to_dev(b, "cuda"), a.steps, prongs=False); X1.append(s["x1"].cpu()); ex.append(s["exist"].cpu()); mi.append(s["minos"].cpu()); ch.append(s["charge"].cpu()); VC.append(s["vclass"].cpu())
    x1 = torch.cat(X1).numpy(); vc = torch.cat(VC).numpy(); y = tf.inverse(x1, d["mu_true"][idx], d["ctx"][idx]); zs = tf.planes.z_for_class(vc, d["ctx"][idx, 2], y[:, 5]); y[:, 5] = np.where(vc > 0, zs, y[:, 5])
    ex = torch.cat(ex).numpy(); print(f"{md} (seed {seed}): sampled {len(idx):,} events, reco {ex.mean():.4f}", flush=True)
    return ex, torch.cat(mi).numpy(), torch.cat(ch).numpy(), y
EX, MI, CH, Y = [], [], [], []
for md in a.members:
    ex_, mi_, ch_, y_ = one_pass(md, 0); EX.append(ex_); MI.append(mi_); CH.append(ch_); Y.append(y_)
EX2, _, _, Y2 = one_pass(a.members[0], 1)  # the nominal again with other noise
EX, MI, CH, Y = np.array(EX), np.array(MI), np.array(CH), np.array(Y)
t0 = d["tier0"][idx]; r_ex, r_mi, r_ch = t0[:, 0] > 0, t0[:, 1] > 0, t0[:, 2] > 0; yr = d["tier1"][idx]
mu_t = d["mu_true"][idx]; th_t = np.degrees(theta_phi_beam(mu_t[:, 0], mu_t[:, 1], mu_t[:, 2])[0]); P_t = np.linalg.norm(mu_t, axis=1) / 1e3; pT_t, pL_t = P_t * np.sin(np.radians(th_t)), P_t * np.cos(np.radians(th_t))
def kin(y): th = np.degrees(theta_phi_beam(y[:, 0], y[:, 1], y[:, 2])[0]); P = np.linalg.norm(y[:, :3], axis=1) / 1e3; return P, th, y[:, 6] / 1e3, y[:, 5]
Pr, thr, Er, zr = kin(yr); Pm = [kin(y) for y in Y]
fid_r = in_volume(np.c_[yr[:, 3:6], np.zeros(len(idx))], "fiducial"); fid_m = np.array([in_volume(np.c_[y[:, 3:6], np.zeros(len(idx))], "fiducial") for y in Y])
sel_r = r_ex & fid_r & (thr < 20) & r_mi & r_ch; sel_m = np.array([EX[k] & fid_m[k] & (Pm[k][1] < 20) & MI[k] & CH[k] for k in range(K)])
R = {"n": int(len(idx)), "members": a.members, "tables": {}}
def binned(name, xr, xm, wr, wm, edges, lab):
    """bin fractions of real (xr with mask wr) and of each member (xm[k] with wm[k]); the ensemble mean, spread, residual, errors."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        fr = ((xr >= lo) & (xr < hi) & wr).sum() / len(idx); fm = np.array([((xm[k] >= lo) & (xm[k] < hi) & wm[k]).sum() / len(idx) for k in range(K)])
        if fr * len(idx) < 100: continue
        mean, ens = fm.mean(), fm.std(ddof=1); stat = np.sqrt(fr * (1 - fr) / len(idx)); samp = np.sqrt(mean * (1 - mean) / len(idx)) / np.sqrt(K)
        rows.append({"bin": f"[{lo:g},{hi:g})", "real": fr, "mean": mean, "ens": ens, "stat": stat, "samp": samp, "resid": mean - fr, "members": fm.tolist()})
    R["tables"][name] = {"xlabel": lab, "rows": rows}; return rows
def report(key, name, rows):
    res = np.array([r["resid"] for r in rows]); ens = np.array([r["ens"] for r in rows]); st = np.array([r["stat"] for r in rows]); sp = np.array([r["samp"] for r in rows])
    p0 = res / np.sqrt(st ** 2 + sp ** 2); p1 = res / np.sqrt(st ** 2 + sp ** 2 + ens ** 2)
    print(f"{name:34s} {len(rows):3d} bins  pull RMS without / with the ensemble term {np.sqrt(np.mean(p0**2)):.2f} / {np.sqrt(np.mean(p1**2)):.2f}   median ens/stat {np.median(ens/st):.2f}   bins with ens > |resid| {100*np.mean(ens > np.abs(res)):.0f}%   median |resid|/mean {100*np.median(np.abs(res)/np.array([r['mean'] for r in rows])):.2f}%  median ens/mean {100*np.median(ens/np.array([r['mean'] for r in rows])):.2f}%", flush=True)
    R["tables"][key]["summary"] = {"pull_rms_no_ens": float(np.sqrt(np.mean(p0**2))), "pull_rms_ens": float(np.sqrt(np.mean(p1**2))), "median_ens_over_stat": float(np.median(ens/st)), "frac_ens_gt_resid": float(np.mean(ens > np.abs(res)))}
ones = np.ones(len(idx), bool); onesK = np.ones((K, len(idx)), bool)
print(f"{K} members, {len(idx):,} {a.subset} test events; reco fraction real {r_ex.mean():.4f}, members {EX.mean(1).round(4).tolist()}\n")
report("marg_muP", "marginal: reco muon P [GeV]", binned("marg_muP", np.where(r_ex, Pr, -1), [np.where(EX[k], Pm[k][0], -1) for k in range(K)], ones, onesK, np.r_[np.linspace(0, 10, 21), 12, 15, 20, 30], "reco muon P [GeV]"))
report("marg_theta", "marginal: reco muon angle [deg]", binned("marg_theta", np.where(r_ex, thr, -1), [np.where(EX[k], Pm[k][1], -1) for k in range(K)], ones, onesK, np.r_[np.linspace(0, 30, 31), 35, 40, 50, 70], "reco muon angle [deg]"))
report("marg_recoil", "marginal: log10 reco recoil [MeV]", binned("marg_recoil", np.where(r_ex & (Er > 0), np.log10(np.clip(Er * 1e3, 1, None)), -9), [np.where(EX[k] & (Pm[k][2] > 0), np.log10(np.clip(Pm[k][2] * 1e3, 1, None)), -9) for k in range(K)], ones, onesK, np.linspace(1, 4.5, 36), "log10 reco recoil [MeV]"))
report("marg_vz", "marginal: reco vertex z [mm]", binned("marg_vz", np.where(r_ex, zr, -1), [np.where(EX[k], Pm[k][3], -1) for k in range(K)], ones, onesK, np.linspace(4000, 9000, 26), "reco vertex z [mm]"))
PT = [0, .075, .15, .25, .325, .4, .475, .55, .7, .85, 1, 1.25, 1.5, 2.5, 4.5]; PL = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 15, 20, 40, 60]
report("reco_by_pT", "P(reco) by true pT", binned("reco_by_pT", pT_t, [pT_t] * K, r_ex, EX, PT, "true muon pT [GeV]"))
report("sel_by_pT", "P(selected) by true pT", binned("sel_by_pT", pT_t, [pT_t] * K, sel_r, sel_m, PT, "true muon pT [GeV]"))
report("sel_by_pL", "P(selected) by true pL", binned("sel_by_pL", pL_t, [pL_t] * K, sel_r, sel_m, PL, "true muon pL [GeV]"))
report("selrec_by_pT", "selected by reco pT", binned("selrec_by_pT", np.where(sel_r, Pr * np.sin(np.radians(thr)), -1), [np.where(sel_m[k], Pm[k][0] * np.sin(np.radians(Pm[k][1])), -1) for k in range(K)], ones, onesK, PT, "reco muon pT [GeV]"))
report("selrec_by_pL", "selected by reco pL", binned("selrec_by_pL", np.where(sel_r, Pr * np.cos(np.radians(thr)), -1), [np.where(sel_m[k], Pm[k][0] * np.cos(np.radians(Pm[k][1])), -1) for k in range(K)], ones, onesK, PL, "reco muon pL [GeV]"))
# ---- per-event spread between members (events reconstructed by every member), binned in true pT and angle
allex = EX.all(0) & EX2; P2, _, E2, z2 = kin(Y2); resP = np.abs(Pm[0][0] - P2) / (0.5 * (Pm[0][0] + P2)) / np.sqrt(2); resz = np.abs(Pm[0][3] - z2) / np.sqrt(2)  # per-draw resolution from two draws of the nominal
sdP = np.std([Pm[k][0] for k in range(K)], 0, ddof=1) / np.mean([Pm[k][0] for k in range(K)], 0); sdE = np.std([Pm[k][2] for k in range(K)], 0, ddof=1) / np.clip(np.mean([Pm[k][2] for k in range(K)], 0), 1e-3, None); sdz = np.std([Pm[k][3] for k in range(K)], 0, ddof=1)
print("\nper-event between-member spread (events reconstructed by all members), median: rel. muon P, rel. recoil, vertex z [mm]")
R["per_event"] = {}
for name, x, edges in (("true pT [GeV]", pT_t, PT), ("true angle [deg]", th_t, [0, 5, 10, 15, 20, 30, 45, 90])):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = allex & (x >= lo) & (x < hi)
        if m.sum() < 200: continue
        rows.append({"bin": f"[{lo:g},{hi:g})", "n": int(m.sum()), "relP": float(np.median(sdP[m])), "relE": float(np.median(sdE[m])), "dz": float(np.median(sdz[m])), "resP": float(np.median(resP[m])), "resz": float(np.median(resz[m]))})
        print(f"  {name:16s} {rows[-1]['bin']:12s} n {m.sum():7d}  P {100*rows[-1]['relP']:.2f}% (resolution {100*rows[-1]['resP']:.2f}%)  recoil {100*rows[-1]['relE']:.2f}%  z {rows[-1]['dz']:.1f} mm (resolution {rows[-1]['resz']:.1f} mm)")
    R["per_event"][name] = rows
json.dump(R, open(out / "spread.json", "w"), indent=1, default=float)
# ---- figure: residual and ensemble band per bin for the selection tables and two marginals
fig, axs = plots.plt.subplots(2, 3, figsize=(14, 7)); axs = axs.ravel()
for ax, name in zip(axs, ("marg_muP", "marg_theta", "marg_recoil", "sel_by_pT", "sel_by_pL", "selrec_by_pT")):
    T = R["tables"][name]; rows = T["rows"]; x = np.arange(len(rows)); mean = np.array([r["mean"] for r in rows]); real = np.array([r["real"] for r in rows]); ens = np.array([r["ens"] for r in rows]); st = np.array([r["stat"] for r in rows])
    ax.fill_between(x, -ens / real, ens / real, color=plots.PALETTE["model"], alpha=0.35, label="between-member spread"); ax.errorbar(x, (mean - real) / real, st / real, fmt="o", ms=3, color=plots.PALETTE["real"], label="ensemble mean - open dataset (stat. error)")
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([r["bin"] for r in rows], rotation=70, fontsize=5.5); ax.set_xlabel(T["xlabel"]); ax.set_ylabel("relative to the open dataset"); ax.set_title(name, fontsize=9, loc="left")
axs[0].legend(frameon=False, fontsize=7); plots.save(fig, out / "figures" / "ensemble_spread.png"); print("written", out)
