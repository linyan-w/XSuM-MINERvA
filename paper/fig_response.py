#!/usr/bin/env python
"""Paper figures in one style for the full test sample and the held-out 2p2h events:
 (muon) median P_reco/P_true vs true muon P; (recoil) median reconstructed recoil vs true hadronic KE;
 (hadron) per species (proton, charged pion): fraction of events with a fit under that hypothesis (left axis) and the
 median leading fitted momentum (right axis) vs the true leading-hadron KE of that species.
Open dataset and one or more surrogates. Usage:
 paper/fig_response.py OUT_PREFIX --models A=weights/full [B=weights/holdout_2p2h] [--only-inttype 8] [--n 300000] [--replot]"""
import argparse, glob, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch
from torch.utils.data import DataLoader
from xsum.data.compact import load_compact, CompactDataset, collate, selection_from_config, subset_mask
from xsum.data.dataset import split_by_subrun
from xsum.prep.features import event_features
from xsum.train.prong_stage import load_model
from xsum.train.event_stage import to_dev, _pad
from xsum.eval import plots
NAMES = {"A": "full model", "B": "2p2h holdout"}  # legend names of the model labels
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--only-inttype", type=int, nargs="*", default=None); ap.add_argument("--subset", default=None, help="all-events models: restrict the test events to this subset (ccnumu, nc, other)"); ap.add_argument("--n", type=int, default=300000); ap.add_argument("--draws", type=int, default=8); ap.add_argument("--replot", action="store_true", help="re-draw from OUT_PREFIX_cache.pkl"); ap.add_argument("--no-band", nargs="*", default=[], help="model labels drawn as a line only, without a band (e.g. B: the holdout model has no ensemble)"); ap.add_argument("--ensemble", nargs="*", default=[], help="LABEL=DIR further ensemble members of a model (e.g. A=weights/full_s1): the band of that model is the between-member spread with common noise"); a = ap.parse_args()
ENS = {}; NOBAND = {f"surrogate ({NAMES.get(l, l)})" for l in a.no_band} | ({"surrogate"} if a.no_band and len(a.models) == 1 else set())
for spec in a.ensemble: lab_, md_ = spec.split("="); ENS.setdefault(lab_, []).append(md_)
cfg0 = torch.load(a.models[0].split("=")[1] + "/model.pt", map_location="cpu", weights_only=False)["config"]
d = load_compact(sorted(p[:-len(".truth.parquet")] for p in glob.glob("data/slim_1A/*.truth.parquet")), **selection_from_config(cfg0)); split = split_by_subrun(d["subrun"], seed=0)
te = np.where(split == 2)[0]
if a.only_inttype: te = te[np.isin(d["intType"][te], a.only_inttype)]
if a.subset: te = te[subset_mask(d, a.subset)[te]]
idx = np.random.default_rng(0).permutation(te)[:a.n]
cls, mom, mask = _pad(d, idx); X, names = event_features(cls, mom, mask, d["ctx"][idx])
kep = X[:, names.index("maxKE_p")]; kpi = np.maximum(X[:, names.index("maxKE_pip")], X[:, names.index("maxKE_pim")]); nch = d["n_charged_true"][idx].astype(int); Pt = np.linalg.norm(d["mu_true"][idx], axis=1) / 1e3
from xsum.prep.features import class_mass
_P = np.linalg.norm(mom, axis=-1); _m = class_mass(cls); _had = mask & ~np.isin(cls, [1, 2, 13])  # hadrons: every final-state particle but the charged leptons (mu-, e, mu+); event_features' sumKE_had excludes only the mu-
keh = (np.where(_had, np.sqrt(_P ** 2 + _m ** 2) - _m, 0.0)).sum(1)
Pt = np.where((d["current"][idx] == 1) & (np.abs(d["incoming"][idx]) == 14), Pt, np.nan)  # muon response only for events with a true muon (CC nu_mu and nu_mu-bar); NC and nu_e have no muon
def summarise(reco, y, prongs, offs):
    """per-event quantities for reconstructed events: muon ratio, recoil [GeV], leading proton-fit and pion-hypothesis P [MeV] (nan if none), n prongs, n with kinematics"""
    n = np.diff(offs); seg = np.repeat(np.arange(len(n)), n); lead = np.full(len(n), -1.0); leadpi = np.full(len(n), -1.0)
    np.maximum.at(lead, seg, np.where(prongs[:, 4] > 0.5, prongs[:, 5], -1.0)); np.maximum.at(leadpi, seg, np.where(prongs[:, 2] > 0.5, prongs[:, 3], -1.0))
    nkin = np.bincount(seg, weights=(prongs[:, 2] > 0.5).astype(float), minlength=len(n))
    return {"reco": reco, "ratio": np.linalg.norm(y[:, :3], axis=1) / 1e3 / Pt[reco], "recoil": y[:, 6] / 1e3, "pP": np.where(lead > 0, lead, np.nan), "piP": np.where(leadpi > 0, leadpi, np.nan), "n": n.astype(float), "nkin": nkin}
import pickle
SE = {}
if not a.replot:
    first = None; S = {}
    for spec in a.models:
        lab, D = spec.split("="); model, tf, ptf = load_model(f"{D}/model.pt"); ds = CompactDataset(d, idx, tf, 0, prong_tf=ptf)
        if first is None:  # open dataset, decoded with the first model's transforms (identical raw values)
            r = d["reco_exists"][idx] & ds.valid; ir = idx[r]
            offs = np.concatenate([[0], np.cumsum(np.diff(d["p_offsets"])[ir])]); pr = np.concatenate([d["prongs"][d["p_offsets"][e]:d["p_offsets"][e + 1]] for e in ir])
            S["open dataset"] = summarise(r, d["tier1"][ir], pr, offs); first = lab
        members = ENS.get(lab, [])
        def one_pass(mdl, seed):
            torch.manual_seed(seed); X1, EX, PR, PM = [], [], [], []
            with torch.no_grad():
                for i_, b in enumerate(DataLoader(ds, batch_size=4096, collate_fn=collate, num_workers=4)):
                    torch.manual_seed(seed * 100003 + i_)  # per-batch seed: common noise across members
                    s = mdl.sample(to_dev(b, "cuda"), 64); X1.append(s["x1"].cpu()); EX.append(s["exist"].cpu()); PR.append(s["prongs"].cpu()); PM.append(s["pmask"].cpu())
            Np = max(p.shape[1] for p in PR); PR_ = torch.cat([torch.nn.functional.pad(p, (0, 0, 0, Np - p.shape[1])) for p in PR]).numpy(); PM_ = torch.cat([torch.nn.functional.pad(p, (0, Np - p.shape[1])) for p in PM]).numpy()
            ex = torch.cat(EX).numpy(); X1_ = torch.cat(X1).numpy(); y = tf.inverse(X1_[ex], d["mu_true"][idx][ex], d["ctx"][idx][ex])
            pm = PM_[ex]; pr = ptf.inverse(PR_[ex][pm]); offs = np.concatenate([[0], np.cumsum(pm.sum(1))]); return summarise(ex, y, pr, offs)
        key = f"surrogate ({NAMES.get(lab, lab)})" if len(a.models) > 1 else "surrogate"
        S[key] = [one_pass(model, kd) for kd in range(a.draws)]  # regenerations with different noise: the sampling (statistical) spread of a sample of this size
        if members: SE[key] = [one_pass(model, 0)] + [one_pass(load_model(f"{md}/model.pt")[0], 0) for md in members]  # one common-noise pass per member: the between-member (model) spread
        print(lab, "sampled", len(S[key]), "regenerations,", len(SE.get(key, [])), "members", flush=True)
if not a.replot: pickle.dump({"S": S, "SE": SE, "Pt": Pt, "keh": keh, "kep": kep, "kpi": kpi, "nch": nch}, open(a.out + "_cache.pkl", "wb"))
else: C_ = pickle.load(open(a.out + "_cache.pkl", "rb")); S, Pt, keh, kep, kpi, nch = (C_[k] for k in ("S", "Pt", "keh", "kep", "kpi", "nch")); SE = C_.get("SE", {})
STY = {"open dataset": (plots.PALETTE["real"], "o", "-")}; cyc = [(plots.PALETTE["model"], "s", "--"), (plots.PALETTE["third"], "^", ":")]
for i, k in enumerate([k for k in S if k != "open dataset"]): STY[k] = cyc[i]
def stat(x, yv, a_, b_, kind):
    s = (x >= a_) & (x < b_)
    if kind == "frac": return (np.isfinite(yv[s]).mean(), np.sqrt(np.isfinite(yv[s]).mean() * (1 - np.isfinite(yv[s]).mean()) / max(s.sum(), 1))) if s.sum() >= 30 else (np.nan, np.nan)
    q = yv[s][np.isfinite(yv[s])]
    if len(q) < 30: return np.nan, np.nan
    if kind == "mean": return q.mean(), q.std() / np.sqrt(len(q))
    return np.median(q), 1.2533 * q.std() / np.sqrt(len(q)) if False else (np.median(q), 1.4826 * np.median(np.abs(q - np.median(q))) * 1.2533 / np.sqrt(len(q)))[1]
def prof(ax, xfull, key, edges, kind="median", logx=True, xs=None, q=None, legend=True):
    """q: per-quantity style override {"marker", "ls", "mfc"} for two quantities on one panel (twin axes)"""
    xs = np.sqrt(edges[:-1] * edges[1:]) if xs is None else xs
    ok = np.isfinite(np.array([stat(xfull[S["open dataset"]["reco"]], S["open dataset"][key], a_, b_, kind)[0] for a_, b_ in zip(edges[:-1], edges[1:])]))
    xs = np.asarray(xs, float)[ok]; edges_ok = [(a_, b_) for (a_, b_), o in zip(zip(edges[:-1], edges[1:]), ok) if o]
    for k, v in S.items():
        c, mk, ls = STY[k]
        if q: mk, ls = q["marker"], q["ls"]
        if k == "open dataset":
            m, e = np.array([stat(xfull[v["reco"]], v[key], a_, b_, kind) for a_, b_ in edges_ok]).T
            ax.errorbar(xs, m, e, fmt=mk, ms=3.5, color=c, mfc=(q or {}).get("mfc", c), capsize=0, label="open simulation dataset", zorder=3)
        else:
            M = np.array([[stat(xfull[dv["reco"]], dv[key], a_, b_, kind)[0] for a_, b_ in edges_ok] for dv in v])
            mu, sd = np.nanmean(M, 0), np.nanstd(M, 0, ddof=1)  # regenerations: sampling spread
            if k in SE:  # members with common noise: the model spread, added in quadrature
                ME = np.array([[stat(xfull[dv["reco"]], dv[key], a_, b_, kind)[0] for a_, b_ in edges_ok] for dv in SE[k]]); sd = np.sqrt(sd ** 2 + np.nanstd(ME, 0, ddof=1) ** 2)
            if k not in NOBAND: ax.fill_between(xs, mu - sd, mu + sd, color=c, alpha=0.35, lw=0)
            ax.plot(xs, mu, ls=ls, color=c, lw=1.4, label=k)
    if logx: ax.set_xscale("log")
    if legend: ax.legend(frameon=False, fontsize=7)
from matplotlib.lines import Line2D
fig, ax = plots.plt.subplots(figsize=(4.2, 3.3))
prof(ax, Pt, "ratio", np.logspace(np.log10(0.5), np.log10(40), 13)); ax.set_xlabel("true muon momentum [GeV]"); ax.set_ylabel(r"median $P_\mathrm{reco}/P_\mathrm{true}$ of the muon")
plots.save(fig, pathlib.Path(a.out + "_muon.png"))
fig, ax = plots.plt.subplots(figsize=(4.2, 3.3))
prof(ax, keh, "recoil", np.logspace(np.log10(30), np.log10(20000), 12)); ax.set_yscale("log"); ax.set_xlabel("true hadronic kinetic energy [MeV]"); ax.set_ylabel("median recoil energy [GeV]")
plots.save(fig, pathlib.Path(a.out + "_recoil.png"))
fig, (ax1, ax2) = plots.plt.subplots(1, 2, figsize=(8.4, 3.3))  # muon and recoil as two panels of one figure (paper)
prof(ax1, Pt, "ratio", np.logspace(np.log10(0.5), np.log10(40), 13)); ax1.set_xlabel("true muon momentum [GeV]"); ax1.set_ylabel(r"median $P_\mathrm{reco}/P_\mathrm{true}$ of the muon")
prof(ax2, keh, "recoil", np.logspace(np.log10(30), np.log10(20000), 12), legend=False); ax2.set_yscale("log"); ax2.set_xlabel("true hadronic kinetic energy [MeV]"); ax2.set_ylabel("median recoil energy [GeV]")
ax1.text(0.03, 0.05, "(a)", transform=ax1.transAxes, va="bottom", fontsize=10); ax2.text(0.03, 0.95, "(b)", transform=ax2.transAxes, va="top", fontsize=10)
plots.save(fig, pathlib.Path(a.out + "_response.png"))
QF, QP = {"marker": "o", "ls": "-", "mfc": None}, {"marker": "s", "ls": "--", "mfc": "white"}
e = np.array([50, 80, 120, 180, 270, 400, 600, 900, 1400, 2500])
SPEC = [(x, key, hyp, sp) for x, key, hyp, sp in ((kep, "pP", "proton", "proton"), (kpi, "piP", "pion", "charged-pion")) if (x[S["open dataset"]["reco"]] >= e[0]).sum() >= 300]  # species absent from the sample (no pions in 2p2h) get no panel
fig, axs = plots.plt.subplots(1, len(SPEC), figsize=(4.8 * len(SPEC), 3.4), squeeze=False); axs = axs[0]
for ax, (x, key, hyp, sp) in zip(axs, SPEC):
    QF["mfc"] = None; prof(ax, x, key, e, kind="frac", q=QF, legend=False); ax.set_ylim(0, 1.05)
    ax2 = ax.twinx(); prof(ax2, x, key, e, q=QP, legend=False); ax2.grid(False); ax.set_xlim(e[0], e[-1])
    ax.set_xlabel(f"true leading {sp} kinetic energy [MeV]"); ax.set_ylabel(f"fraction of events with a {hyp} fit"); ax2.set_ylabel(f"median leading {hyp}-fit momentum [MeV]")
    ax2.spines["right"].set_visible(True)
hs = [Line2D([], [], color=STY[k][0], lw=6, alpha=0.6 if k != "open dataset" else 1, label="open simulation dataset" if k == "open dataset" else k) for k in S]
hq = [Line2D([], [], color="0.3", marker="o", ls="-", label="fraction with a fit (left)"), Line2D([], [], color="0.3", marker="s", mfc="white", ls="--", label="median fitted momentum (right)")]
fig.legend(handles=hs + hq, frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(hs) + len(hq) if len(axs) > 1 else 2)
fig.tight_layout(); fig.savefig(a.out + "_hadron.png", bbox_inches="tight"); plots.plt.close(fig); print("written", a.out)
if len(SPEC) == 1:  # one species (2p2h: protons only): recoil and the proton reconstruction as two panels of one figure (paper)
    x, key, hyp, sp = SPEC[0]
    fig, (ax2, ax3) = plots.plt.subplots(1, 2, figsize=(9.4, 3.5))
    prof(ax2, keh, "recoil", np.logspace(np.log10(30), np.log10(20000), 12), legend=False); ax2.set_yscale("log"); ax2.set_xlabel("true hadronic kinetic energy [MeV]"); ax2.set_ylabel("median recoil energy [GeV]")
    QF["mfc"] = None; prof(ax3, x, key, e, kind="frac", q=QF, legend=False); ax3.set_ylim(0, 1.05)
    ax3b = ax3.twinx(); prof(ax3b, x, key, e, q=QP, legend=False); ax3b.grid(False); ax3.set_xlim(e[0], e[-1]); ax3b.spines["right"].set_visible(True)
    ax3.set_xlabel(f"true leading {sp} kinetic energy [MeV]"); ax3.set_ylabel(f"fraction of events with a {hyp} fit"); ax3b.set_ylabel(f"median leading {hyp}-fit momentum [MeV]")
    for ax, lab in ((ax2, "a"), (ax3, "b")): ax.text(0.03, 0.95, f"({lab})", transform=ax.transAxes, va="top", fontsize=10)
    fig.legend(handles=hs + hq, frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout(); fig.savefig(a.out + "_combined.png", bbox_inches="tight"); plots.plt.close(fig); print("written", a.out + "_combined.png")
