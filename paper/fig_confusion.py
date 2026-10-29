#!/usr/bin/env python
"""Paper figure: reconstructed prong multiplicity against true charged hadrons, row-normalised, open dataset and
surrogate, redrawn from an evaluation directory's metrics_tier2.json (the matrices of xsum.train.prong_stage.evaluate_tier2).
Usage: paper/fig_confusion.py [--eval weights/full] [--out paper/figures/multiplicity_confusion_truth.png]"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from xsum.eval import plots
ap = argparse.ArgumentParser(); ap.add_argument("--eval", default="weights/full"); ap.add_argument("--out", default="paper/figures/multiplicity_confusion_truth.png"); a = ap.parse_args()
M = json.load(open(pathlib.Path(a.eval) / "metrics_tier2.json"))["confusion_true_charged_vs_reco"]
rownorm = lambda C: (lambda C: C / np.clip(C.sum(1, keepdims=True), 1, None))(np.asarray(C, float))
labels = [str(k) for k in range(6)] + ["6+"]
fig, axs = plots.plt.subplots(1, 2, figsize=(7.8, 3.4), layout="constrained")
for ax, (C, title) in zip(axs, ((rownorm(M["masteranadev"]), "open simulation dataset"), (rownorm(M["surrogate"]), "surrogate"))):
    im = ax.imshow(C, vmin=0, vmax=1, cmap="Blues", origin="lower")
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] >= 0.005: ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if C[i, j] > 0.6 else "black")
    ax.set_xticks(range(7)); ax.set_yticks(range(7)); ax.set_xticklabels(labels, fontsize=8); ax.set_yticklabels(labels, fontsize=8); ax.grid(False)
    ax.set_xlabel("reco prongs", fontsize=9); ax.set_ylabel("true charged hadrons after FSI", fontsize=9); ax.set_title(title, fontsize=10, loc="left")
fig.colorbar(im, ax=axs.tolist(), shrink=0.85, pad=0.02, label="row fraction"); fig.savefig(a.out, bbox_inches="tight", pad_inches=0.05); plots.plt.close(fig); print("written", a.out)
