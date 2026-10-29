"""Metrics shared by the baselines and the later generative models."""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import log_loss, roc_auc_score


def binary_metrics(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    base = np.clip(np.full_like(p, y.mean()), eps, 1 - eps)
    return {"logloss": float(log_loss(y, p)), "logloss_marginal": float(log_loss(y, base)),
            "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"), "rate": float(y.mean())}


def multiclass_metrics(y, P, eps=1e-9):
    K = P.shape[1]
    marg = np.bincount(y, minlength=K) / len(y)
    ll = -np.log(np.clip(P[np.arange(len(y)), y], eps, None)).mean()
    llm = -np.log(np.clip(marg[y], eps, None)).mean()
    return {"logloss": float(ll), "logloss_marginal": float(llm), "accuracy": float((P.argmax(1) == y).mean()),
            "accuracy_marginal": float(marg.max())}


def calibration_by_bin(x, y, p, edges):
    """Observed vs predicted mean of a binary target in bins of x."""
    idx = np.digitize(x, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        s = idx == b
        if s.sum() < 20:
            continue
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(s.sum()),
                     "observed": float(y[s].mean()), "predicted": float(p[s].mean())})
    return rows


def sample_vs_real_1d(real, fake, lo=None, hi=None):
    """Wasserstein-1 and quantile summaries for one variable (same units as the input)."""
    lo = np.percentile(np.concatenate([real, fake]), 0.5) if lo is None else lo
    hi = np.percentile(np.concatenate([real, fake]), 99.5) if hi is None else hi
    r, f = np.clip(real, lo, hi), np.clip(fake, lo, hi)
    q = [16, 50, 84]
    return {"w1": float(wasserstein_distance(r, f)), "real_q16_50_84": np.percentile(r, q).tolist(),
            "fake_q16_50_84": np.percentile(f, q).tolist(), "n": int(len(real))}


def confusion(y, yhat, K):
    M = np.zeros((K, K), dtype=int)
    np.add.at(M, (y, yhat), 1)
    return M
