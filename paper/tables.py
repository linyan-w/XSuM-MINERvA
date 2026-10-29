#!/usr/bin/env python
"""Tables III and IV of the paper from the closure-classifier results (scripts/evaluate/closure_classifier.py ->
paper/inputs/closure/closure.json): the seven-variable closure on the test sample by block, and the 2p2h holdout
columns, with the statistical resolution of each AUC (the label-swap null spread where it was measured, otherwise the
Mann-Whitney spread of an uninformative classifier on the evaluation half, sqrt((n1 + n2 + 1) / (12 n1 n2)) with n1 = n2 = n/2).
Usage: paper/tables.py [--closure paper/inputs/closure/closure.json] [--out paper/tables]"""
import argparse, json, pathlib
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--closure", default="paper/inputs/closure/closure.json"); ap.add_argument("--out", default="paper/tables"); a = ap.parse_args()
C = json.load(open(a.closure)); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); f3 = lambda x: f"{x:.3f}"


def resolution(r):
    if "null_cond_std" in r: return float(np.mean([r["null_marg_std"], r["null_cond_std"]]))
    m = r["n"] / 2; return float(np.sqrt((2 * m + 1) / (12 * m * m)))


A = C["A_full"]  # the full model on the full test sample
rows = [("reco variables", "ccinc_marg"), ("\\quad muon block", "mu_marg"), ("\\quad vertex block", "vtx_marg"), ("\\quad recoil block", "recoil_marg"),
        ("reco variables + truth features", "ccinc_cond"), ("\\quad + muon block", "mu_cond"), ("\\quad + vertex block", "vtx_cond"), ("\\quad + recoil block", "recoil_cond")]
t3 = ["\\begin{tabular}{lc}", "\\toprule", "Classifier input & test sample \\\\", "\\midrule"] + [f"{lab} & {f3(A[k])} \\\\" for lab, k in rows] + \
     ["\\midrule", f"test events & {A['n']:,} \\\\".replace(",", "{,}"), f"statistical resolution & {f3(resolution(A))} \\\\", "\\bottomrule", "\\end{tabular}"]
(out / "closure_test_sample.tex").write_text("\n".join(t3) + "\n")

cols = [("A_twop", "full model", "2p2h"), ("B_twop", "2p2h holdout", "2p2h")]
t4 = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", "closure AUC & " + " & ".join(c[1] for c in cols) + " \\\\", " & " + " & ".join(c[2] for c in cols) + " \\\\", "\\midrule",
      "reconstructed only & " + " & ".join(f3(C[c[0]]["ccinc_marg"]) for c in cols) + " \\\\", "with truth features & " + " & ".join(f3(C[c[0]]["ccinc_cond"]) for c in cols) + " \\\\",
      "\\midrule", "statistical resolution & " + " & ".join(f3(resolution(C[c[0]])) for c in cols) + " \\\\", "\\bottomrule", "\\end{tabular}"]
(out / "holdout_summary.tex").write_text("\n".join(t4) + "\n")
print("written", out / "closure_test_sample.tex", out / "holdout_summary.tex")
