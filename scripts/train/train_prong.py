#!/usr/bin/env python
"""Train/evaluate the M3 surrogate (Tier 2 prongs + vertex plane head), warm-started from M2.
Usage: scripts/train/train_prong.py OUT_DIR [--epochs 12] --init STAGE1/model.pt [--n-files N] [--eval-only] [--steps 64]"""
import sys, pathlib, argparse, json, glob
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--slim-dir", default="data/slim_1A"); ap.add_argument("--n-files", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--bs", type=int, default=1024); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--init", required=True, help="stage-one checkpoint (scripts/train/train_event.py)"); ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--exclude-inttype", type=int, nargs="*", default=None, help="GENIE intType codes held out of train/val (8 = 2p2h)"); ap.add_argument("--only-inttype", type=int, nargs="*", default=None, help="evaluate only test events of these intType codes"); ap.add_argument("--subset", choices=["ccnumu", "nc", "other"], default=None, help="all-events models: evaluate only this subset of the test events")
    ap.add_argument("--ke-cut", type=float, default=10.0, help="hadron KE threshold [MeV] for input tokens"); ap.add_argument("--neutrons", dest="neutrons", action="store_true", default=True, help="admit neutrons as input tokens (default)"); ap.add_argument("--no-neutrons", dest="neutrons", action="store_false"); ap.add_argument("--tier2-only", action="store_true"); ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--cache", default=None, help="write the sampling pass over the evaluated events to this .npz (one draw reused by all evaluations and subsets)"); ap.add_argument("--from-cache", default=None, help="read the sampling pass from this .npz instead of sampling (a full-split cache serves its subsets)")
    ap.add_argument("--vtx-cond-z", nargs="?", const=True, default=False, help="vertex-class head conditioned on the reco z residual (consistent snapping): bare flag = raw residual, 'far' = near/far indicator only"); ap.add_argument("--seed", type=int, default=None, help="initialisation and shuffling seed (ensemble members); the split stays at seed 0"); ap.add_argument("--only-vtx", action="store_true", help="train only the vertex head, warm-started from a finished M3 model (--init)")
    a = ap.parse_args()
    from xsum.train import event_stage as m2, prong_stage as m3
    from xsum.data.compact import load_compact, selection_from_config
    from xsum.data.dataset import split_by_subrun
    stems = sorted(p[:-len(".truth.parquet")] for p in glob.glob(f"{a.slim_dir}/*.truth.parquet"))
    if a.n_files: stems = stems[:a.n_files]
    def restrict(d, idx, ld, tf, ptf=None):
        if not a.only_inttype and not a.subset: return idx, ld
        from torch.utils.data import DataLoader
        from xsum.data.compact import CompactDataset, collate, subset_mask
        keep = idx["test"]
        if a.only_inttype: keep = keep[np.isin(d["intType"][keep], a.only_inttype)]
        if a.subset: keep = keep[subset_mask(d, a.subset)[keep]]
        idx = dict(idx, test=keep); ld = dict(ld, test=DataLoader(CompactDataset(d, keep, tf, 0, prong_tf=ptf), batch_size=a.bs, collate_fn=collate, num_workers=4))
        print(f"evaluating on {len(keep):,} test events (intType {a.only_inttype}, subset {a.subset})"); return idx, ld
    print(f"{len(stems)} files")
    if a.eval_only:
        import torch
        c = torch.load(pathlib.Path(a.out_dir) / "model.pt", map_location="cpu", weights_only=False)["config"]
        d = load_compact(stems, **selection_from_config(c)); split = split_by_subrun(d["subrun"], seed=0)
        model, tf, ptf = m3.load_model(pathlib.Path(a.out_dir) / "model.pt")
        idx, ds, ld = m3.make_loaders(d, split, tf, ptf, a.bs, 0)
        if a.max_test:  # quick checks on a subset of the test split
            from torch.utils.data import DataLoader
            from xsum.data.compact import CompactDataset, collate
            idx["test"] = idx["test"][:a.max_test]
            ld["test"] = DataLoader(CompactDataset(d, idx["test"], tf, 0, prong_tf=ptf), batch_size=a.bs, collate_fn=collate, num_workers=4)
    else:
        d, split, tf, ptf, idx, ld = m3.train(stems, a.out_dir, a.init, a.epochs, a.bs, a.lr, exclude_inttype=a.exclude_inttype, vtx_cond_z=a.vtx_cond_z, only_vtx=a.only_vtx, init_seed=a.seed)  # selection inherited from --init
        model, tf, ptf = m3.load_model(pathlib.Path(a.out_dir) / "model.pt")
    idx, ld = restrict(d, idx, ld, tf, ptf)
    m2.write_data_table(stems, d, split, a.out_dir, None if a.eval_only else a.epochs, sum(p.numel() for p in model.parameters()))
    m1 = None
    from xsum.train.samples import get_samples
    SM = get_samples(model, ld["test"], idx["test"], a.steps, cache=a.cache, from_cache=a.from_cache)  # one draw for both evaluations
    if a.tier2_only:
        M2 = m3.evaluate_tier2(model, tf, ptf, d, idx["test"], ld["test"], a.out_dir, n_steps=a.steps, samples=SM)
        print(json.dumps({"closure_prongs": M2["closure_prongs"], "validation": M2["validation"]}, indent=1, default=float)); sys.exit(0)
    M1 = m2.evaluate(model, tf, d, idx["test"], ld["test"], a.out_dir, n_steps=a.steps, m1_metrics=m1, samples=SM)
    M2 = m3.evaluate_tier2(model, tf, ptf, d, idx["test"], ld["test"], a.out_dir, n_steps=a.steps, samples=SM)
    print(json.dumps({"tier0": {k: round(v["logloss"], 4) for k, v in M1["tier0"].items()}, "auc_marginal": round(M1["classifier_auc_marginal"], 4), "auc_conditional": round(M1["classifier_auc_conditional"], 4),
                      "vertex": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in M2["vertex_class"].items() if not isinstance(v, list)},
                      "vertex_offset": M2["vertex_offset_to_nearest_plane"], "prong_counts": M2["prong_counts"], "closure_prongs": M2["closure_prongs"],
                      "prong_w1": {k: round(v["w1"], 4) for k, v in M2["prong_marginals"].items()}}, indent=1, default=float))
