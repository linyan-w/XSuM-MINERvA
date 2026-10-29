#!/usr/bin/env python
"""Slim AnaTuples straight from the xrootd door (no local ROOT copy) until a disk budget is reached.

Usage: scripts/curate/slim_remote.py PLAYLIST.txt OUT_DIR [--budget-gb 9.5] [--max-files N]
Files already slimmed in OUT_DIR are skipped. uproot streams only the requested baskets (~1.5% of each file).
"""
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from xsum.io.slim import slim_file


def dir_size_gb(d):
    return sum(p.stat().st_size for p in pathlib.Path(d).glob("*")) / 1e9


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("playlist"); ap.add_argument("out_dir"); ap.add_argument("--budget-gb", type=float, default=9.5)
    ap.add_argument("--max-files", type=int, default=10**6)
    a = ap.parse_args()
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    urls = [l.strip() for l in open(a.playlist) if l.strip()]
    done = 0
    for url in urls:
        stem = pathlib.Path(url).stem
        if (out / f"{stem}.truth.parquet").exists() and (out / f"{stem}.reco.typenames.json").exists():
            print(f"skip {stem} (exists)"); continue
        used = dir_size_gb(out)
        if used > a.budget_gb:
            print(f"budget reached: {used:.2f} GB > {a.budget_gb} GB"); break
        if done >= a.max_files:
            break
        t0 = time.time()
        try:
            res = slim_file(url, str(out))
        except Exception as e:  # keep going on a bad file, report at the end
            print(f"FAILED {stem}: {e!r}"); continue
        done += 1
        print(f"{stem}: reco {res['reco']['entries']} truth {res['truth']['entries']} entries, "
              f"{time.time()-t0:.0f} s, dir now {dir_size_gb(out):.2f} GB", flush=True)
    print(f"finished: {done} new files, {dir_size_gb(out):.2f} GB in {out}")
