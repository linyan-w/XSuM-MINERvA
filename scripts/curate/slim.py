#!/usr/bin/env python
"""Slim one or more AnaTuple ROOT files to Parquet.  Usage: scripts/curate/slim.py OUT_DIR FILE.root [FILE.root ...]"""
import sys, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from xsum.io.slim import slim_file

if __name__ == "__main__":
    out_dir, files = sys.argv[1], sys.argv[2:]
    for f in files:
        t0 = time.time()
        res = slim_file(f, out_dir)
        for tag, r in res.items():
            size = pathlib.Path(r["path"]).stat().st_size / 1e6
            print(f"{pathlib.Path(f).name} [{tag}] {r['entries']} entries -> {r['path']} ({size:.1f} MB)")
        print(f"  {time.time() - t0:.0f} s")
