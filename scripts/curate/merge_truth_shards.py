#!/usr/bin/env python
"""Concatenate Truth-layout Parquet shards (docs/generator_input.md) into one file, in the order given, with row groups of
--rows events so that scripts/infer/surrogate_to_ntuple.py --range reads one slice at a time. All shards must share one
schema. The event count and, when present, the sum of `gen_weight` are printed.
Usage: scripts/curate/merge_truth_shards.py OUT.parquet SHARD.parquet [SHARD.parquet ...] [--rows 200000]"""
import argparse
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("shards", nargs="+"); ap.add_argument("--rows", type=int, default=200000); a = ap.parse_args()
writer = None; buf = []; nbuf = 0; total = 0; wsum = 0.0; has_w = None
def flush():
    global buf, nbuf, writer
    if not buf: return
    t = pa.concat_tables(buf); buf = []; nbuf = 0
    if writer is None: writer = pq.ParquetWriter(a.out, t.schema, compression="zstd")
    writer.write_table(t, row_group_size=a.rows)
for s in a.shards:
    t = pq.read_table(s); total += t.num_rows; buf.append(t); nbuf += t.num_rows
    if has_w is None: has_w = "gen_weight" in t.column_names
    if has_w: wsum += float(np.sum(t["gen_weight"].to_numpy()))
    if nbuf >= a.rows: flush()
flush(); writer.close()
print(f"{a.out}: {total:,} events from {len(a.shards)} shards" + (f", sum gen_weight {wsum:,.1f}" if has_w else "") + f", {pq.ParquetFile(a.out).metadata.num_row_groups} row groups")
