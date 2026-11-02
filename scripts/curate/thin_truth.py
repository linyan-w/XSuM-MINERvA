#!/usr/bin/env python
"""Thin a geometry-placed generator sample (scripts/curate/merge_truth_shards.py output) to a smaller exposure: every event is kept
with probability POT / gen_pot (seeded, row group by row group), `gen_pot` is set to POT so that
`sum(gen_weight) x POT_data / gen_pot` stays the prediction at the data exposure; nothing else changes.
Usage: scripts/curate/thin_truth.py IN.parquet OUT.parquet --pot 9.702e19 [--seed 0] [--rows 200000]"""
import argparse, json, pathlib
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("--pot", type=float, required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--rows", type=int, default=200000); a = ap.parse_args()
pf = pq.ParquetFile(a.inp); pot_gen = float(pf.read_row_group(0, columns=["gen_pot"])["gen_pot"][0].as_py()); f = a.pot / pot_gen; assert 0 < f <= 1, (a.pot, pot_gen)
rng = np.random.default_rng([a.seed, 20261102]); writer = pq.ParquetWriter(a.out, pf.schema_arrow, compression="zstd"); n_in = n_out = 0; w_in = w_out = 0.0; buf = []; nbuf = 0
for i in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(i); keep = rng.random(t.num_rows) < f; w = t["gen_weight"].to_numpy(); n_in += t.num_rows; w_in += float(w.sum()); w_out += float(w[keep].sum())
    t = t.filter(pa.array(keep)); t = t.set_column(t.schema.get_field_index("gen_pot"), pf.schema_arrow.field("gen_pot"), pa.array(np.full(t.num_rows, a.pot))); buf.append(t); nbuf += t.num_rows; n_out += t.num_rows
    if nbuf >= a.rows: writer.write_table(pa.concat_tables(buf), row_group_size=a.rows); buf = []; nbuf = 0
if buf: writer.write_table(pa.concat_tables(buf), row_group_size=a.rows)
writer.close()
json.dump({"input": a.inp, "pot_gen": pot_gen, "pot": a.pot, "fraction": f, "seed": a.seed, "events_in": n_in, "events_out": n_out, "sum_gen_weight_in": w_in, "sum_gen_weight_out": w_out}, open(pathlib.Path(a.out).with_suffix("").with_suffix(".json"), "w"), indent=1)
print(f"{a.out}: {n_out:,} of {n_in:,} events kept (fraction {f:.4f}, POT {a.pot:.4e} of {pot_gen:.4e}), sum gen_weight {w_out:,.1f} of {w_in:,.1f}")
