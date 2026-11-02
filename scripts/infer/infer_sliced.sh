#!/usr/bin/env bash
# The ensemble on a large truth file in slices of SLICE events (the writer holds every universe draw of a slice in memory:
# about 20 GB for 4 M events), merged in order into one two-tree file. A slice whose output exists is skipped, so the
# script resumes after an interruption; the slice files are removed after the merge.
# Usage: scripts/infer/infer_sliced.sh TRUTH.parquet OUT.root [SLICE=2000000] [further surrogate_to_ntuple.py options]
set -euo pipefail; cd "$(dirname "$0")/../.."
T=$1; OUT=$2; SL=${3:-2000000}; shift $(( $# < 3 ? $# : 3 ))
PY=${PY:-.venv/bin/python}; W=${OUT%.root}_slices; mkdir -p "$W"
N=$($PY -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$T').metadata.num_rows)"); PARTS=""
for ((s=0, i=0; s<N; s+=SL, i++)); do
  P=$W/slice_$(printf %02d $i).root; PARTS="$PARTS $P"; [ -f "$P" ] && continue
  $PY scripts/infer/surrogate_to_ntuple.py weights/full "$T" "$P.new" --range $s $((s+SL)) --seed $i --ensemble weights/full_s1 weights/full_s2 weights/full_s3 "$@" && mv "$P.new" "$P"
done
$PY scripts/infer/merge_ntuples.py "$OUT.new" $PARTS && mv "$OUT.new" "$OUT" && rm -r "$W"
echo "written $OUT"
