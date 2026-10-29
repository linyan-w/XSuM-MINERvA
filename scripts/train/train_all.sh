#!/usr/bin/env bash
# Train the released models from the slimmed open dataset (data/slim_1A, see scripts/curate/slim_remote.py).
#
# Three stages per model (see docs/model.md and paper/schematic):
#   1. event stage  : set encoder, event flag heads, hadron multiplicity head, event flow        (train_event.py)
#   2. prong stage  : adds the vertex-plane head and the prong set flow, fine-tunes everything    (train_prong.py, warm start)
#   3. vertex head  : re-trains the vertex-plane head alone, conditioned on the sampled vertex z   (train_prong.py --only-vtx)
# The released weights/full/model.pt is the output of stage 3; the three ensemble members repeat the chain with
# --seed 1..3 (initialisation and shuffling only; the train/validation/test split by subrun is the same);
# the 2p2h holdout model repeats it with --exclude-inttype 8 (GENIE interaction type 8 = 2p2h held out of training).
# One chain is ~4 h on an RTX 3090 for 15.8 M events. Every stage is skipped when its output exists.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}; OUT=${OUT:-reports}; mkdir -p "$OUT"
chain() {  # chain NAME [extra args for every stage]
  local n=$1; shift
  [ -f "$OUT/${n}_stage1/metrics.json" ]        || $PY -u scripts/train/train_event.py "$OUT/${n}_stage1" --epochs 30 --steps 64 --population all "$@"
  [ -f "$OUT/${n}_stage2/metrics_tier2.json" ]  || $PY -u scripts/train/train_prong.py "$OUT/${n}_stage2" --init "$OUT/${n}_stage1/model.pt" --epochs 16 --lr 2e-4 --steps 64 "$@"
  [ -f "$OUT/$n/model.pt" ]                     || $PY -u scripts/train/train_prong.py "$OUT/$n" --init "$OUT/${n}_stage2/model.pt" --epochs 3 --lr 2e-4 --steps 64 --vtx-cond-z far --only-vtx "$@"
}
chain full
for s in 1 2 3; do chain full_s$s --seed $s; done
chain holdout_2p2h --exclude-inttype 8

# Evaluations used by the paper (metrics.json / metrics_tier2.json in each directory; the figures read them):
eval_on() {  # eval_on MODEL_DIR OUT_DIR [selection args]: --eval-only reads and writes in OUT_DIR, so the model is copied there
  local m=$1 o=$2; shift 2; [ -f "$o/metrics_tier2.json" ] && return; mkdir -p "$o"; cp "$m/model.pt" "$o/"; $PY -u scripts/train/train_prong.py "$o" --eval-only --steps 64 "$@"
}
eval_on "$OUT/full"         "$OUT/eval_full_on_2p2h"     --only-inttype 8
eval_on "$OUT/holdout_2p2h" "$OUT/eval_holdout_on_2p2h"  --only-inttype 8
eval_on "$OUT/holdout_2p2h" "$OUT/eval_holdout_control"  --only-inttype 1 2 3 4 6 --subset ccnumu
echo "done: $OUT/{full,full_s1,full_s2,full_s3,holdout_2p2h}/model.pt"
