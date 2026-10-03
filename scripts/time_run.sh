#!/bin/bash
# Wall-time one short BO run: scripts/time_run.sh <dataset> <method> <seed> <n_iter>
# Writes to a temp dir (never to results/). Example: scripts/time_run.sh dft_descriptors fabo 42 10
set -e
DATASET=$1; METHOD=$2; SEED=$3; N_ITER=$4
OUT=$(mktemp -d)
START=$(date +%s.%N)
python -m src.cli --mode "$METHOD" --input "data/${DATASET}.csv" --cache data/dft_G.json \
  --output-dir "$OUT" --n-initial 10 --n-iter "$N_ITER" --seed "$SEED" \
  --kernels Matern --acquisitions EI > "$OUT/run.log" 2>&1
END=$(date +%s.%N)
python - "$START" "$END" "$N_ITER" "$DATASET" "$METHOD" <<'PY'
import sys
s, e, n, d, m = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
print(f"{d} {m}: total {e - s:.1f}s for {n} iterations ({(e - s) / n:.1f}s/iter incl. startup)")
PY
grep -E "GP θ|Iter [0-9]+ →" "$OUT/run.log" | tail -n 3
