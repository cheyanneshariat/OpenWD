#!/bin/bash
# Sequential sdB robustness grid: LTE H/He structure, then hybrid NLTE H/He (MALI) on it.
#   research/sdb_grid_check.sh OUTPUT_DIR
# One model at a time, single-threaded.  Each point writes OUTPUT_DIR/<tag>-lte and <tag>-hybrid.
set -u
OUT=${1:?output directory}
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src:research
DATA=src/wd_spectra/data/runtime
# teff logg log_he_h
POINTS="20000 5.0 -3.0
20000 5.6 -3.0
25000 5.2 -3.0
25000 5.8 -3.0
30000 5.3 -3.0
30000 6.0 -3.0
35000 5.5 -2.5
35000 6.1 -2.5
40000 5.8 -2.0
40000 6.2 -2.0
28000 5.6 -4.0
28000 5.6 -1.0"
echo "$POINTS" | while read -r TEFF LOGG HE; do
  TAG="t${TEFF}-g${LOGG}-he${HE}"
  echo "=== $TAG start $(date +%H:%M:%S)"
  if [ ! -f "$OUT/$TAG-lte/run-summary.json" ]; then
    rm -rf "$OUT/$TAG-lte"
    python research/sdb_lte_hhe.py --teff "$TEFF" --logg "$LOGG" --log-he-h "$HE" --data-root "$DATA" \
      --output "$OUT/$TAG-lte" > "$OUT/$TAG-lte.log" 2>&1
    echo "=== $TAG lte exit $? $(date +%H:%M:%S)"
  fi
  if [ -f "$OUT/$TAG-lte/run-summary.json" ] && [ ! -f "$OUT/$TAG-hybrid/run-summary.json" ]; then
    rm -rf "$OUT/$TAG-hybrid"
    python research/sdb_hybrid_nlte.py "$OUT/$TAG-lte" --data-root "$DATA" --hydrogen-levels 16 \
      --accelerated-lambda --population-iterations 200 200 200 --output "$OUT/$TAG-hybrid" \
      > "$OUT/$TAG-hybrid.log" 2>&1
    echo "=== $TAG hybrid exit $? $(date +%H:%M:%S)"
  fi
done
echo "=== grid done $(date +%H:%M:%S)"
