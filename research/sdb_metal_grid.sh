#!/bin/bash
# Typical-sdB metals on every finished host of the H/He robustness grid (sdb_grid_check.sh).
#   research/sdb_metal_grid.sh GRID_DIR
# Writes GRID_DIR/<tag>-metals, one model at a time, single-threaded; no blanketing
# (the metal-free LTE H/He structure is kept).
#
# Abundances: Geier (2013, A&A 549, A110; VizieR J/A+A/549/A110) median over the
# 106-star sample of each element's measured log eps (stars with only upper limits
# omitted; C, N, Si, S average the ions measured in each star).  log N/N(H) =
# log eps - 12:  C 7.17, N 7.62, O 7.72, Si 6.71, S 6.77 (NLTE trace atoms);
# Mg 6.60, Al 5.60, Fe 7.37 (LTE line opacity in the fixed background).
set -u
GRID=${1:?grid directory}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src:research
DATA=src/wd_spectra/data/runtime
for HOST in "$GRID"/*-hybrid; do
  TAG=$(basename "$HOST" -hybrid)
  [ -f "$HOST/run-summary.json" ] && [ -f "$GRID/$TAG-lte/run-summary.json" ] || continue
  [ -f "$GRID/$TAG-metals/run-summary.json" ] && continue
  rm -rf "$GRID/$TAG-metals"
  echo "=== $TAG start $(date +%H:%M:%S)"
  python research/sdb_trace_metals.py "$GRID/$TAG-lte" "$HOST" --data-root "$DATA" \
    --abundance C=-4.83 --abundance N=-4.38 --abundance O=-4.28 --abundance Si=-5.29 --abundance S=-5.23 \
    --lte-abundance Mg=-5.40 --lte-abundance Al=-6.40 --lte-abundance Fe=-4.63 \
    --output "$GRID/$TAG-metals" > "$GRID/$TAG-metals.log" 2>&1
  echo "=== $TAG metals exit $? $(date +%H:%M:%S)"
done
echo "=== metal grid done $(date +%H:%M:%S)"
