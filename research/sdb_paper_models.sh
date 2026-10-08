#!/bin/bash
# Models for the sdB paper figure, at fixed published parameters (nothing fitted).
#   research/sdb_paper_models.sh [OUTPUT_DIR]   (default results/sdb/paper)
# One model at a time, single-threaded: LTE H/He structure -> hybrid NLTE H/He
# (MALI, H16, dominant He conservation row) -> metals (NLTE C N O Si S; other
# measured elements as LTE line opacity; upper limits omitted).
#
# HD 4539, Feige 38: Teff, log g, He/H from Schneider et al. (2018, hybrid
#   LTE/NLTE); metals from Geier (2013), log N/N(H), mean of the ions measured.
# LS IV-14 116: 35500 K, 5.85, -0.60 (Dorsch et al. 2020 refit of FORS2);
# Feige 46: 36100 K, 5.93, -0.32 (Latour et al. 2019b); metals from Dorsch et
#   al. (2020) Tables 6-7, log N/N(H).
set -u
OUT=${1:-results/sdb/paper}
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src:research
D=src/wd_spectra/data/runtime

host() {  # TAG TEFF LOGG LOG_HE_H
  [ -f "$OUT/$1-lte/run-summary.json" ] || { rm -rf "$OUT/$1-lte"; python research/sdb_lte_hhe.py --teff $2 --logg $3 \
    --log-he-h $4 --data-root $D --output "$OUT/$1-lte" > "$OUT/$1-lte.log" 2>&1; echo "=== $1 lte exit $? $(date +%H:%M:%S)"; }
  [ -f "$OUT/$1-hybrid/run-summary.json" ] || { rm -rf "$OUT/$1-hybrid"; python research/sdb_hybrid_nlte.py "$OUT/$1-lte" \
    --data-root $D --hydrogen-levels 16 --accelerated-lambda --population-iterations 200 200 200 \
    --output "$OUT/$1-hybrid" > "$OUT/$1-hybrid.log" 2>&1; echo "=== $1 hybrid exit $? $(date +%H:%M:%S)"; }
}
metals() {  # TAG LTE_DIR HYBRID_DIR ARGS...
  local tag=$1 lte=$2 hybrid=$3; shift 3
  [ -f "$OUT/$tag-metals/run-summary.json" ] && return
  rm -rf "$OUT/$tag-metals"
  python research/sdb_trace_metals.py "$lte" "$hybrid" --data-root $D --output "$OUT/$tag-metals" "$@" \
    > "$OUT/$tag-metals.log" 2>&1
  echo "=== $tag metals exit $? $(date +%H:%M:%S)"
}

host lsiv14116 35500 5.85 -0.60
metals lsiv14116 "$OUT/lsiv14116-lte" "$OUT/lsiv14116-hybrid" \
  --abundance C=-3.70 --abundance N=-3.82 --abundance O=-4.48 --abundance Si=-6.03 \
  --lte-abundance Ne=-4.50 --lte-abundance Mg=-5.40 --lte-abundance Ar=-5.55 --lte-abundance Ni=-4.62
host feige46 36100 5.93 -0.32
metals feige46 "$OUT/feige46-lte" "$OUT/feige46-hybrid" \
  --abundance C=-3.19 --abundance N=-3.57 --abundance O=-4.21 --abundance Si=-5.51 \
  --lte-abundance Ne=-4.31 --lte-abundance Mg=-5.05 --lte-abundance Ar=-5.75 --lte-abundance Fe=-4.64 \
  --lte-abundance Ni=-4.53
metals hd4539 results/sdb/hd4539-lte-hhe-40 results/sdb/hd4539-hybrid-dominant \
  --abundance C=-4.06 --abundance N=-4.10 --abundance O=-4.97 --abundance Si=-5.30 --abundance S=-5.17 \
  --lte-abundance Mg=-5.40 --lte-abundance Al=-6.40 --lte-abundance Ar=-4.83 --lte-abundance Fe=-4.62
metals feige38 results/sdb/feige38-schneider-lte-hhe-40 results/sdb/feige38-schneider-hybrid \
  --abundance C=-4.23 --abundance N=-3.98 --abundance O=-4.35 --abundance Si=-4.52 --abundance S=-4.87 \
  --lte-abundance Mg=-4.90 --lte-abundance Al=-6.20 --lte-abundance Ar=-4.90 --lte-abundance Fe=-4.84
echo "=== paper models done $(date +%H:%M:%S)"
