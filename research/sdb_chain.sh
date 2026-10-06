#!/bin/bash
# research/sdb_chain.sh TAG TEFF LOGG LOG_HE_H LOG_C_H : LTE H/He -> hybrid NLTE H/He (MALI) -> NLTE carbon.
set -u
TAG=$1; TEFF=$2; LOGG=$3; HE=$4; C=$5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src:research
D=src/wd_spectra/data/runtime; O=results/sdb
[ -f $O/$TAG-lte-hhe-40/run-summary.json ] || python research/sdb_lte_hhe.py --teff $TEFF --logg $LOGG --log-he-h $HE --data-root $D --output $O/$TAG-lte-hhe-40 > $O/$TAG-lte.log 2>&1; echo "=== lte exit $? $(date +%H:%M:%S)"
[ -f $O/$TAG-hybrid/run-summary.json ] || python research/sdb_hybrid_nlte.py $O/$TAG-lte-hhe-40 --data-root $D --hydrogen-levels 16 --accelerated-lambda --population-iterations 200 200 200 --output $O/$TAG-hybrid > $O/$TAG-hybrid.log 2>&1; echo "=== hybrid exit $? $(date +%H:%M:%S)"
rm -rf $O/$TAG-carbon; python research/sdb_trace_metals.py $O/$TAG-lte-hhe-40 $O/$TAG-hybrid --abundance C=$C --data-root $D --output $O/$TAG-carbon > $O/$TAG-carbon.log 2>&1; echo "=== carbon exit $? $(date +%H:%M:%S)"
echo "=== done"
