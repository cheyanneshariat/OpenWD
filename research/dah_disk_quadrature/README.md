# DAH disk-quadrature diagnostics

These are diagnostic outputs, **not new frozen regression controls**.
`data/` contains baseline, 32-, 16- and 8-Å spectra and their original run
records from the October 8 fixed-atmosphere investigation. The stellar
parameters and immutable input atmospheres are in `tests/data/dah_paper`.
The option was the only synthesis override. Defaults, tolerances and
historical controls were not changed.

Run a cheap comparison of the saved spectra:

```sh
PYTHONPATH=src python research/dah_disk_quadrature/benchmark.py compare \
  --output results/dah-disk-comparison
```

It uses each observation's bundled resolution, one archived velocity per
object and a 3820–6950 Å mask. Fluxes are converted to F_nu before applying
the resolution kernel; relative differences share the same 8-Å denominator.
No variant is independently flux-scaled or fitted. `comparison.json` records
the exact definition, file hashes, bin counts and original timings.

To repeat a saved-atmosphere synthesis on the current code:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENWD_NUM_THREADS=1 \
  PYTHONPATH=src nice -n 10 python research/dah_disk_quadrature/benchmark.py \
  synthesize j1018+0111 --drift 16 --output results/dah-j1018-drift16
```

Omit `--drift` to exercise the public default. Output directories must not
exist. Each refined synthesis takes minutes on one core; run one process
at a time and measure local cost before scaling. This explicitly reuses an
atmosphere for a numerical diagnostic and never claims a fresh atmosphere
solve or magnetic radiative equilibrium. It does not enable RWA, magnetic
EOS, altered continuum, fitted widths or changed geometry.

The archived spectra were generated before the PR review added validation
and boundary/resource guards or independent limb sampling. They use one
mean limb ray per field bin. Current drift-resolved synthesis keeps up to
eight limb subgroups in every field bin and shares their mean-field
opacity/source calculation. The archive is historical evidence, not an
accuracy or runtime measurement of this corrected angular quadrature.
