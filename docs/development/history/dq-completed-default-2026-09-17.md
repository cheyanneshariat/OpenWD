# DQ 2024 C–A and Swan-completeness default

The default now uses the same combination tested in the fixed-literature
J1311, J1225, J0804 and J1803 research comparisons. The two packaged numerical
artifacts are unchanged, including their checksums. The runtime has no research
path dependency, no additional free stellar parameter, and no fitted opacity scale.

- `c2-ca-2024.npz`: Lino da Silva 2024 A coefficients, 63 bands, historical
  approximate envelope/origins, unchanged ExoMol populations and partition.
- `swan-completed.npz`: all 29,004 original Hornkohl lines plus 1,232,442
  ExoMol lines outside their per-band v/J coverage; no inside-envelope filling.
- Exact-temperature, immutable-strength/prefix LRU, capped at 1 GiB in synthesis
  and 32 MiB during structure iteration. The
  transfer's 4 GiB process guard and all physical convergence gates are unchanged.
- The original C–A and Hornkohl files are retained for reproduction. The old
  saved-spectrum test explicitly selects its historical physics; its fixture
  and numerical tolerance are not changed to accommodate new predictions.

## Scientific evidence

Earlier paired warm reconvergence and independent 218520-point checks gave:

| Case | Teff | Absolute optical RMS change | Optical shape RMS change |
| --- | ---: | ---: | ---: |
| J1311 | 5529 K | -13.13% | -19.57% |
| J1225 | 6294 K | -2.31% | +3.10% |
| J0804 | 5364 K | -0.52% | +0.24% |
| J1803 | 4400 K | -0.59% | +0.05% |

Negative means smaller residuals, not a fractional change in predicted flux.
All literature Teff/log g/C/He, mass and distance values were fixed. No hydrogen,
magnetism or final absolute normalization was introduced. Red normalization is
only a separately reported shape diagnostic. The cooler cases are effectively
neutral, with small blue-region regressions; this is not a uniform accuracy gain.
The original SDSS observations confirm the cooler figure-based comparison.
Warm results do not themselves establish cold-start convergence.

Sources and detailed research records:

- [Published 2024 C–A coefficients](https://indico.esa.int/event/466/contributions/9848/).
- [Blouin & Dufour (2019)](https://arxiv.org/abs/1910.06168), fixed stellar parameters.
- `research/dq_ca_2024_2026-09-17.md` and `research/dq_cool_validation_2026-09-17.md`.
- Source/data identities, per-star outputs, and qualification reports:
  `results/dq-completed-default/` (local generated validation artifacts).

## Release verification

Targeted suite: **92 passed, one cold canary deselected** in 146.50 seconds.
The full regular repository suite subsequently gives **917 passed, five
optional-data skips and eight deselected** in 337.25 seconds.
This includes public worker contracts, C–A population conventions, both line
lists' exact cache checks and memory-policy restoration on failure, original
Hornkohl identity, the unmodified historical spectrum fixture, and a separate
30,000-point J1311 default-spectrum fixture archived from the pre-promotion
research model. The new fixture is not a cold-start input.

The packaged runtime reproduces all four pre-promotion optical spectra
**bit-for-bit**, with unchanged production-source hashes throughout the check:
`results/dq-completed-default/packaged-equivalence-v3/report.json`.
An earlier equivalence attempt was invalidated by a concurrent cache-policy
edit; its source-consistency check refused qualification, so it is not counted.

Rebuilding the Swan artifact reproduces its complete SHA256 exactly. Rebuilding
the C–A artifact reproduces all five numerical arrays bit-for-bit; its archive
hash differs because the standalone builder records new provenance metadata.
Neither builder imports the research directory at runtime.

The cp312 macOS ARM64 wheel builds and contains both data files and the cache
module, with no research/results imports or packaged research output. Installed
outside the checkout, it loads and checksum-validates all seven data files and
passes **12 selected tests**, including both spectral fixtures and the continuum
batching comparison, in 191.32 seconds. Only two unregistered-marker warnings
arise from deliberately running outside the checkout without its pytest config.
Final wheel: `wheel-v2/openwd-0.1.3-cp312-cp312-macosx_11_0_arm64.whl`, SHA256
`17c672484b8a54ed9783072571dfc2566e0dfb5405efa3d6a31bd12469b176d2`.

The first new public cold attempt hit the unchanged 4-GiB guard at 4.14 GiB:
a 1-GiB thermal cache competed with retained structure-response arrays. The
fix caps that exact cache at 32 MiB during structure solves, restoring the
1-GiB cap for synthesis after response arrays are released. The second cold run
then reached a later 4.10-GiB peak. A short allocation probe isolated 717.7 MiB
of transient arrays inside the shared helium-continuum evaluator, much larger
than the Swan-map transients (at most about 67 MiB in the sampled state).

DQ now calls that unchanged continuum evaluator in 1,024-wavelength batches.
On the failing run's 28,310-by-40 grid, all 1,132,400 continuum values are
bit-for-bit identical and the peak extra allocation drops from **717.7 to
34.7 MiB**. CPU time is 1.84–1.98 seconds versus 1.53–1.88 seconds unbatched
in this small traced benchmark: the purpose is bounded memory, not a speedup
claim. Atomic line support and all population, opacity and transfer formulas
are unchanged. Artifacts: `memory-probe-stages.json` and
`continuum-batch-check.json` under the local results directory.

Fresh-process paired J1311 synthesis benchmarks on the same 30,000-point
grid give 145.20 seconds un-cached versus 79.12 seconds with the exact
thermal cache (CPU: 144.40 versus 78.78 seconds, **1.83x**). Both spectra
are bit-for-bit identical. Process peak RSS rises from 1.01 to 1.79 GiB;
the cache retains 770 MiB for 40 exact temperatures. Other work was running,
so these are not isolated wall-clock measurements or a cold-convergence
speedup claim. Reports: `cache-benchmark-{reference,cached}/report.json`.

No physics, memory guard or convergence tolerance was relaxed. The fresh
gray-start v3 qualification **completed through the public `compute_dq` API**,
including its output reader (process exit 0). The worker took 3537.49 seconds
(58.96 minutes) within the 4200-second budget. It extended its own lower domain
from 40 to 41 nodes when the first boundary-screening check failed, then
re-solved and passed all five atmosphere gates:

| Measured check | Value | Unchanged limit |
| --- | ---: | ---: |
| All-depth flux | 3.90371258e-4 | 0.002 |
| Cell-local energy | 1.88632427e-3 | 0.002 |
| Temperature stationarity | 0 | 0.0002 |
| Source closure | 2.37774197e-15 | 1e-6 |
| Bottom-boundary response | 9.81462143e-4 | 0.002 |

The independent **218,520-point** spectrum is finite and positive, with
`Fbol/(sigma*Teff^4) = 0.9999375372794359`, an absolute fractional error of
6.2463e-5, without normalization. Production source hashes and all constitutive
data identities are unchanged throughout. The 4-GiB process guard remained
enabled. This is a new cold qualification of the promoted default, not an
attribution of an older run; it does not establish depth-grid independence or
full observational accuracy. Report: `j1235-cold-v3/run.json`.

The bounded public cold qualification uses a new output directory:

```python
from wd_spectra import DQConfig, compute_dq

result = compute_dq(
    DQConfig(9347., 8.041, -4.107, maximum_seconds=4200),
    output_directory="results/dq-completed-default/j1235-cold-v3",
)
```

It runs with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. The time limit is a
resource budget, not a relaxed numerical criterion. Saved research atmospheres
and the regression fixture are not passed to this calculation.

Further research this turn is recorded in
`research/dq_c3_and_cminus_2026-09-17.md`. Charge/pressure-reclosed C3 chemistry
and a source-table check of the blue He-minus coefficient each improve the
tested star's absolute optical RMS by only about 0.5% after reconvergence;
both pass independent luminosity checks but remain research-only. Sampled
carbon-negative-ion opacity estimates are too small to justify another solve.
Optical C3 absorption remains a potentially important, untested lead requiring
a defensible temperature-dependent profile and checks on both J1311 and J1803.
None of these further physics changes is part of the new default.
