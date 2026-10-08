# DQ stride-four structure sampling default

The user accepted the measured accuracy tradeoff and requested promotion and
publication on September 18. The standard DQ worker now retains every fourth
node of its original continuum-refined structure mesh, plus both endpoints.
The rule exactly matches the research experiment: 20,493 to 5,124 points in
the two cooler cases. It is applied after refinement, once per newly generated
domain grid. There is no atmosphere reuse or parameter fitting.

The opacity data, four actual refractive ray angles, gray initialization,
depth-extension policy, nonlinear solver, all five equilibrium tolerances,
4-GiB memory guard, and final 218,520-point synthesis are unchanged. The final
absolute bolometric tolerance is still 0.002, with no normalization. Runtime
metadata records the stride, node counts, and unthinned final synthesis under
protocol `refractive-current-energy-ca2024-swan-complete-stride4-v4`.

This publication also includes the previously approved 2024 C–A/completed
Swan data and exact caching/continuum batching changes described in the
[preceding release record](dq-completed-default-2026-09-17.md). Rejected helium
physics changes and unrelated research edits are not included.

## Evidence before promotion

| Case | From-scratch result | Native optical max error | 3 Å FWHM max error |
| --- | --- | --- | --- |
| J1225, 6294 K, log g 7.924, log C/He −5.33 | 17.58 min, qualified | 1.49% | 0.29% |
| J1311, 5529 K, 8.178, −5.27 | Intentionally stopped at 27.46 min, unconverged | Unqualified | Unqualified |

For J1225, structure took 677.58 s and final synthesis 376.94 s. The
independent Fbol/(σTeff⁴) was 0.99917649; a narrow UV difference reached
4.36%, so the good smoothed optical result is not a full-spectrum 1% bound.
For J1311, a separate warm reconvergence on the same coarse grid passed all
five gates but differed by 8.13% native optical, 1.29% at 3 Å and 0.58% at
10 Å. Its luminosity ratio was 1.00178413. It is explicitly not a from-scratch
timing result. The cold attempt was stopped while local energy error remained
8.03%; this does not prove that thinning caused the startup difficulty.

The earlier J1235 experiment completed the coarse solve plus staged full
synthesis in approximately 23.9 minutes, versus an archived dense-grid cold
run of 59.0 minutes. This historical ≈2.47× ratio is not a contemporaneous
paired benchmark; the cooler cases have no default cold timing controls.

Accuracy references are certified dense-grid atmospheres with packaged-default
equivalence checks, at unchanged literature parameters. Coarse-grid errors
are relative to that numerical reference, not observations. These results
do not establish uniform 1% accuracy, observational accuracy, or robust cold
convergence over the entire DQ range. Promotion accepts those limitations;
none of the convergence tests was relaxed to force a successful result.

## Release checks

Unit tests protect the exact experimental subset, endpoint retention, invalid
inputs, actual runtime default, per-domain regeneration without double
thinning, and unchanged final grid. Fixed-atmosphere spectral regressions
protect the previously approved physics; they are not cold-convergence tests.

A fresh public `compute_dq` check uses J1225, with no saved atmosphere or grid:

```python
from wd_spectra import DQConfig, compute_dq
result = compute_dq(
    DQConfig(6294., 7.924, -5.33, maximum_seconds=2400),
    output_directory="results/dq-stride4-default/j1225-public",
)
```

The targeted sampling/public-contract suite passes **44 tests**, with one
slow cold canary deselected. The full regular suite passes **933 tests**,
with five missing-optional-data skips and eight slow canaries deselected,
in 337.29 seconds. `git diff --cached --check` passes. No source or
constitutive data edits were made while the cold qualification was running.

Both cp39 and cp312 macOS ARM64 wheels build. The cp39 wheel was installed
into a fresh temporary target and tested from outside the checkout, with
imports explicitly verified to come from that installation. All seven
constitutive data files passed checksum validation, and **27 tests passed**
in 200.01 seconds, including the sampling contracts, continuum equivalence,
and both fixed-state spectral regressions. Four pytest warnings concern
missing marker registration/cache access when deliberately using `/dev/null`
instead of the checkout's pytest configuration; none is a numerical failure.
The cp312 wheel was built but not runtime-tested (its build environment lacks
SciPy). The cp39 wheel SHA256 is
`556f53ef7e80ead18d230deda74c2bf0e4c58fefca5afd2b020794ee1126baa5`.

The fresh public J1225 calculation reproduced the research experiment's
5,124-point structure grid and all four saved structure fields **bit-for-bit**
at 40, 41, and 42 depths. Its final 42-depth atmosphere passed all five gates:

| Measured check | Value | Unchanged limit |
| --- | ---: | ---: |
| All-depth flux | 7.60658968e-4 | 0.002 |
| Cell-local energy | 7.54278666e-5 | 0.002 |
| Temperature stationarity | 0 | 0.0002 |
| Source closure | 2.04516495e-15 | 1e-6 |
| Bottom-boundary response | 4.64691965e-6 | 0.002 |

The public worker **completed**, and `compute_dq` read its result successfully
(caller exit 0). The full 218,520-point wavelength and flux arrays are
**bit-for-bit identical** to the accepted cold experiment. Independent
Fbol/(σTeff⁴) = **0.999176494939219**, passing the original 0.2% limit
without rescaling. All 115 production-source hashes remained unchanged,
and the runtime also verified its constitutive data identities.

Worker time was **1374.53 seconds (22.91 minutes)**. Other test/build jobs ran
concurrently during part of the solve; this is a release qualification, not
an isolated speed benchmark or a paired timing comparison. The atmosphere
needed the same 40→41→42-node extensions as the original experiment.

Local evidence: `results/dq-stride4-default/j1225-public/run.json`, SHA256
`4ec35888e6b0bdb8b0f73dc8fb526de51c7ced65cdc330c78daf1d70bb09ab3d`.
The independent spectrum SHA256 is
`a3afa45d464f78375eb836c17e1c624c599d3d5ad353efed435d4f6f8b8c85ac`,
also identical to the accepted experiment's saved spectrum. All launched
test and qualification processes have exited; no background calculation is
required for this release.
