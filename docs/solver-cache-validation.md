# Contained solver improvements

This report covers the original cache/robustness bundle, based on released commit
`d9640656dce01bc6fc788c612d7a153c0d51c756`. Qualification on 2026-10-06/07 used
short component tests, fixed atmospheres and reduced Jacobian fixtures, plus
one completed public DAZ cold-start pair and bounded standard D6 attempts.
Those measurements did not include the native frequency-Voigt optimization;
its separate evidence is documented in
[native frequency-Voigt validation](native-frequency-voigt-validation.md).

## Changes

- Ordinary hydrogen, helium and D6 LTE adapters retain centered material
  probes' depth-sized Rosseland vectors within one linearization. This avoids
  rebuilding metal opacity after the opposite probe displaces the material
  cache. The shared option defaults to false for composition-owned adapters.
- Mixed H/He EOS evaluates the neutral-H fraction directly from the log Saha
  ratio, preserving its small positive tail and temperature response when the
  ion fraction rounds to unity. No population floor is introduced.
- The shared Newton driver's iteration-cap exit uses the same accepted-state
  convergence predicate as its normal loop exit, including an explicitly
  enabled physical certificate for limited proposals.
- DO/DAO Jacobians reuse the canonical residual's transfer, coefficients and
  statistical-equilibrium candidate/rate matrix. Failed trials and evaluation
  overrides cannot replace this canonical bundle.
- Deferred hot-trace-metal synthesis callbacks retain the evaluated mapping,
  iteration and copied metadata after internal iteration updates. This does
  not make caller-held NumPy arrays immutable.

DQ-specific code, PG1159 diagnostic scheduling, streaming, alternative linear
solvers, stationarity policies and finite-difference retry experiments are
excluded.

## Measurements

Python 3.9.16, NumPy 1.26.4 and SciPy 1.11.1 on macOS arm64, with numerical
threads pinned to one. The native extension was unchanged in these cache-bundle
measurements. The LTE comparison
used five warmed alternating pairs on eight-depth fixtures, 80 continuum
points and at most 16 structure metal lines. Both variants use the candidate
source, toggling reuse off/on.

| Case | Reuse off median (s) | Reuse on median (s) | Time reduction |
| --- | ---: | ---: | ---: |
| DA | 0.380 | 0.377 | Timing noise |
| DB | 1.078 | 1.075 | Timing noise |
| DAZ | 0.946 | 0.681 | 28% |
| DZ | 6.209 | 3.987 | 36% |
| D6 | 0.603 | 0.349 | 42% |

Residuals and Jacobians were bitwise identical. Metal-opacity builds fell
from four to two per measured Jacobian in DAZ, DZ and D6. These reductions
measure Jacobian construction, not complete model runtime or convergence.
The LTE timings preceded promotion of the three independent hot-only source
changes; their LTE/EOS source matches the final cache/robustness candidate.
All selected regression tests and hot comparisons used the complete production
code of that bundle, without the subsequent native frequency-Voigt changes.

Reduced real-atom DO and DAO fixtures at eight and sixteen depths also
preserved bitwise-identical seeds, states, residuals and Jacobians. They
removed one transfer and coefficient build, plus one hydrogen solve in DAO.
The four single timing pairs ranged from about 3% faster to 7% slower and do
not establish a significant hot-model speedup. One full-grid canonical
physics bundle is retained; memory scaling needs broader qualification.

## Public cold-start screen

Fresh processes called the public `run_model` with `require_convergence=True`,
standard quality, the default wavelength grid and no initial atmosphere,
checkpoint, neighbor, warmup or line-budget override. The release checkout
was pinned to the commit above; the candidate used only the frozen
cache/robustness patch described here. Runs were serial on the same macOS arm64 Python/NumPy/SciPy stack and
one-thread controls stated above. Source, data and loaded-native SHA256
inventories and lossless callback checkpoints were retained.

Standard DAZ G149-28 (8600 K, log g 8.10 and the release research abundances)
completed in 273.748861 s for the release and 247.861861 s for the candidate:
9.46% less public `run_model` elapsed time in this single pair. That timer
includes selection, physical initialization, solve, synthesis, saving and
observer I/O. All 33 callback states and diagnostics, final atmosphere,
physical diagnostic arrays and spectrum were bitwise identical. Both runs
recorded 34 radiative-equilibrium iterations, 33 shared-driver residual
evaluations, six Jacobians, 16 accepted iterations and eight rejected trials.
Callback count is not accepted-Newton-iteration count; recorded counters do
not cover every hidden initializer/opacity/synthesis call.

Both DAZ runs pass the five required structure-grid checks with identical
values: flux 2.678794e-7, local energy 5.068815e-4, unrestricted log-temperature
correction 6.076422e-5, source closure 1.221157e-15 and boundary escape
1.625636e-9. Completed-run source/data/native inventories were unchanged and
the physical inputs match. This single pair provides no repeated-pair
uncertainty estimate or attribution to one source hunk; independent-grid and
full-physics validation remain outside the certificate.

Both standard J1637 D6 calls reached their original 14400-second monotonic
caps during the convective preconditioner, with 12 candidate and 11 release
callback events. The 11 common checkpoint states are bitwise identical,
diagnostics and phase/iteration labels match, and their initial canonical
configurations, frozen source subsets, data/native and runtime controls pass
the separate prefix checks. No final model, spectrum or physical certificate
was produced, so the complete cold comparison stays unqualified and all five
final D6 gates stay unmeasured. The unmatched candidate callback does not
establish divergence.

At common callback 11, raw model times were 14285.369892 s release and
12610.541615 s candidate. Separately subtracting only recorded preceding
process pauses gives 12506.636025 s and 12596.063512 s. Candidate pauses total
14.478104 s; release pauses total 1778.733866 s for separately authorized
native tests in another checkout. The frozen cold numerical source and native
extension were preserved. Caps were unchanged and included those pauses.
Candidate laptop sleep accounts for a 2737.127090 s UTC-versus-model-clock gap;
the model/monotonic timer excludes that sleep, so it is not subtracted again.
These are matched callback milestone times, with no complete D6 runtime or
convergence claim.

The frozen cold observer and comparator are identified by SHA256
`9245ebd621de565a079d367a97e57851f86aa43ce0734e13062b5c7efd632a48`
and `3d27bf7de4a695cbf517c96924d87e1eb6abe3d77906c567c3245a14808527f6`.
The separate artifact reports preserve raw clocks, pause journals, missing
final-artifact vetoes and the strict prefix checks. The native optimization
evaluated during release suspensions belongs to a separate change.

## Validation

330 selected tests passed: 326 component/public-adapter tests and four saved
spectrum controls (DA 5000 K, DB 10000 K, DAZ G149-28 and DZ PG1225). Coverage
includes shared-driver policy/correction/domain checks, adaptive tangents,
mixed EOS conservation/Saha balance, hot material/radiation responses, cache
invalidation and deferred trace callbacks. The strengthened failed-trial
regression also passed in a separate rerun.

Against the unmodified release, the four new highly ionized EOS/derivative
cases fail and the warm neutral/partly ionized control passes. The new
one-iteration physical-certificate case fails while its two-iteration control
passes, confirming that the fixes address reproducible release defects.

Short component qualification and fixed-state benchmark commands each had a
120-second outer cap; none of those timed out. An
initial invalid test filename and transient OpenMP shared-memory startup
failure were corrected/rerun successfully. Contained benchmark provenance and source
identity checks passed. The bounded runner's completion and process-group
timeout paths were checked separately with a deliberately short-lived
command and an expected timeout.

See [benchmark instructions](../benchmarks/README.md) for reproducible
contained comparisons. Complete D6/hot cold-start convergence, cross-platform
performance and broader spectral qualification remain outside this commit's
evidence. The completed DAZ pair qualifies its declared structure grid.

## Combined performance candidate

The cache/robustness and native frequency-Voigt bundles were combined without
additional numerical edits for a fresh J1637 cold start on 2026-10-07.
Before launch, 131 focused tests passed in 2.51 seconds, both new native APIs
were available, and an eight-depth D6 Jacobian comparison retained bitwise
residual/Jacobian equality with Rosseland reuse off/on, reducing material
builds from four to two. These are contained checks; their test count overlaps
the separate bundle screens above.

The combined public standard J1637 cold start uses no supplied atmosphere,
checkpoint or warmup and has an eight-hour work cap. Its final convergence,
spectrum and whole-model runtime remain pending as of 2026-10-07. Intermediate
callback timing is not a completed cold-start speedup measurement.

The PR applies these same numerical changes on top of main commit
`cfe2d2ff99a434cd502696c1eb7b2f5daf8d11c5`, preserving the subsequently merged
physics/boundary fixes and bounded J1637 regression. The historical measurements
above use their declared released baseline; they are not a requalification of
that newer main branch. Integration checks run through the existing PR CI.
No new convective phase-handoff, conditioning-normalization, bounded-direction
or Jacobian-refresh experiment is included.
