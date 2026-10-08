# Microphysics audit — 2026-09-10

[History index](README.md) · [Development guide](../README.md)

Scope: public main `f4dcfa5c5ac9b81ff9b1f2389046fbeea326036f` and the
affected local research implementations. The initial sections describe the
candidate audit chronologically; the final sections record the user-approved
DA default and regression promotion. The shared atmosphere solver was not
changed. Historical recovery archives and spectral controls were not edited.

## Findings and corrections

| Report | Public-code finding | Action |
| --- | --- | --- |
| Doubled MHD critical field | Confirmed in the common charged-perturber occupation probability | Remove the empirical factor of two; all H, He and metal callers use the same corrected function |
| Squared energy ratio in He I collision strength | Confirmed only in the local, unpublished `helium_nlte.py`; the public package has no hot-He NLTE module | Change `(Ry / delta-E)**2` to `Ry / delta-E` in the research closure |
| Unreleased metal-line emissivity buffer | Confirmed in the compiled routine distributed with the public package and in the research copy | Use one buffer-count constant for acquisition and both cleanup paths; release all 17 buffers |
| Abrupt H2–H2 CIA cutoff | Already fixed in the public loader; still present in the old research loader | Align the research H2–H2 loader with the public declining-tail policy; add shared regression coverage |

The public LTE Python modules do not currently call the affected NLTE metal-line
C routine. Its presence in the extension does not imply that ordinary DA/DB/DZ
runs were leaking this buffer. The research hot-metal NLTE module does call it.

The generic CIA table constructor remains conservative by default. The actual
Borysow **reader** enables extrapolation when every terminal temperature column
declines. At 5000 K the final coefficients are 6.523e-10 and 6.402e-10 at
16460 and 16480 cm^-1, respectively: the reported 6.4e-10 is correct, but
“still rising” is not. At 16479.99, 16480, and 16480.01 cm^-1 the public
interpolator returns approximately 6.40205994e-10, 6.402e-10, and 6.40194006e-10.
The table edge is 6067.96 Angstrom. This continuation is an extrapolation,
not new molecular-opacity data. A flat or rising input table is not given an
unphysical increasing exponential tail.

## Independent reference checks

The [TLUSTY reference manual II (2017)](https://arxiv.org/abs/1706.01935),
equations 107–108, gives the undoubled critical-field prefactor 8.59e14.
OpenWD's modern constants give 8.580024e14. The new tests compare with the
published rounded prefactor over nine electron densities, six levels, three
radiator charges, and both uncorrelated and correlated microfields.
TLUSTY 200's hardcoded doubling is **not** an undoubled reference; the old test
comment incorrectly described it that way.
The [operational manual](https://arxiv.org/abs/1706.01937) sets BERGFC=1.
[Tremblay & Bergeron (2009)](https://arxiv.org/abs/0902.4182) use the undoubled
convention with their nonideal profiles.

The He I research test checks all 33 allowed term pairs against the
`19.7363 T**(-3/2) exp(-u) gbar f / u` form of TLUSTY's CREGER, at the
**same** Gaunt factor. It does not assert that the approximate He I Gaunt
factor equals TLUSTY's exponential-integral prescription. Forbidden links and
the supplied Storey–Hummer data remain unchanged.
See the [TLUSTY guide](https://tlusty.oca.eu/tlusty/Tlusty2002/pdf/tlguide202.pdf),
atomic collision-rate prescriptions.

Resource tests reproduced one extra retained reference per call on the final
emissivity array before the fix, on both success and post-acquisition errors.
They now check successful accumulation, shape/dtype/contiguity failures, and
read-only output rejection. The corrected research C kernel also agrees with
its Python implementation.

## Spectral consequences — unchanged atmosphere structures

These comparisons reconstruct material properties on saved structures; they
do not certify fresh atmospheric equilibrium. No reference spectrum was
renormalized or overwritten.

| Case | Maximum relative change in significant flux | Integrated absolute change / reference flux |
| --- | ---: | ---: |
| DA 3000 K | 3.6e-8% | 1.4e-8% |
| DA 4000 K | 1.9e-7% | 2.1e-8% |
| DA 5000 K | 0.000049% | 0.00000017% |
| DA 12000 K | 5.02% | 0.229% |
| DA 20000 K | 3.97% | 0.203% |
| DB 10000 K | 0.000171% | 0.0000079% |
| DB 22000 K | 1.66% | 0.129% |
| DAB 9000 K | 0.121% | 0.0041% |
| DAB 20000 K | 30.7% | 0.141% |
| DZ PG 1225 | 0.444% | 0.0068% |
| DZ J0738 | 11.4% | 0.0393% |
| DAZ G149-28 | 1.35% | 0.354% |
| DAZ GALEX J1931 | 3.70% | 0.974% |

“Significant” means lambda-F-lambda above 1% of its peak, as in the existing
tests. The largest DAB 20000 and DZ J0738 differences lie in the far UV
(about 934 and 919 Angstrom). The DAZ controls cover limited spectral windows,
so their integrated differences are not bolometric differences.

For the DA controls, independently continuum-normalized maximum changes
(Halpha, Hbeta, Hgamma, Hdelta), in percentage points, are:

- 12000 K: 0.093, 0.110, 1.161, 3.188.
- 20000 K: 0.082, 0.119, 0.657, 2.297.

These differences are consistent with the changed occupation probabilities,
but consistency with the intended equations does not establish better
observational agreement. The immutable spectral regression gates remain
unchanged and continue to flag this candidate.

The broad component run returned 744 passed, 5 skipped, and 2 failures
(the public-default 12000-K and 20000-K DA spectral guards). Separate
fixed-structure diagnostics cover the 13 cases above, all with converged
radiation sources. Several exceed their existing spectral tolerances.
The final focused EOS/CIA/resource tests passed 59 checks in the public
package and 110 in the research package (the latter includes the He I NLTE
tests). The public normalization and compiled-resource checks were exercised
with Python 3.9 and 3.11.

## Cold-start checks and release status

Fresh checks use no saved atmosphere or neighboring model as a solver input.

| Case | Iterations | Equilibrium certificate | Existing regression gate |
| --- | ---: | --- | --- |
| DA 3000 K | 72 | Passed | Passed, including spectrum preservation |
| DA 20000 K | 39 | Passed | Failed old structure comparison |
| DB 22000 K | 30 | Passed | Failed old structure comparison |
| DAB 20000 K | 38 | Passed | Failed old spectrum comparison |

The first three took approximately 10.7, 6.2, and 7.2 minutes, respectively,
while running concurrently. Their maximum all-depth fractional flux residuals
were 0.0012691, 4.35e-8, and 1.84e-9; their certificates also check local energy,
temperature stationarity, source closure, and boundary screening.

The warm DA and DB tests failed the historical pressure/column-mass comparison
at up to 0.493% and 3.39%, respectively, not the equilibrium checks. On their
common spectral grids, the cold-start DA 20000 and DB 22000 spectra changed by
up to 3.96% and 1.69% in significant flux; integrated absolute differences were
0.121% and 0.0731%. The cold-start DA 3000 spectrum stayed within its existing
regression bounds. These results do not constitute full release qualification.

DAB 20000 took 19.1 minutes including 13 thermal-conditioning sweeps. Its
maximum all-depth flux residual was 8.58e-8, local energy residual 0.001564,
and final temperature correction 0.0001806, all within the unchanged checks.
Its spectrum changed by up to 30.9% in significant far-UV flux (2.43% of the
peak lambda-F-lambda scale). The integrated absolute spectral change divided
by the stellar flux was 0.107%; the 3500–7000-A band changed by -0.148%.

The selected cold-start run therefore returned one regression pass and three
regression failures, despite four verified equilibrium certificates. The
reference structures/spectra have deliberately not been overwritten. A
published-grid/observational assessment and explicit baseline review are
needed before promoting this candidate; other protected cold points were not
rerun in this audit.

The public request-physics revision is now
`openwd-0.1.3-qmhd-undoubled-v4`. A dedicated test rejects same-physics reuse
of checkpoints tagged with the preceding doubled-field revision. The cold
diagnostics above were run after the equation corrections and before this
final provenance-only revision bump; their source hashes record the corrected
equations, but their older request tags must not be used to bypass fresh
qualification or to claim an exact resume under the new revision.
After that bump, the focused public/provenance/documentation run passed
105 tests on Python 3.11, and the public EOS/CIA/resource/provenance subset
passed 69 tests on Python 3.9.

Local diagnostic products:
`results/microphysics-audit-20260910/`.
Reproduce fixed-structure measurements with:

```sh
OPENWD_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=src python research/audit_microphysics_spectra.py OUTPUT --jobs 4
```

This audit does not authorize replacing old regression fixtures or claiming
full protected qualification. Nothing has been committed or pushed.

## Follow-up: re-converged DA/DAB reference comparisons

The initial literature plots isolated the opacity correction on unchanged
atmospheres; they were not the final corrected equilibrium spectra. The
follow-up uses independently cold-converged atmospheres at the original
parameters and resolutions. The 20000-K DA and DAB use the completed corrected
cold starts above. Source/data hashes match those calculations exactly except
for the request-revision string; reversing that string change reproduces the
recorded `common.py` hash. This is reuse of completed scientific results, not
an atmosphere continuation or a bypass of the public checkpoint checks.

The 12000-K standard DA was freshly cold-started with the corrected public API:
46 reported iterations (including 16 thermal-conditioning sweeps), 168.6 s,
all equilibrium checks passed. Its maximum all-depth flux error is 3.93e-9,
local energy error 3.85e-6, and final logarithmic temperature correction 2.08e-6.
It still fails the existing old-spectrum preservation gate. That gate was not
relaxed, and convergence must not be mistaken for spectrum preservation.

The additional 9000-K DAB control did **not** qualify: after 34 iterations and
966.0 s, all-depth flux (2.23e-9), local energy (2.06e-4), source closure,
and boundary screening passed, but its unrestricted logarithmic temperature
correction was 0.00932 against a 0.0003 limit. The damped accepted step was
only 0.000146 because the line-search factor was 1/64. The solver stopped on
its residual/accepted-step condition; the independent certificate correctly
refused convergence. Raising the iteration budget alone would not address
that early stopping condition. No tolerance was relaxed and this control is
excluded from plots labelled re-converged. Its saved old spectral fixture has
no independent equilibrium certificate and used a supplied atmosphere, so
this comparison alone cannot establish that the correction introduced the
stationarity problem. The three significantly changed DA/DAB cases have
verified corrected cold starts; the extra 9000-K control remains unresolved.

Re-converged spectra are evaluated on the **exact original wavelength nodes**,
without smoothing or fitted flux scaling. A second broad-wavelength sampling
covers 100–1000000 Angstrom. Synthesis on the union of the comparison,
original cold-output, and broad grids reproduces every original cold-output
flux exactly at shared nodes, and leaves temperature, pressure, mass density,
electron density, and column mass unchanged. The input equilibrium certificate
belongs to the preceding cold solve, not to this wavelength-sampling operation.

For the two DAs, integrals over the common approximately 900–30000-A range,
divided by the specified stellar flux, are:

| DA | Old | Corrected, fixed atmosphere | Corrected, re-converged | Koester |
| --- | ---: | ---: | ---: | ---: |
| 12000 K | 0.9771004 | 0.9765077 | 0.9771450 | 0.9962861 |
| 20000 K | 0.9996060 | 1.0008710 | 0.9996037 | 0.9993373 |

Thus re-convergence returns their integrated flux scales to within 0.005% and
0.0003% of the old values, respectively. It does not remove the higher-Balmer
shape changes: in 3750–4500 A, the wavelength-integrated absolute discrepancy
relative to the Koester integrated flux changes from 4.12% to 5.39% at 12000 K
and from 1.21% to 0.70% at 20000 K. These absolute-flux measures include both
continuum and line differences; they are not fitted line-profile statistics.

The corrected broad sampled synthesis integrals are 0.97969 (DA 12000),
1.00031 (DA 20000), and 0.99258 (DAB 20000). They are **not** the atmospheric
flux residuals. The roughly 2% deficit of the 12000-K returned spectrum is
consistent with the known separate atmosphere/synthesis discretizations and
was already present over the old spectrum's available comparison range.
Re-convergence alone does not remove this pre-existing synthesis limitation.
No alternate transfer method or compensating normalization was introduced.

The DAB literature control is the Montreal/Warwick 1D LTE mixed H/He grid,
not a Koester mixed grid. All comparisons retain log g = 8 and H/He = 0.01.
Artifacts and their input hashes are under
`results/microphysics-audit-20260910/literature-reconverged/` and
`results/microphysics-audit-20260910/reconverged/`. The plotting entry point is
`research/plot_microphysics_literature.py --reconverged`; wavelength-only checks
use `research/resynthesize_microphysics_cold.py`. Production equations, default
transfer methods, and immutable reference spectra were not changed in this
follow-up.

## Follow-up: paper Figure 2, six independent cold starts

Figure 2 (`fig:da_spy_balmer`) is the six-object SPY/UVES Balmer-profile
comparison. It was regenerated using the corrected public `compute_da` API,
with no supplied atmosphere, stored checkpoint, or neighboring model. All six
use `DAConfig(..., quality="production")` at the original paper parameters,
with 100 structure depths. Three single-threaded workers ran in parallel.
Every model passed the independent equilibrium certificate and the final
synthesis source-closure check. Source hashes were unchanged during each run.

| Target | Teff (K) | log g | Reported iterations | Solve + synthesis (s) |
| --- | ---: | ---: | ---: | ---: |
| WD 1202-232 | 8615 | 8.042 | 36 | 325.8 |
| WD 0024-556 | 10157 | 8.737 | 32 | 305.4 |
| G29-38 | 11485 | 8.071 | 43 | 351.6 |
| PG 1015+161 | 19948 | 7.925 | 41 | 266.8 |
| CD-38 10980 | 24677 | 7.927 | 40 | 238.6 |
| GD 71 | 32959 | 7.731 | 44 | 251.4 |

The largest all-depth flux residual is 3.15e-7; the largest local-energy
residual is 0.001260; the largest unrestricted logarithmic temperature
correction is 0.000185. The certificate concerns the declared equations on
the structure grid, not independent grid refinement or full-physics validity.
The reported iterations include thermal conditioning.

The original 14804 wavelength nodes, Gaussian R=18500 convolution,
rest-frame observation processing, sideband continuum normalization,
Koester interpolation, and six-by-four layout are retained. Reprocessing
both the Koester reference and old OpenWD spectra reproduces all 24 original
line-profile RMS diagnostics to numerical precision. The historical plotting
helpers are used only for observation/reference processing and plotting;
their atmosphere calculation and warm-start path are never called.

H-delta changes most: the peak old/new difference is 0.02271 in normalized
flux for G29-38. H-delta RMS relative to the observations decreases for all
six targets. This is not a uniform improvement in every line: H-alpha and
H-beta have very small increases in RMS, while H-gamma results are mixed.
These locally normalized line profiles do not test absolute bolometric flux.
The original paper figure, atmosphere outputs, and regression fixtures were
left untouched, and no production equations or tolerances were changed for
this rebuild.

Public calculation and diagnostic artifacts are in
`results/figure2-microphysics-20260910/`; `summary.json`, per-model request
hashes/certificates, `reference-check.json`, and `comparison-to-original.json`
record the provenance and numerical comparisons. The standalone figure is
`../../output/pdf/figure2_corrected_cold_starts.pdf` relative to this repository.

Reproduce with a fresh output directory (the model driver refuses overwrites):

```sh
# From release/OpenWD, using its Python 3.11 environment:
PYTHONPATH=src PYTHONNOUSERSITE=1 OPENWD_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -u research/rebuild_paper_figure2.py --jobs 3 --output NEW_OUTPUT
```

The plotting-only driver is `release/OpenWD/research/plot_paper_figure2.py`,
run from the outer research root with that root's `PYTHONPATH=src`, its
historical Python 3.9 observation-reader environment, and `--output` pointing
to the public calculation directory. It refuses uncertified models and
checks that their imported package came from the public source tree. This
separation avoids importing unrelated legacy research modules in the public
model process. The observations and Koester reference grid must be available
locally; their exact input paths and hashes are in `reference-check.json`.

## Follow-up: paper Figure 1, absolute-flux DA grid

The original Figure 1 PDF is byte-identical to
`../../results/da-koester-current-from-gray-v1/current_da_vs_koester_uv_ir_optical.pdf`.
Its six parameter pairs were recalculated independently with the corrected
public API at production quality, with no supplied atmospheres or checkpoint
continuation. All six pass the equilibrium certificate and final synthesis
source-closure test; all use 100 structure depths. The full configuration
objects and saved wavelength nodes match those of the original figure.

| Teff (K) | log g | Iterations | Solve + synthesis (s) | Old blue discrepancy | New blue discrepancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5000 | 8 | 45 | 320.1 | 1.420% | 1.420% |
| 10000 | 8 | 33 | 307.1 | 1.039% | 1.631% |
| 20000 | 8 | 39 | 281.1 | 1.209% | 0.706% |
| 30000 | 8 | 39 | 250.8 | 0.607% | 0.397% |
| 20000 | 7 | 34 | 207.0 | 0.948% | 0.445% |
| 20000 | 9 | 46 | 255.4 | 1.570% | 1.016% |

The blue discrepancy is the wavelength integral of absolute OpenWD-minus-
Koester flux, divided by integrated Koester flux, over native samples in
3750-4500 A. It is not a fitted chi-square or a convergence residual. The
5000-K model is essentially unchanged; the blue agreement improves in the
four hot cases and modestly worsens at 10000 K. The largest all-depth flux
residual among the new atmospheres is 2.85e-6, largest local-energy residual
0.000473, and largest unrestricted temperature correction 5.25e-5.

The original six-by-two layout is retained, showing absolute surface
F_lambda on the Koester wavelength nodes. There is no local continuum
normalization, convolution, fitted scaling, or flux correction. The sampled
UV-IR integrals in the diagnostic JSON cover approximately 900-30000 A only;
they are not bolometric closure tests. The original paper and reference
artifacts are unchanged.

Artifacts and provenance are in `results/figure1-microphysics-20260910/`;
the standalone deliverable is `../../output/pdf/figure1_corrected_cold_starts.pdf`.
Reproduction uses `research/rebuild_paper_figure1.py --jobs 3 --output NEW_OUTPUT`
from this repository with the same public Python 3.11/single-threaded settings
as the Figure 2 rebuild. The plotting-only wrapper is
`release/OpenWD/research/plot_paper_figure1.py`, run from the outer root with
its historical Python 3.9 reader environment and `PYTHONPATH=src`.
It verifies all certificates, configurations, references, and wavelength
nodes before calling the historical plotting function, never its solver.

The earlier 12000-K comparison is not inconsistent with Figure 2. Applying
Figure 2's convolution and local normalization to those same old/corrected
12000-K spectra gives maximum normalized-flux changes of 0.000652 (H-alpha),
0.000525 (H-beta), 0.007488 (H-gamma), and 0.027227 (H-delta). Without local
normalization, peak differences are 4.27% of the old window peak at
3750-4000 A and 4.89% at 4000-4500 A. The original diagnostic includes the
higher Balmer series and between-line absolute continuum; Figure 2 omits
the higher members and separately normalizes each profile. These changes
are real, not eliminated by atmosphere reconvergence. Also, Figure 2 has
different parameter pairs and production resolution, whereas the earlier
12000-K regression control uses standard quality; the figures are not an
otherwise-identical test of depth resolution.

## Follow-up: low-cost, opt-in monotone formal synthesis

The 12000-K standard DA flux deficit is dominated by source-function
interpolation on the 40-depth atmosphere, not angular sampling. On a frozen
corrected cold state, increasing synthesis quadrature from 3 to 8 angles
changes the integrated flux ratio from 0.979663 to 0.979898. Monotone cubic
source interpolation, with its scattering source solved consistently,
instead gives 1.000159. No target flux enters the transfer calculation.
The existing trapezoidal opacity integration and boundary conditions are
retained. A separate power-law opacity-integration diagnostic was not adopted:
combining both changes overshot to about 1.0155 on this coarse state.

The implementation uses PCHIP harmonic source slopes and analytically
integrated cubic cell polynomials. The linear-source Lambda operator
preconditions source corrections. Slow corrections switch to the actual
piecewise PCHIP Jacobian, including analytic derivatives of the limited
slopes; there is no temperature or composition switch. This resolves the
initial prototype's slow convergence on artificial epsilon=0.001 and 1e-6
scattering tests. An independently re-evaluated source equation is required
to close below 1e-10. Failure raises, without changing transfer methods.
Wavelength chunks bound the dense matrix storage. Monotone higher-order
formal solutions are also discussed by
[de la Cruz Rodriguez & Piskunov (2013)](https://arxiv.org/abs/1212.2737);
this implementation specifically uses PCHIP rather than their Bezier scheme.

The public DA API now accepts `synthesis_transfer="formal-pchip"`.
`formal-linear` remains the default. The option is forwarded only to final
synthesis, not atmosphere relaxation, and is recorded in the spectrum
metadata. It does not modify DAConfig or the atmosphere's request identity.
Other public model defaults are untouched; the DB/DAB rows below are
fixed-state transfer diagnostics, not new public DB/DAB options.

| Control | Depths | Established flux / Fstar | Cubic flux / Fstar |
| --- | ---: | ---: | ---: |
| DA 12000, standard, new public cold start | 40 | 0.979663 | 1.000159 |
| DA 12000, production, new cold start | 100 | 0.996496 | 1.000235 |
| DA 5000, production | 100 | 0.996528 | 0.998802 |
| DA 10000, production | 100 | 0.994634 | 1.000382 |
| DA 20000, production | 100 | 1.000307 | 1.001057 |
| DA 30000, production | 100 | 0.999741 | 1.000424 |
| DB 22000, retained corrected cold structure | 80 | 0.996058 | 1.000594 |
| DAB 20000, retained corrected cold structure | 80 | 0.992574 | 0.999554 |

These are sampled 100-1000000-A integrals, with retained line-profile nodes,
not enforced normalization. In particular, the 20000-K DA's small existing
excess increases slightly: cubic interpolation does not guarantee exact
bolometric closure. The two helium-bearing structures predate the
provenance-only microphysics request-tag bump and generated mismatch
warnings when read for fixed-state synthesis. Their original corrected
cold-run certificates are retained; these controls do not claim a new
same-request public convergence qualification.

The completed public standard cold run took 180.6 s, with 46 reported
iterations. It passes all atmosphere checks, including all-depth flux
3.93e-9, local energy 3.85e-6, and unrestricted dlnT 2.08e-6.
Its temperature, pressure, mass, density and electron density match the
preceding corrected standard cold run to rtol=1e-10. Re-running established
synthesis on this state reproduces the pre-option corrected spectrum to
rtol=1e-8; no regression fixture was refreshed.

A three-repeat cached-material benchmark measured median transfer times of
0.681 s for established synthesis and 1.908 s for cubic synthesis, including
source closure and emergent flux. The extra 1.23 s is about 6% of the measured
20.6-s full synthesis, or less than 1% of this cold-start run. Concurrent
multi-object timings varied and should not be used as precise speed ratios.
No C rewrite or extra atmosphere iterations are needed for this improvement.

This is not merely an amplitude change. At 40 depths, maximum changes in
locally normalized H-alpha through H-delta are about 0.014-0.018; the
corresponding 20000-K production changes are below 0.00046. The numerical
and line-shape consequences are why the option is not silently made default.
The new visual comparison includes Koester, the established corrected
spectrum and the cubic spectrum, all on the same freshly cold-started
12000-K atmosphere. Only its H-alpha panel is locally normalized.

Verification: 55 focused transfer/source/API tests pass on both Python 3.9
and 3.11, plus 27 spectrum/model-safety/transfer-selection tests on 3.11.
Tests include independent SciPy quadrature of the PCHIP interpolant,
finite-difference checks of the analytic radiation tangent, strong scattering,
chunk-size independence, explicit failure on nonclosure, and default isolation.
The full historical spectral suite was not relabelled or its tolerances
relaxed to accommodate the preceding MHD correction.

Artifacts are in `results/da-flux-normalization-20260910/`. The final end-to-end
record is `public-cold-standard/summary.json`; its spectra and atmosphere
are preserved alongside the prior diagnostics. Reproduce the public cold
control with `research/validate_da_cubic_public.py` in a fresh output location;
the driver refuses existing output directories. The plotting-only entry point
is `research/plot_da_cubic_comparison.py`. No commit or push was made.

## User-approved DA default promotion and matched DZ comparison

After reviewing the preceding comparison, the user requested cubic synthesis
as the default. `compute_da` now defaults to `synthesis_transfer="formal-pchip"`;
the explicit `"formal-linear"` selection remains available. No atmosphere
equations, grids, convergence tolerances, opacities or input parameters are
changed by this promotion. Low-level routines and the other public model
defaults remain unchanged. Helium synthesis and `compute_dz` now additionally
accept explicit cubic interpolation for the requested polluted-star check.

The default (no synthesis keyword) was tested with another 12000-K, log-g=8
standard cold start. It reaches the same atmosphere in 46 iterations, with
all material arrays matching the preceding corrected cold state to rtol=1e-10.
The broad flux ratios are 0.979662939 for explicit linear and 1.000158579 for
the cubic default. The run took 190.5 s while a metal-rich model was running
on another core; this is not a controlled performance benchmark.
`public-cold-default/summary.json` records the certificate and provenance.

Additional fixed-state (not new equilibrium) DA checks give:

| Teff (K) | Corrected linear flux / Fstar | Corrected cubic flux / Fstar |
| --- | ---: | ---: |
| 3000 | 0.986005 | 0.999025 |
| 4000 | 0.991616 | 0.997373 |
| 20000 | 1.001297 | 1.002035 |

These particular integrals use the retained 900-300000-A regression grids,
not the finer full-coverage grids in the preceding table. Source closure
passes in every case; the warm model again illustrates that this is not an
enforced flux normalization.

New `tests/data/da_cubic_regressions` snapshots retain already checked,
corrected cold-state/cubic spectra for the public-default 12000/20000-K guard.
They do not replace any file in `spectral_regressions`. The same 0.1% spectrum
and normalized Balmer-profile limits are retained; the tests still omit the
synthesis keyword. `research/capture_da_cubic_controls.py` only selects
existing evaluated wavelengths and records their source hashes, without
recalculating a reference to match the test output. Historical spectral tests
remain separate and still expose the intentional earlier microphysics change.

Verification at promotion: Python 3.11 fast suite (`not canary and not spectral`)
771 passed, 7 skipped, 19 deselected; focused source/operator/API suite 86
passed. Python 3.9 focused suite 85 passed, followed by 74 source/default-DA/
high-precision-operator checks passed. Skips in the 3.11 fast suite are optional
profile/reference data and unavailable mpmath, not hidden numerical failures.

### Completed J0738 comparison

SDSS J0738+1835 had the largest fractional change among the earlier audited
helium-host DZ cases (mostly in the far UV, outside the available observation).
The corrected production calculation was started without an input atmosphere,
at the paper's fixed 13950 K, log-g=8.40 and abundances. All five convergence
checks pass after 35 iterations, the same count as the retained pre-fix cold
run. The all-depth flux residual is 4.02e-7, local energy residual 1.56e-6,
and unrestricted temperature correction 2.91e-6. The cubic synthesis source
equation independently closes to 9.52e-12. Repeating linear synthesis on this
new cold atmosphere leaves its material arrays exactly unchanged.

The old comparison uses source archived from GitHub commit
`f4dcfa5c5ac9b81ff9b1f2389046fbeea326036f` and its previously verified,
separately cold-started 80-layer atmosphere. Rebuilding its metal-electron
closure reproduces the stored material arrays; synthesis on the original
wavelength grid reproduces the archived flux to 4.10e-16 of its peak.
This is an old-code control, not a new-code synthesis of an old structure.
Both curves use 80 layers, not the lower-resolution original paper model.

For the comparison all versions are evaluated on the same 23004-node request,
including 100-1000000 A coverage and fine optical nodes. An existing capped-line
selection effect deserves separate follow-up: broadening the requested range
replaces 374 of the 20000 selected transitions with far-IR lines. Consequently,
the old-code wide-range output is not bitwise equal to its old narrow-range
output (maximum relative difference at original nodes 2.36%). The strict
original-grid recovery check is recorded separately. The plotted comparison
controls this effect by using identical requested wavelengths and line-selection
settings for old and new; it does not change the selection algorithm.

| Spectrum | Broad sampled flux / Fstar | Optical normalized RMS vs SDSS |
| --- | ---: | ---: |
| Old GitHub physics and cold atmosphere, linear | 0.974646 | 0.027881 |
| Corrected physics and new cold atmosphere, linear | 0.974520 | 0.028093 |
| Corrected physics and new cold atmosphere, cubic | 0.983080 | 0.028157 |

Cubic synthesis improves the bolometric mismatch but leaves a 1.69% deficit;
this is not a complete DZ flux-consistency fix. Optical agreement is essentially
preserved, not improved overall. The optical panels retain the paper's continuum
normalization and instrumental resolution; the SED panel has no bolometric
rescaling. No parameters or radial velocity were refitted.

The driver is `research/validate_dz_cubic_public.py`; old-source checks use
`research/resynthesize_old_dz_control.py`. Certificates, hashes, spectra,
line-selection details and metrics are in
`results/dz-cubic-microphysics-20260910/j0738/`. The figure is generated by
`research/plot_dz_cubic_comparison.py` and saved in the outer research workspace
as `output/pdf/j0738_corrected_cubic_comparison.{png,pdf}`. Both image and rendered
PDF were visually checked. Host suspend gaps preclude a controlled runtime
comparison for this DZ run. A final DA/source focused test rerun passed all
26 tests. The DZ default remains linear; cubic is explicit for this diagnostic.
No commit or push was made.

## Release promotion — 2026-09-11

The user approved checkpointing the reviewed microphysics corrections and
DA cubic default. Public changes include the undoubled critical field, complete
C-buffer cleanup, explicit CIA-tail regression coverage, and the independently
closed monotone cubic DA synthesis. The public CIA loader was already correct;
the He I collision-rate repair applies only to unpublished research code.
Unfinished DQ development is excluded from this checkpoint. The shared
atmosphere solver, numerical resolutions and convergence limits are unchanged.

`tests/data/approved_regressions` holds separately reviewed fixed-state outputs
and cold-model comparison outputs. The historical spectral/DAZ inputs remain
byte-for-byte unchanged and their hashes are tested. New capture scripts are
never run by pytest or CI. Every cold comparison structure is now from a
verified corrected cold start; reference reconstruction is not presented as
another equilibrium solve, and references are never passed into cold solvers.
The production 10000-K DB workflow's internal lower-boundary extension from
80 to 84 layers remains part of one cold calculation, not a neighboring-model
continuation.

The initial retained 5000-K DA and 10000-K DB reference pressures exposed small
deep-layer changes (0.00300% and 0.00994%) exceeding the unchanged 0.002%
reproducibility gate. Both new atmospheres independently passed all five
equilibrium checks. The 5000-K DA pressure/mass arrays also match the earlier
corrected Figure 1 calculation exactly; its temperature differs by at most
5.62e-8 fractionally. Corrected reference structures replace these stale
comparison inputs without changing any tolerance or solver behavior. The
first capture remains in the local research results for audit.

The public-default DA guard retains the 0.1% spectrum/Balmer-profile limits
and additionally requires the broad sampled 12000/20000-K integrals to be
within 0.3% of the stellar flux, with no rescaling. Other classes retain their
linear synthesis defaults. The tutorial notebook has been genuinely re-executed
from scratch with the corrected DA default and its saved plot visually checked;
its narrower native 900–30000-A interval integrates to 0.9980 of stellar flux.

Final release qualification passed on an isolated export of the staged public
tree, with the compiled acceleration backend present:

- Python 3.12: 661 unit/component tests, 234 cool-workflow component tests,
  all 12 fixed-spectrum controls, and all 16 fresh public cold-start cases.
- Python 3.9: the same 661 unit/component and 234 cool-workflow component
  tests, plus all 12 fixed-spectrum controls, passed independently.
- Five optional external-profile/reference tests were skipped on each Python
  version because their optional data were unavailable; no compiled-kernel
  checks were skipped. Spectral/cold cases excluded from the fast tier were
  run separately as listed above.
- The final full runner reports both `full_qualification: true` and
  `inputs_unchanged_during_run: true`. All 1,733 staged numerical, research and
  test files matched the isolated candidate byte for byte. No numerical
  tolerance was relaxed and no external atmosphere was used as a cold seed.

The cold matrix includes DA 3000/4000/5000/20000 K; DB 10000 K and both
22000-K resolutions; dense-He DB 5000/8000 K; DAB 20000 K and molecular
DAB 7500/9000/10000 K; and the three DAZ objects G29-38, G149-28 and
GALEX J1931. These passes retain the documented declared-equation scopes;
they do not certify every missing-physics or depth-grid question. GALEX J1931,
the final case, reached local-energy completion at iteration 107, with
local-energy residual 8.50e-4, all-depth flux residual 7.86e-6 and maximum
log-temperature correction 1.64e-4.

Local logs, model outputs, the executed notebook's model and reference-capture
evidence are preserved under `results/release-qualification-20260911/` (not
distributed as package data). The full-run numerical identity is
`7704451b3640f5e67337451fda59bbe256289e20cadb4087112eafa59318d44b`.
Temporary QA-export Git IDs in those raw reports are not public repository
commits; file hashes and the staged-tree comparison identify the tested inputs.
