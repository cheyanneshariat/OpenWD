# Hot DA trace-metal development

This branch adds **experimental fixed-atmosphere C/Si NLTE line formation**.
It does not yet establish a reliable abundance fit at 50,000 K. The ordinary
DAZ and DO/DAO presets retain their existing physics and interfaces.

## First observational benchmark: G191-B2B

Use [Preval et al. 2013, MNRAS 436, 659](https://doi.org/10.1093/mnras/stt1604)
and its [MAST STIS/FUSE atlas](https://archive.stsci.edu/prepds/wd-linelist/).
Their STIS E140H coadd covers the C III 1175 multiplet, Ly alpha and the
Si IV 1393/1402 doublet. It is a useful test near the requested temperature,
even though the instrument is STIS rather than COS.

The published abundance analysis fixes Teff=52,500 K, log g=7.53 and
He/H=1e-5. Table 10 gives C/H=(1.72 +/- 0.02)e-7 from C III and
Si/H=3.68(-0.14,+0.13)e-7 from Si IV (number ratios, quoted statistical 1-sigma
errors). These are elemental abundances inferred using those ions, not
ionic fractions. Separate C IV and Si III measurements are retained in the
machine-readable benchmark. Do not combine these numbers with the hotter
parameters from a different G191-B2B analysis.

The paper's background includes other metals. Fe/Ni blends overlap several
C III components, and Si IV contains a non-photospheric component near
+8 to +9 km/s, separated from the photosphere at +23.8 km/s. A C/Si-only
background cannot reproduce the complete line forest. Ly alpha also needs
interstellar absorption and airglow handling. These are reasons to stage
validation, not reasons to retune the atomic data against the observation.

```bash
python research/hot_daz_benchmark.py --fetch
```

This checksum-verifies two MAST files and writes `benchmark.json`, an observed
four-panel plot, and `observed_windows.npz` under `results/hot-daz/g191-b2b/`.
The JSON records the exact URLs, SHA-256 values, parameter convention and
line catalogue entries including competing blend identifications. Raw files
and derived results are ignored by Git. Initial Si IV/core masks are explicitly
candidate regions, not certified fitting masks. Observed wavelengths stay in
the observer frame and observed fluxes keep their physical units.

## First implementation

`wd_spectra.hot_trace_metals.synthesize_dao_trace_metals` takes a matched
`compute_dao` result. It checks that the H population state belongs to the
supplied atmosphere and normally requires a converged host. The lower-level
`solve_hot_trace_metals` accepts a callable supplying the host transfer
coefficients on requested wavelength grids.

```python
import numpy as np
from wd_spectra import DAOConfig, compute_dao
from wd_spectra.hot_trace_metals import synthesize_dao_trace_metals

wave = np.arange(1150., 1450., 0.01)  # vacuum Angstrom, stellar rest frame
host = compute_dao(DAOConfig(52500., 7.53, log_hydrogen_to_helium=5.), wave)
trace = synthesize_dao_trace_metals(
    host, {'C': np.log10(1.72e-7), 'Si': np.log10(3.68e-7)}, wave,
)
flux = trace.spectrum.surface_flux_lambda
```

The host solve can take hours. The reproducible runner saves the host before
attempting trace synthesis and raises on failed convergence by default:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python research/run_hot_trace_diagnostic.py
```

For a small numerical integration check only:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python research/run_hot_trace_diagnostic.py \
  --smoke --output results/hot-daz/smoke
```

The smoke option uses eight gray atmosphere depths and Planck-initialized
H/He populations. Even if its metal populations converge, its spectrum is
**not a converged stellar atmosphere or a validation fit**.

For research restarts, `run_hot_trace_from_checkpoint.py` accepts a tagged
numerical checkpoint saved by the runner, verifies its LTE reference against
the declared atmosphere/atom, and solves the full coupled H/He equations
again. No certificate is inherited. The optional population initializer
changes only the starting point, not the final equations or tolerances:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python research/run_hot_trace_from_checkpoint.py \
  results/hot-daz/g191-b2b-model/host-stage-0.000-final.npz \
  --equilibrate-populations --output results/hot-daz/g191-b2b-seeded
```

The numerical checkpoint format does not contain H/He composition; this
G191-B2B-specific runner declares He/H=1e-5 and rejects a different rebuilt
LTE reference. It records the input checkpoint hash. This research route does
not change the public DO/DAO cold-start contract.

## Comparing the prediction

```bash
PYTHONPATH=src python research/compare_hot_daz_benchmark.py \
  --model results/hot-daz/g191-b2b-model
```

The comparator verifies the published parameter/abundance request and rejects
unconverged or smoke predictions by default. `--allow-unconverged` produces
an explicitly labelled diagnostic only. It saves `comparison.png`, a PDF,
`comparison.json`, `comparison_arrays.npz` and `comparison.fits` with one table
per spectral window. The FITS tables contain observed-frame vacuum wavelength,
locally normalized data/error/model, the photospheric-only model, and the
boolean score mask. The JSON includes the model and observation checksums.

The comparison shifts by +23.8 km/s, convolves at constant R=144,000, and
integrates over the observed bins. This Gaussian is an approximation to the
STIS response, not a reconstruction of the individual exposure LSFs. Both
model and data are divided by the same fitted local continuum. Only two
continuum nuisance coefficients per region are optimized; temperature,
gravity, abundances, velocity and resolution remain fixed.

For Ly alpha, a separate fixed diagnostic H I screen uses log N(H I)=18.18
from [Lemoine et al. 2002](https://arxiv.org/abs/astro-ph/0112180), with assumed
b=10 km/s and v=+19.4 km/s. It is applied before instrumental convolution.
The 1214--1217.5-A core is excluded from the score. The single screen does not
replace a multi-component ISM fit.

Catalogue contaminants/competing identifications are marked and excluded
within +/-6 km/s; non-photospheric Si IV is excluded within +/-10 km/s.
These approximate masks cannot guarantee freedom from blended wings. Whole
window equivalent widths explicitly include those blends. Separate fixed
+/-20-km/s apertures measure the C III 1175.987 and 1176.370 components, which
have no competing Fe/Ni identification in the atlas. Reported chi-square
per pixel is conditional on the continuum and ignores pixel covariance and
systematics; it is not a formal goodness-of-fit probability.

`prediction.fits` contains the intrinsic stellar-rest-frame surface F_lambda
and its H/He-only background, without ISM absorption or instrumental smoothing.
Numerical model products also retain the local code/data content identities.
Use the unchanged converged host to test sensitivity or produce wider coverage:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python research/resynthesize_hot_trace.py \
  results/hot-daz/g191-b2b-seeded --refine-radiation-grid --full-uv \
  --tolerance 0.0001 --output results/hot-daz/g191-b2b-refined
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python research/resynthesize_hot_trace.py \
  results/hot-daz/g191-b2b-seeded --expanded-atoms \
  --tolerance 0.0001 --output results/hot-daz/g191-b2b-expanded
```

The refined grid includes H/He edges and lines as well as the metal grid and
inserts midpoints throughout. The expanded atoms retain C=(40,54,1) and
Si=(50,40,1) levels. These are sensitivity checks, not replacements for
audited photoionization/collision data or a metal-blanketed host. Full-UV
output adds 1150--1700 A every 0.01 A and retains the finer diagnostic grids.
The reader rejects changed numerical code/tables, a changed rebuilt EOS, or
an unconverged saved atmosphere.

## Physics and numerical scope

The prototype:

- Holds T, rho, ne and host H/He populations fixed. A separate Saha reference
  uses exactly that ne and abundances relative to H nuclei; it never calls a
  metal EOS that would silently reclose the host electron density.
- Solves explicit fine-structure C III/IV/V and Si III/IV/V atoms using the
  existing multilevel statistical-equilibrium kernel. Initial level counts
  are C=(20,30,1), Si=(30,23,1); these are provisional Stout level selections,
  not replicas of the paper's TLUSTY term atoms.
- Includes metal lines and selected bound-free opacity/emissivity in the
  radiation field that feeds the metal rate equations. The final formal
  solution combines these with the same host, including its Lyman lines.
- Uses Verner ground continua and Kramers excited continua consistently in
  rates and transfer, with the existing approximate electron collision rates.
  There is no new fitted oscillator-strength or line-width correction.
- Includes only closed explicit-atom line transitions. Omitted populations
  remain inert LTE reservoirs. Reports represented population fractions and
  particle-conservation error, rather than claiming atom completeness.
- Uses positivity-preserving population acceleration and reports the
  undamped fixed-point defect; a small damped step is not convergence.
- Returns exactly the host spectrum for an empty abundance dictionary, and
  rejects mixtures above conservative fixed-background mass/electron limits.

Metal free-free, thermal feedback, host-population feedback, metal charge
feedback, complete collision/photoionization atoms, abundance stratification,
rotation, ISM absorption and instrument convolution are not part of this first
interface. These limits are also recorded in result metadata. Convergence
certifies the fixed-host population iteration only. Optical-depth/wavelength
refinement and comparison with TLUSTY/TMAP remain necessary.

## Validation sequence

1. Numerical checks: LTE detailed balance, particle conservation, unchanged
   electron reference, zero-metal recovery, continuum Kirchhoff law and honest
   nonconvergence reporting. Regression-test the existing hot adapters.
2. A converged H/He background at the paper's fixed parameters, followed by
   metal population/transfer convergence. Refine depths and radiation sampling.
3. Replace the provisional atom selections with audited C and Si model atoms,
   checking metastable C III levels, dielectronic recombination, excited-state
   photoionization, collision-only links, higher ion stages and line broadening.
4. Compare the fixed published abundances to the STIS line windows after
   applying +23.8 km/s and the STIS response. Model or mask non-photospheric
   components and identify Fe/Ni blends explicitly. Do not optimize abundances
   first and use the resulting fit as proof that the physics is correct.
5. Quantify other-metal blanketing and the error of the fixed-background
   approximation against a metal-blanketed NLTE atmosphere. Only then attempt
   abundance recovery, followed by the user's COS spectrum with its actual LSF.

The benchmark-preparation script intentionally produces no chi-square or
claim of an observed abundance match.

## Initial numerical evidence (2026-10-02)

The [recorded smoke check](history/hot-daz-trace-smoke.json) converged in 18
metal iterations on 12,767 radiation wavelengths, with an undamped population
defect of 5.86e-4, particle-conservation error 2.17e-16 and source-closure
residual 7.97e-16. It produces C III and both Si IV absorption features.
Thirteen new trace/benchmark tests and 68 existing hot-adapter/metal tests
passed, including the compiled backend checks. Acceleration and ordinary
iteration agree in an independent small-atom test.

Across **all** smoke-atmosphere depths, the explicit atoms represent at least
0.319 of the LTE carbon population and 0.658 of silicon. This records a
substantial omitted reservoir somewhere in the depth grid; it is not an
estimate of the emergent-line error. Atom/ion-stage expansion and
line-formation-depth checks are still needed. None of the smoke residuals
constitutes a comparison with the observed star.

## Converged G191-B2B comparison (2026-10-02)

The [recorded comparison](history/hot-daz-g191-b2b-comparison.json) uses a
40-depth H/He background, 32 He II levels, 8 H levels and 3 angular nodes.
A full-NLTE restart from the saved fresh LTE initialization converged after
13 coupled iterations, following fixed-temperature population initialization.
Its independently measured maximum log-temperature correction is 1.13e-4;
the H/He population defect is 2.44e-5, local-energy residual 2.29e-5 and
all-depth flux residual 5.30e-12. Its complete structure-grid certificate
passes. This certifies the declared restricted H/He equations, not a
self-consistent metal-blanketed atmosphere.

The expanded C=(40,54,1), Si=(50,40,1) calculation on 137,402 radiation
wavelengths converged in 22 iterations, with population defect 8.17e-5.
It supplies a 65,149-point intrinsic spectrum spanning 1150--1700 A. All
abundances and stellar parameters remain fixed at the published values.

| Isolated C III component | STIS aperture EW (mA) | Compact/refined model | Expanded/refined model |
| --- | ---: | ---: | ---: |
| 1175.987 A | 17.00 | 3.73 | 7.75 |
| 1176.370 A | 20.04 | 4.57 | 9.34 |

These fixed +/-20-km/s apertures avoid the competing Fe/Ni identifications.
The quoted observational statistical errors are 0.70 and 0.61 mA, excluding
continuum and covariance uncertainty. Expanding the atom doubles the predicted
C III strengths but leaves them factors 2.1--2.2 too weak. **Atom-size
convergence is not established, and abundance fitting is not qualified.**
Grid refinement changes these equivalent widths by less than 1%; it does not
explain the discrepancy.

The expanded/refined Si IV comparison has conditional normalized-flux RMS
3.20% at 1393 A and 2.68% at 1402 A. It is substantially closer than C III,
but retains systematic profile residuals. Ly alpha plus the fixed diagnostic
H I screen is too deep in the wings, with conditional RMS 3.78%. The masks,
continuum conventions and Gaussian instrumental approximation above apply;
these RMS values are not formal fit probabilities. Changing R from 144,000
to 114,000 changes the predicted normalized profiles by at most 0.003 in
these windows and does not remove the discrepancies.

The small minimum represented Si population fraction occurs at the deepest
boundary (m=197 g/cm2, T=273,000 K). Near the H/He UV tau_lambda=1 locations,
the retained LTE fractions are approximately 99.6--99.8% for C and 99.9% for
Si. This depth check is an orientation aid, not a full metal-line contribution
function or proof of atomic completeness.

Reproduce the final spectrum and comparison from the saved converged host:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python research/resynthesize_hot_trace.py \
  results/hot-daz/g191-b2b-seeded --expanded-atoms --refine-radiation-grid \
  --full-uv --tolerance 0.0001 --output results/hot-daz/g191-b2b-expanded-refined
PYTHONPATH=src python research/compare_hot_daz_benchmark.py \
  --model results/hot-daz/g191-b2b-expanded-refined \
  --output results/hot-daz/g191-b2b-comparison
```

The final FITS files pass checksum verification, and the intrinsic spectrum
round-trips exactly against the numerical arrays. All 88 targeted
trace/benchmark/checkpoint/comparison and existing hot/metal regression tests
pass. The remaining priorities are model-atom convergence, audited C III
metastable/cascade and photoionization rates, and comparison with an atmosphere
including the other published metals, especially Fe/Ni blanketing. The
present comparison cannot uniquely assign the residuals to one of those
approximations.

## Radiative levitation development (2026-10-02)

`research/hot_trace_levitation.py` now evaluates the **element-specific photon
force** using the converged metal populations and the combined H/He/C/Si
radiation field. This is distinct from the bulk radiative-pressure term in
the PG1159 hydrostatic solver. At fixed structure and fixed abundances,
computing this force does not change the emergent spectrum. A levitation
spectrum requires solving for abundances as a function of depth, followed by
new statistical-equilibrium and transfer solutions.

The diagnostic uses

\[
g_{\mathrm{rad},Z}(m)=\frac{1}{cX_Z(m)}
  \int\kappa_{\lambda,Z}(m) F_\lambda(m)\,d\lambda,
\]

where opacity is per total atmospheric mass, `X_Z` is the element mass
fraction, and `F_lambda` is the centered local flux, already equal to `4 pi H`.
The Angstrom spectral-density units cancel the Angstrom integration measure.
The calculation retains signed flux and stimulated emission; it reports
bound-bound and bound-free forces separately. All bound-free photon momentum
is assigned to the parent element, with **no photoelectron momentum
redistribution**. Free-free metal momentum and omitted levels/ion stages are
absent. Force integrals include wavelengths outside the HST band.

`assess_hot_trace_levitation.py` compares wavelength quadrature and atom
selections, computes monochromatic `tau_lambda=1` locations, and records a
restricted zero-drift diagnostic. For an ideal trace element with mean charge
`q`, the required local abundance slope is

\[
\frac{d\ln(n_Z/n_H)}{d\ln m}=
\frac{A_Zm_u m(g-g_{\mathrm{rad},Z})}{\rho kT}
-q\frac{d\ln(n_eT)}{d\ln m}
-\frac{d\ln(n_HT)}{d\ln m}.
\]

This includes the electric field inferred from ideal electron-pressure
balance. It neglects thermal diffusion, electron radiative force/inertia,
ion-specific mobility and momentum redistribution, accretion, and winds.
For fully ionized isothermal hydrogen it recovers
`2 A (1 - g_rad/g) - (q+1)` (taking `m_H=m_u`). A zero abundance gradient
therefore does **not** require exactly `g_rad=g`. This diagnostic is not
integrated into a profile: the radiation force and ion populations must be
recomputed as the abundances change. Neither a diffusion equilibrium nor a
levitation-modified spectrum is claimed.

The zero-gradient pure-H limit agrees with the effective-gravity expression
in [Schuh, Dreizler & Wolff (2002), equation 3](https://arxiv.org/abs/astro-ph/0111245).
Their local equilibrium ansatz neglects concentration and thermal diffusion;
it should not be confused with a solution retaining a concentration gradient
and a specified material reservoir or boundary flux.

The first force test used C=(40,54,1), Si=(50,40,1) and the existing converged
homogeneous spectrum. Increasing force-quadrature sampling from 137,402 to
274,780 wavelengths changed the force by at most 0.43% (C) and 0.69% (Si)
over `1e-4 <= m <= 1 g/cm2`, at fixed populations. Relative to the smaller
atoms, forces changed by as much as 70% and 45%, respectively. This is
wavelength-quadrature convergence, **not atomic or depth convergence**.

In that initial test the C III 1175.987/1176.370 tau=1 locations were near
`m=0.049–0.052 g/cm2`, with `g_rad,C/g=4.73–4.74` (bound-bound alone about
3.78). The Si IV core locations were near `m=0.0018–0.0025 g/cm2`, with
`g_rad,Si/g=0.28–0.35`. These are depth indicators, not contribution
functions. The bulk acceleration reached only 1.15% of gravity.

**Ion-stage completeness matters:** the mean silicon charge at those Si IV
core locations was approximately 3.96–3.97. The abundant Si V stage was
only a one-level ionization sink, so its missing lines could bias the force.
The `--higher-ions` synthesis option explicitly adds C V and Si V levels,
with C VI/Si VI as the new terminal sinks: C=(40,54,10,1),
Si=(50,40,30,1). It reruns NLTE populations before testing the new spectrum
and radiative forces; it does not append LTE opacity to an unchanged NLTE
state. These selections remain provisional.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python research/resynthesize_hot_trace.py \
  results/hot-daz/g191-b2b-seeded --higher-ions --refine-radiation-grid \
  --full-uv --tolerance 0.0001 --output results/hot-daz/g191-b2b-higher-ions
PYTHONPATH=src:research OPENBLAS_NUM_THREADS=1 python research/hot_trace_levitation.py \
  results/hot-daz/g191-b2b-expanded-refined \
  --output results/hot-daz/g191-b2b-levitation-force
PYTHONPATH=src:research OPENBLAS_NUM_THREADS=1 python research/hot_trace_levitation.py \
  results/hot-daz/g191-b2b-expanded-refined --refinement 2 \
  --output results/hot-daz/g191-b2b-levitation-force-finer
PYTHONPATH=src:research OPENBLAS_NUM_THREADS=1 python research/hot_trace_levitation.py \
  results/hot-daz/g191-b2b-refined \
  --output results/hot-daz/g191-b2b-levitation-force-compact
PYTHONPATH=src:research python research/assess_hot_trace_levitation.py \
  --baseline results/hot-daz/g191-b2b-levitation-force \
  --finer results/hot-daz/g191-b2b-levitation-force-finer \
  --compact results/hot-daz/g191-b2b-levitation-force-compact \
  --output results/hot-daz/g191-b2b-levitation-assessment
```

The new tests check momentum units, thin-limit abundance independence,
additivity of element forces, signed flux/opacity, the pure-H electric-field
limit, and retention of the omitted LTE charge reservoir. All 98 targeted
tests pass, including detailed-balance checks with the higher ion stages. These checks establish numerical identities, not a validated
diffusion calculation.

The literature gives a reason to test levitation without assuming it will
fix the observed residuals. [Rauch et al. (2013), sections 4.3–4.4](
https://doi.org/10.1051/0004-6361/201322336) compared homogeneous models to
NGRT diffusion calculations for G191-B2B; the stratified model did not
improve the UV metal-line fit, and they discussed missing transport physics
such as a weak wind. Their 60,000 K/log g=7.60 analysis is separate from the
52,500 K/log g=7.53 Preval benchmark used here. Completing the force-carrying
atoms and metal blanketing, specifying transport boundaries, and iterating
diffusion with the radiation field are necessary before calling a changed
spectrum a prediction including radiative levitation.

### Higher-ion result and updated observation comparison

The [recorded levitation assessment](history/hot-daz-g191-b2b-levitation.json)
includes the completed four-stage calculation. With 105 C and 121 Si levels,
the metal solution converged in 90 iterations on 145,486 radiation
wavelengths (population defect `8.48e-5`). A separate 26,003-point population
grid converged in 108 iterations; its clean C III equivalent widths differ
by about 0.5%. The force integral was independently repeated on 290,948
wavelengths, changing C/Si acceleration by at most 0.42%/0.66% at the tested
depth nodes between `1e-4` and `1 g/cm2`.

| UV diagnostic | Core tau=1 mass (g/cm2) | g_rad / g | Uniform-abundance support / g |
| --- | ---: | ---: | ---: |
| C III 1175.987 | 0.0523 | 4.733 | 0.818 |
| C III 1176.370 | 0.0488 | 4.726 | 0.818 |
| Si IV 1393.755 | 0.00190 | 0.292 | 0.910 |
| Si IV 1402.770 | 0.00262 | 0.364 | 0.910 |

The last column includes the pressure/electric-field terms in the restricted
diagnostic above. At these depths the photon force is sufficient to reverse
the sign of the required carbon abundance gradient. For silicon it reduces
the required gradient, while remaining below the support value for a uniform
abundance. This is a statement about this model at these abundances; the
force changes with abundance, line saturation, ionization, and blanketing.
It does not rule out levitation-supported silicon at another abundance.

The ion-resolved momentum sum closes to `2e-11` relative tolerance. C IV
supplies most of the carbon force; the C V contribution at the C III depth
indicators is only about `3e-8 g`. Si V supplies about `0.012 g` near the
Si IV cores. Higher ions therefore do not remove the main force result,
although they affect deeper silicon layers more strongly. The large
sensitivity to the initial lower-ion atom expansion remains; atomic
completeness has not been established. Only about 11–12% of the carbon force
at the C III depth indicators comes from the 1150–1700-A band.

The new **uniform-abundance** spectrum still predicts clean C III equivalent
widths of 7.754 and 9.345 mA, versus observed 17.005 +/- 0.697 and
20.042 +/- 0.608 mA (statistical aperture errors). Adding the higher ions
changes those predictions by only 0.06%. Conditional normalized RMS values
are C III 5.46%, Ly alpha 3.78%, Si IV 1393 3.25%, and Si IV 1402 2.74%.
The new FITS prediction and comparison pass checksum and numerical
round-trip checks. The remaining carbon mismatch is not resolved by this
ion-stage extension, and no levitation-modified abundance profile or
spectrum has yet been computed.

```bash
PYTHONPATH=src:research python research/compare_hot_daz_benchmark.py \
  --model results/hot-daz/g191-b2b-higher-ions \
  --output results/hot-daz/g191-b2b-higher-ions-comparison
PYTHONPATH=src:research OPENBLAS_NUM_THREADS=1 python research/hot_trace_levitation.py \
  results/hot-daz/g191-b2b-higher-ions --ion-breakdown \
  --output results/hot-daz/g191-b2b-higher-ions-force
PYTHONPATH=src:research OPENBLAS_NUM_THREADS=1 python research/hot_trace_levitation.py \
  results/hot-daz/g191-b2b-higher-ions --refinement 2 \
  --output results/hot-daz/g191-b2b-higher-ions-force-finer
PYTHONPATH=src:research python research/assess_hot_trace_levitation.py \
  --baseline results/hot-daz/g191-b2b-higher-ions-force \
  --finer results/hot-daz/g191-b2b-higher-ions-force-finer \
  --compact results/hot-daz/g191-b2b-levitation-force \
  --output results/hot-daz/g191-b2b-higher-ions-levitation-assessment
```

### Short physical screens with intermediate spectra

The next experiments use warm population guesses and save the **evaluated**
spectra at iterations 1, 4, 8, and 12. They keep the published parameters and
C/Si abundances above fixed. A spectrum becoming stable is enough to decide
which experiment to pursue; it does not grant population convergence. The
[complete numerical record](history/hot-daz-g191-b2b-exploration.json) contains
the population defects, intermediate measurements, atomic-data hashes, and
FITS round-trip checks. Results are under `results/hot-daz/exploration-summary`.

| Experiment (C III / C IV fine-structure levels) | EW 1175.987 (mA) | EW 1176.370 (mA) | C III normalized RMS | Population defect |
| --- | ---: | ---: | ---: | ---: |
| STIS data | 17.005 +/- 0.697 | 20.042 +/- 0.608 | — | — |
| Prior converged baseline, 40 / 54 | 7.754 | 9.345 | 5.46% | 8.48e-5 |
| Short unchanged-physics control, 40 / 54 | 7.717 | 9.300 | 5.48% | 1.53e-2 |
| OP photoionization tables, 40 / 54 | 6.291 | 7.620 | 6.16% | 3.61e-3 |
| C III enlarged alone, 80 / 54 | 10.930 | 13.022 | 4.13% | 7.32e-1 |
| Both enlarged, 80 / 100 | 13.304 | 15.746 | 3.38% | 1.00e-3 |
| OP tables, 80 / 100 | 9.858 | 11.776 | 4.56% | 1.93e-3 |
| 80 / 100 plus LTE Fe ground continuum | 13.580 | 16.064 | 3.31% | 5.19e-3 |
| OP tables, 120 / 150 | 12.561 | 14.884 | 3.60% | 3.59e-3 |

All new rows are **unconverged** at the requested `1e-4` tolerance. The 80/100
approximate-cross-section result includes a second batch of 12 iterations;
the clean equivalent widths changed by only 0.05–0.06% in that batch. The
120/150 OP calculation changed those widths by 0.08–0.10% between its eighth
and twelfth iterations. This is useful for screening without waiting for
every high-level population to settle. It does not establish atom-size,
depth-grid, or radiation-grid convergence of the new models. The very large
defect in the C-III-only run especially prevents interpreting it as a final
solution. Enlarged atoms retain C V/VI=(10,1); silicon stays at
Si III/IV/V/VI=(50,40,30,1).

The strongest lead is completeness of the C III **and** C IV atoms and their
coupled rate equations. The 120/150 OP model reaches 73.9/74.3% of the two
observed equivalent widths, compared with 45.6/46.6% for the baseline.
Ly alpha and Si IV are largely unchanged. The atom is still incomplete;
arbitrarily stopping the level expansion when a line matches would not be
a physical validation. Likewise, the stronger result with approximate
photoionization should not be preferred solely because it fits better.

The optional carbon tables are the existing public TLUSTY/Opacity Project
[`c3.dat`](https://tlusty.oca.eu/tlusty/Tlusty2002/database/atom/c3.dat) and
[`c4_35+2lev.dat`](https://tlusty.oca.eu/tlusty/Tlusty2002/database/atom/c4_35+2lev.dat).
They enter photoionization/recombination rates and continuum
opacity/emissivity together, as well as existing table-dependent collision
normalizations. Matching uses threshold energies and term spin/orbital
labels. In the 120/150 selection, 59 C III and 89 C IV levels match tables;
the rest retain Verner/Kramers fallbacks. In particular, all three
`2s.2p 3Po` fine-structure levels match the C III metastable OP term.
Thus this is **partial OP coverage**, not a complete new atomic dataset.
New tests using the real tables recover Planck-field LTE populations and
verify Kirchhoff's law. Fine-structure counts are not comparable directly
to the term/superlevel counts in published TLUSTY atmospheres.

The Fe experiment uses the elemental Fe/H=5.00e-6 inferred from Fe V in
[Preval et al. (2013), Table 10](https://doi.org/10.1093/mnras/stt1604).
Only LTE ground-state bound-free absorption and its LTE emissivity are
added to the unchanged H/He background. The roughly 2% increase in the
clean C III widths is **not a test of full metal line blanketing**: no
Fe/Ni line forest, metal thermal feedback, or electron-density feedback
is included. A proposed Fe/Ni version stopped before synthesis because
the cached Ni IV+ ionization ladder and Ni Verner cross-sections are
missing. Nickel was not replaced with invented data. Complete blanketing
remains a priority alongside higher-level rates and collisions.

`solve_hot_trace_metals` now accepts `initial_populations`, per-element
`photoionization_threshold_data`, and
`state_callback(iteration, defect, states, synthesize)`. Warm starts map
shared departure coefficients to the new atom and conserve its represented
particle reservoir. They carry no convergence certificate. The callback
can synthesize its current evaluated state and return `True` to stop;
the undamped defect still determines the reported convergence status.

The frozen-host reader remains strict by default. The exploration runner
explicitly permits changes confined to `hot_trace_metals.py` only when a
manifest recorded under the original valid code identity proves every
other package file, compiled transfer library, and physical-data identity
unchanged. It also rebuilds/checks the host EOS as before. The original
host certificate and identity are never rewritten. The older resynthesis
CLI exposes this exception as `--allow-trace-only-changes`; a changed
H/He solver or physical table still requires a new host solve.

Example reproductions (each output directory must be new):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=src:research MPLBACKEND=Agg MPLCONFIGDIR=/tmp/openwd-mpl-cache
python research/explore_hot_trace.py --case large-c --iterations 12 \
  --output results/hot-daz/explore-large-c
python research/explore_hot_trace.py --case large-c-op --iterations 12 \
  --seed results/hot-daz/explore-large-c --output results/hot-daz/explore-large-c-op
python research/explore_hot_trace.py --case op --carbon-levels 120 150 10 1 \
  --seed results/hot-daz/explore-large-c-op --iterations 12 \
  --output results/hot-daz/explore-larger-c-op
python research/explore_hot_trace.py --case fe-continuum --iterations 12 \
  --seed results/hot-daz/explore-large-c --output results/hot-daz/explore-large-c-fe
```

The summary figure is reproduced by `research/summarize_hot_trace_exploration.py`
once the cases listed in that script are present. Each run also contains
its own four-window comparison, intrinsic prediction FITS, and saved
population arrays. All 101 targeted tests pass, including the new warm-start,
intermediate-spectrum, OP detailed-balance, and host-identity checks.
No diffusion equilibrium or radiative-levitation abundance profile is
imposed in these experiments; improved atomic populations are needed
before interpreting a more elaborate transport calculation.

### Carbon collision data and a larger atom ladder

The subsequent screens retain the OP tables and replace selected approximate
electron-excitation rates with the public CHIANTI C III and C IV SCUPS data.
The [C III database record](https://db.chiantidatabase.org/c/c_3.html) attributes
these calculations to Fernandez-Menchero, Del Zanna & Badnell (2014,
[A&A 566, A104](https://doi.org/10.1051/0004-6361/201423864)); the
[C IV record](https://db.chiantidatabase.org/c/c_4.html) attributes its collision
data to Liang & Badnell (2011,
[A&A 528, A69](https://doi.org/10.1051/0004-6361/201016417)). These are calculated,
Maxwellian-averaged effective collision strengths, not measured or fitted
stellar abundances. C III data are restricted by CHIANTI to bound levels
through n=5, so higher Rydberg levels still lack this coverage.

`research/hot_trace_collisions.py` maps source ELVLC levels to Stout using
configuration, spin multiplicity, orbital term, statistical weight (J),
and energy. The energy tolerance is `max(5, 2e-4 |E|) cm^-1`. It rejects
ambiguous/repeated mappings and records every accepted and rejected match.
**Index equality is not assumed:** for example CHIANTI C III level 21 maps
to Stout level 24. A source C III 2s 5f singlet fails the energy check and
is deliberately left on the existing approximation. We map 74 of 75 C III
source levels and use 2,701 C III collision pairs. For C IV, 61 source levels
map, but only 66 source collision pairs have both endpoints in the selected
bound-level model. Autoionizing source levels are not inserted as ordinary
bound levels. In total, 2,767 pairs are supplied, including 2,210 that have
no radiative partner in the selected network.

The adapter evaluates the existing Burgess–Tully spline implementation and
caches exact values at the fixed depth temperatures. It does not interpolate
onto a new temperature grid. Excitation/de-excitation use the model energy
gap and statistical weights to maintain detailed balance. A direct Planck
test with the real C III/C IV tables and the 180/200 atom recovers LTE to
`1.37e-14` maximum relative error. Unprovided radiatively connected pairs
keep the approximate rates; unprovided collision-only links remain absent.

The collision audit exposed a separate rate-counting error in the shared
light-metal solver: a pair with several radiative multipoles received the
electron rate once per multipole. The fix keeps every radiative contribution
and inserts the electron rate once per level pair. For the default
approximation, an E1 channel takes precedence when one exists. Regression
tests split one radiative rate across M1/E2 channels and require identical
non-LTE populations, both with approximate and explicitly supplied collisions.
The corrected 180/200 control changes the clean equivalent widths by only
0.08/0.11%, so this fix does not explain the larger atomic-data improvement.

| OP model / collision treatment | EW 1175.987 (mA) | EW 1176.370 (mA) | C III RMS |
| --- | ---: | ---: | ---: |
| Previous 120/150, approximate collisions | 12.561 | 14.884 | 3.60% |
| 180/200, corrected approximate collisions | 14.081 | 16.622 | 3.22% |
| 180/200, CHIANTI only among C III levels 1–8 | 14.275 | 16.797 | 3.18% |
| 120/150, all available mapped CHIANTI pairs | 13.126 | 15.484 | 3.44% |
| 180/200, all available mapped CHIANTI pairs | 14.848 | 17.431 | 3.09% |
| 240/243, all available mapped CHIANTI pairs | 14.940 | 17.536 | 3.07% |

The independent short run with a doubled radiation grid (63,511 to 127,021
points) gives 14.846/17.430 mA for the 180/200 CHIANTI atom, changes of
0.015/0.006%. Its population defect is still `4.65e-3`, so this is a
spectral sensitivity screen, not a converged grid comparison. Source and
population hashes, all intermediate measurements, coverage audits, and
FITS checks are in the [collision-screen record](history/hot-daz-g191-b2b-collisions.json).

These remain **unconverged exploratory spectra**, with defects around
`8e-4` versus the requested `1e-4`. The 180/200 CHIANTI equivalent widths
change by only 0.005% between iterations 8 and 12. The next expansion to
240/243 changes them by 0.62/0.60%, much less than the preceding 120/150 to
180/200 expansion. This is encouraging size sensitivity on this atom ladder,
not proof of a complete physical atom. The metal reference remains ideal,
high levels retain approximate rates, and thermal/electron-density feedback
and Fe/Ni line blanketing are absent. The new predictions still leave the
two cleaner observed C III widths about 12–13% too weak. Si IV and Ly alpha
remain largely unchanged.

The raw files are saved separately from the H/He tables under
`results/hot-daz/chianti-carbon`, with source URLs and SHA256 hashes in
`manifest.json`. The frozen-host exception now permits changes in the two
metal-only modules `hot_trace_metals.py` and `light_metal_nlte.py`; every
H/He package file, compiled library, and original physical-data identity
must still match. Neither metal-only module enters the pure H/He DAO
equilibrium equations. As an additional check, the newly reconstructed
H/He background is bitwise identical to the original converged-host
prediction at every diagnostic wavelength. Tests still reject changed
H/He equations or physical tables.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=src:research MPLBACKEND=Agg MPLCONFIGDIR=/tmp/openwd-mpl-cache
python research/explore_hot_trace.py --case op --carbon-levels 180 200 10 1 \
  --seed results/hot-daz/explore-larger-c-op --iterations 12 \
  --output results/hot-daz/explore-c180-op
python research/explore_hot_trace.py --case op \
  --seed results/hot-daz/explore-c180-op --iterations 12 \
  --chianti-carbon results/hot-daz/chianti-carbon \
  --output results/hot-daz/explore-c180-chianti
python research/explore_hot_trace.py --case op --carbon-levels 240 243 10 1 \
  --seed results/hot-daz/explore-c180-chianti --iterations 12 \
  --chianti-carbon results/hot-daz/chianti-carbon \
  --output results/hot-daz/explore-c240-chianti
python research/explore_hot_trace.py --case op --refine-grid \
  --seed results/hot-daz/explore-c180-chianti --iterations 8 \
  --chianti-carbon results/hot-daz/chianti-carbon \
  --output results/hot-daz/explore-c180-chianti-fine
python research/summarize_hot_trace_exploration.py --suite collisions \
  --output results/hot-daz/collision-summary
```

The first command above now uses the corrected collision counting; the
historical pre-fix 180/200 result is retained separately for comparison.
The summary requires all cases declared in `COLLISION_CASES` to be present.
All 138 targeted tests pass, including shared light-metal and PG1159
regressions; the force/spectrum workflow has not yet been extended to
compute a levitation-modified abundance profile from these new populations.

### Nine-element composition screen (2026-10-02)

The next experiment adds the other seven elements with reported abundances
in [Preval et al. (2013), Table 10](https://doi.org/10.1093/mnras/stt1604).
Temperature, gravity and He/H stay at the same values associated with that
analysis: 52,500 K, log g = 7.53, He/H = 1e-5. We select **one elemental
abundance per element**, inferred from the ion shown below. These are not
ionic fractions, and estimates from different ions are not summed.

| Element | Adopted N(Z)/N(H) | Diagnostic used by the paper | Treatment in this screen |
| --- | ---: | --- | --- |
| C | 1.72e-7 | C III | NLTE, same OP/CHIANTI atom as the preceding screen |
| N | 2.16e-7 | N V | NLTE, III/IV/V/VI = 40/60/40/1 levels |
| O | 4.12e-7 | O IV | NLTE, III/IV/V/VI/VII = 40/60/40/10/1 |
| Al | 1.60e-7 | Al III | NLTE, III/IV/V = 40/20/1 |
| Si | 3.68e-7 | Si IV | NLTE, same 50/40/30/1 atom as before |
| P | 1.64e-8 | P V | NLTE, III/IV/V/VI = 30/50/40/1 |
| S | 1.71e-7 | S IV | NLTE, III/IV/V/VI/VII = 30/60/40/30/1 |
| Fe | 5.00e-6 | Fe V | LTE line and ground bound-free opacity |
| Ni | 1.01e-6 | Ni V | LTE line and ground bound-free opacity |

The paper's other estimates are retained in `hot_trace_composition.py`:
N IV 1.58e-7, P IV 8.40e-8, S VI 5.23e-8, Fe IV 1.83e-6, Ni IV 3.24e-7.
The runner supports a separate alternative-abundance experiment; the values
above are fixed inputs, not adjusted to improve the observed fit. Ge IV is
also detected in this star, but Table 10 does not give a Ge abundance and
our atomic database has no Ge atom. This screen covers the nine elements
with abundance measurements in that table, not every identified species.

The experimental trace API now accepts additional supported elements only
with explicit atom sizes. It retains full-ion-ladder Saha references at the
host's unchanged electron density. Partial warm starts preserve the existing
C/Si departures and initialize new elements at LTE. Tests require those new
LTE states to be present in the first saved, evaluated spectrum. The combined
LTE-plus-NLTE composition is checked against the same trace mass and electron
budgets; Fe/Ni cannot evade them by entering through the background callback.

#### Additional atomic data

`research/hot_trace_composition.py` verifies SHA256 hashes before loading
supplements from `results/hot-daz/multimetal-data`. These files do not change
the H/He runtime tables or invalidate the certified frozen host.

- [Verner & Yakovlev (1995)](https://www.pa.uky.edu/~verner/photo.html), Table 1,
  supplies missing Al/Ni ionization thresholds and P/Ni photoionization data.
  The new adapter implements Eq. 1 for the **lowest-threshold subshell only**,
  with the published orbital-angular-momentum exponent and a 100-keV upper
  energy bound. Inner-shell/Auger absorption is omitted. Existing Verner 1996
  data are retained for the elements they cover. Thus this is an explicit
  outer-shell approximation, not invented ground-state Kramers data or the
  later, detailed Ni cross-section calculations.
- The [SYNSPEC gfFUV99 list](https://tlusty.oca.eu/tlusty/Synspec49/synspec-line.html)
  supplies 28,851 input Fe III–VII and Ni III–VI lines over 880–1990 Å.
  It derives from Kurucz lists with measured energy levels. A numeric format
  adapter feeds the existing Kurucz reader, retaining parity-order energy
  records, log(gf), and damping coefficients. It replaces E1 transitions in
  the covered interval rather than adding a second copy. Existing Stout
  forbidden transitions and transitions outside this interval remain.
  Level energies give the vacuum wavelengths; every selected validation
  transition agrees with its atlas laboratory wavelength within 0.001 Å.
- Fe/Ni levels added by that reader also enter their LTE partition functions.
  Fe/Ni include ground bound-free opacity and ordinary Doppler/natural/impact
  line profiles. Their populations remain LTE. Predicted-energy line lists,
  complete EUV Ni opacity, excited-state Fe/Ni continua and Fe/Ni statistical
  equilibrium remain absent. **This is not a fully metal-blanketed atmosphere.**
- N/O/Al/P/S use Stout line data, approximate electron collisions and Kramers
  excited continua. C retains the preceding OP/CHIANTI treatment. No new
  OP resonance data or accurate collision datasets are claimed for those
  five elements. All selected explicit levels are below their ionization
  thresholds; omitted levels remain the existing inert LTE reservoirs.

#### Observations and comparisons

The emitted surface spectrum spans 910–1990 Å, with 0.01-Å base sampling and
finer diagnostic sampling. The all-elements radiation grid has 206,118
points, including a 0.02-Å mesh across the added UV Fe/Ni line forest. The
light-element-only control has 150,618 radiation points. These are initial
sampling choices, not a resolution-convergence claim.

In addition to the existing E140H data, we downloaded the matched
[MAST atlas](https://archive.stsci.edu/prepds/wd-linelist/) E230H and FUSE
coadds and their line catalogues. `compare_hot_composition.py` compares
N V 1238/1242, O IV 1338/1343, Al III 1854/1862, P V 1117/1128,
S IV 1062/1072, Fe V 1409.453 and Ni V 1306.624. Diagnostic transitions
and published abundances are fixed before viewing these new predictions.
The Fe/Ni lines have no neighbouring catalogue identification within 0.13 Å.

STIS keeps the fixed photospheric velocity of 23.8 km/s and Gaussian
R=144,000 approximation. FUSE uses assumed Gaussian R=20,000 and the
published velocity of each diagnostic line because MAST explicitly warns
that its coadd has uncorrected segment shifts. Those shifts are not fitted
to our model. Only a local two-coefficient continuum scale is fitted, using
sidebands with catalogue lines removed. Diagnostic RMS excludes catalogue
contaminants. Equivalent widths use the same fixed apertures on model and
data (±20 km/s for STIS, ±35 km/s for FUSE); they are **aperture widths** and
need not equal the paper's fitted equivalent widths. Particularly for FUSE,
only a few resolution elements sample each line and profile/LSF/continuum
uncertainties matter. No ISM components are added in these new windows.

The original C III/Si IV/Ly-alpha comparison retains its masks so its RMS
can still be compared directly with earlier screens. Adding predicted blends
does not quietly change which pixels define that score. Prediction FITS
headers now record every elemental abundance and separate NLTE and LTE
species; the intrinsic spectrum has no instrument, velocity or ISM applied.

#### First eight-iteration results

| Composition | C III 1175.987 EW (mÅ) | C III 1176.370 EW (mÅ) | C III RMS | Si IV 1402 RMS |
| --- | ---: | ---: | ---: | ---: |
| Previous C/Si control | 14.940 | 17.536 | 3.073% | 2.789% |
| + N/O/Al/P/S NLTE | 15.282 | 17.919 | 3.031% | 2.700% |
| + Fe/Ni LTE opacity | 16.501 | 19.568 | 2.985% | 3.699% |
| Observed aperture EWs | 17.005 ± 0.697 | 20.042 ± 0.608 | — | — |

The full mixture's cleaner C III widths are now 3.0% and 2.4% below the
observations, versus 12.1% and 12.5% before. Most of the additional change
comes when Fe/Ni opacity is included, not from the five new NLTE light
metals alone. The C III widths change by about 1.2% between iterations 4
and 8. This is evidence of a promising opacity effect, not an abundance
measurement. The entire multiplet RMS improves only modestly, and the
Si IV 1402 region is worse because additional predicted absorption does
not all coincide with observed features. Ly-alpha and Si IV 1393 RMS
change from 3.780/3.287% to 3.708/3.051%.

Several newly predicted lines are encouraging, but there is no uniform fit:

| Diagnostic | Observed aperture EW (mÅ) | Full mixture at iteration 8 (mÅ) |
| --- | ---: | ---: |
| N V 1242 | 53.8 | 77.2 |
| O IV 1338 | 16.5 | 4.5 |
| O IV 1343 | 22.3 | 6.8 |
| Al III 1854 | 17.1 | 13.1 |
| Al III 1862 | 11.1 | 7.1 |
| P V 1117 | 70.8 | 60.2 |
| P V 1128 | 43.4 | 47.7 |
| S IV 1062 | 15.4 | 12.6 |
| S IV 1072 | 19.0 | 18.1 |
| Fe V 1409 | 28.5 | 18.9 |
| Ni V 1306 | 19.2 | 6.3 |

N V 1238 is close in central depth, while the other doublet member is too
strong. O IV remains substantially too weak. Al III and S IV have plausible
strengths, though S IV is still changing appreciably. The LTE Fe/Ni lines
recover genuine absorption absent from the previous model but remain weak
in the chosen diagnostics. The delivered FUSE P V profiles visibly retain
an offset from the published catalogue positions even after using their
catalogue velocities. Their RMS therefore includes a wavelength-registration
problem and should not be interpreted solely as an opacity/abundance error.
We did not optimize a new wavelength shift against the model.

**Both new population solutions remain strongly unconverged:** the maximum
undamped defect is approximately 1, compared with the requested 1e-4.
The iteration-8 spectrum is a saved, evaluated state, not an unchecked
proposal. A few relatively stable diagnostic lines do not certify the
other populations, especially in the new atoms and deep layers. This screen
is deliberately stopped for diagnosis rather than presented as a converged
nine-element atmosphere. Before abundance fitting, the new light-metal
atoms need further checks, Fe/Ni need NLTE populations, and the structure
needs the metal opacity in its energy balance.

Numerical safeguards passed: represented-particle closure is better than
3e-15, all sampled surface fluxes are finite and positive, and H/He-only
fluxes are bitwise unchanged at 133,159 shared wavelengths. The total metal
mass fraction is 3.68e-4 and the fully stripped electron bound is 1.75e-4,
within the fixed-host trace limits. All 147 distinct targeted tests passed,
including new-atom Planck detailed balance, partial warm starts, Verner95
units/domain, format conversion, and Fe/Ni LTE Kirchhoff checks. Exported
FITS abundances, NLTE/LTE species, convergence flags and checksums were
roundtripped. None of these checks establishes astrophysical accuracy.

Results, source/data hashes, intermediate carbon metrics and all twelve
additional-line measurements are recorded in
[the composition history](history/hot-daz-g191-b2b-composition.json).
The main artifacts are:

- `results/hot-daz/composition-summary/additional-elements.png` and `.pdf`
- `results/hot-daz/composition-summary/carbon-silicon.png` and `.pdf`
- `results/hot-daz/composition-all/comparison/prediction.fits`
- Each case's `spectrum.npz`, `populations.npz`, `metadata.json`, and
  `iteration-001/004/008` snapshots (the control has only iteration 1).

Reproduction after installing the pinned data in the recorded manifest:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=src:research MPLBACKEND=Agg MPLCONFIGDIR=/tmp/openwd-mpl-cache
python research/explore_hot_composition.py --mode csi --iterations 1 \
  --output results/hot-daz/composition-csi
python research/explore_hot_composition.py --mode light --iterations 8 \
  --output results/hot-daz/composition-light
python research/explore_hot_composition.py --mode all --iterations 8 \
  --output results/hot-daz/composition-all
python research/compare_hot_composition.py \
  --model CSi=results/hot-daz/composition-csi \
  --model Light=results/hot-daz/composition-light \
  --model All=results/hot-daz/composition-all \
  --output results/hot-daz/composition-summary
python research/summarize_hot_composition.py
```

The runner refuses to overwrite existing runs. Choose new output directories
for another experiment and adjust the comparison inputs accordingly. No
radiative-levitation profile is imposed by this composition test.

## Further line diagnostics: oxygen rates, Fe/Ni NLTE, and FUSE registration

The next experiments keep the same certified 52,500 K / log g 7.53 H/He
structure. They are **separate experiments**, not a combined atmosphere.
The continued control restores all seven NLTE elements from the previous
checkpoint. The restart reader previously restored only C and Si; it now
discovers every saved element and optionally filters the requested set.
The solver reports residuals for each element and the level/depth with the
largest residual. A large defect in a negligible outer-layer population
must not be confused with an equally large change in a measured line.

The control runs for twelve further evaluated iterations; the other screens
save iterations 1, 4 and 8. The runner now preserves seed abundance choices,
oxygen atom sizes, oxygen OP/collision settings and promoted iron-group
species on continuation. Each new run captures its research source files
at startup. Saved spectra, population tables and source snapshots are
checksummed in the new history record.

### Oxygen and iron-group experiments

O III–VII was expanded from 40/60/40/10/1 to 100/150/120/60/1 bound levels.
Levels above the ionization threshold are rejected. TLUSTY/OP tables supply
photoionization for 88 O IV, 82 O V and 57 O VI selected levels, with the
existing fallbacks for unmatched levels. A subsequent experiment adds 5,557
CHIANTI collision pairs: 190 O III, 351 O IV, 4,950 O V and 66 O VI.
Matching requires configuration, spin, angular momentum and energy; source
level numbers are not assumed to be Stout numbers. Both endpoints of the
O IV 1338/1343 diagnostics match. The sources and hashes are recorded in
`results/hot-daz/chianti-oxygen/manifest.json`; the
[CHIANTI O IV documentation](https://db.chiantidatabase.org/o/o_4.html)
identifies the collision calculation and subsequent level corrections.

The expanded OP atom recovers Planck LTE to 7.6e-15 relative error; adding
the real collision data recovers it to 7.4e-15. These detailed-balance checks
test consistency, not the accuracy of the nonthermal prediction. The O IV
oscillator strengths also agree with the independent SYNSPEC line list
at the percent level. Nevertheless, the O IV lines remain much too weak.
Collision data change the two aperture EWs by less than 0.02 mÅ from the
OP-only result. Simply enlarging this atom or replacing these collision
rates is therefore insufficient at the present fixed structure.

Separate Fe and Ni runs promote all available bound levels of stages IV–VI
plus the VII ground state: Fe 276/332/248/1 and Ni 235/323/271/1. Closed
explicit lines and their continua use NLTE populations. Lines/stages outside
those atoms retain their original LTE opacity; they are neither removed nor
double counted. A regression test verifies that the closed and remaining
line opacity sum to the original LTE result. These remain incomplete atoms
with approximate excited-state continua and collision data.

| Aperture EW (mÅ) | Observed | Continued control | Expanded O + OP + collisions | Fe NLTE | Ni NLTE |
| --- | ---: | ---: | ---: | ---: | ---: |
| O IV 1338 | 16.45 | 4.43 | 4.69 | 4.29 | 2.94 |
| O IV 1343 | 22.31 | 6.80 | 7.17 | 7.16 | 6.86 |
| Al III 1854 | 17.06 | 13.03 | 13.65 | 15.15 | 13.02 |
| Al III 1862 | 11.13 | 7.07 | 7.46 | 8.34 | 7.29 |
| S IV 1062 | 15.40 | 12.21 | 12.78 | 13.22 | 9.98 |
| S IV 1072 | 18.97 | 17.69 | 18.34 | 18.58 | -0.12 |
| Fe V 1409 | 28.50 | 18.87 | 18.87 | 19.29 | 19.23 |
| Ni V 1306 | 19.17 | 6.31 | 6.31 | 6.40 | 25.86 |

Fe NLTE modestly improves Al III and S IV, while its own Fe V diagnostic
remains weak. Its C III component EWs remain 16.68/19.49 mÅ, and the C III
region RMS is 2.974%, compared with 2.985% in the control. Al III and S IV
EWs change by at most 2.6% between Fe iterations 4 and 8.

Ni NLTE strengthens Ni V and weakens the unwanted Ni IV absorption near
observed 1402.65 Å. That feature is the retained Ni IV 1402.542 transition,
lower/upper Stout indices 38/253; it is not improved by deleting the line.
However, Ni NLTE creates strong absorption absent from the observations,
damages continuum placement in several FUSE windows, and raises the C III
RMS to 10.21%. The negative S IV aperture EW above reflects that distorted
local comparison, not a reliable prediction of intrinsic S IV emission.
This Ni treatment is **not an acceptable overall improvement**. More complete
iron-group atoms and photoionization data remain necessary; the controlled
[Preval et al. nickel study](https://doi.org/10.1093/mnras/stw2800) also
demonstrates the sensitivity to nickel atomic data.

A final separate run uses the paper's N IV-based elemental abundance,
N/H=1.58e-7, instead of its N V-based 2.16e-7, on the expanded-oxygen case.
N V 1238/1242 EWs decrease from 90.97/82.04 to 87.99/77.76 mÅ; observed
values are 73.12/53.78. This is a modest improvement using another published
abundance, not an abundance fit or a solution to the N V discrepancy.

All these metal solutions remain unconverged. Final maximum defects are
0.9985 (continued control), 0.9297 (expanded oxygen plus collisions),
0.9989 (Fe), 1.0000 rounded (Ni), and 0.0338 (lower N), versus 1e-4 requested.
The nearly stable O IV lines and individual improvements justify screening
the experiments, but do not certify any of them for abundance inference.

### P V 1128 wavelength offset

The plotted P V offset was a data-registration problem. The
[MAST atlas](https://archive.stsci.edu/prepds/wd-linelist/) explicitly warns
that its FUSE coadd has uncorrected segment shifts. Even using the catalogue's
individual line velocities does not align the delivered coadd.

`research/register_fuse_diagnostics.py` fits empirical, bin-integrated
Gaussian absorption profiles with a local linear continuum to nearby N IV
1122.055 and Si IV 1122.485/1128.340. Their catalogue-to-coadd offsets are
approximately +0.0593, +0.0579 and +0.0629 Å. Their median is **+0.05934 Å**,
equivalent to +15.77 km/s at 1128 Å, with 0.00258 Å scatter among anchors.
P V is excluded from estimating this correction. Its independently measured
offset is +0.05965 Å, a check consistent with the neighbouring lines.
The scatter is not a complete systematic-error estimate: profiles are
approximate and the coadd is sampled at 0.04 Å.

Applying the local correction to the model-to-data comparison reduces the
control's P V 1128 diagnostic RMS from **25.01% to 2.60%**, with unchanged
phosphorus abundance and intrinsic spectrum. Recentring the aperture gives
an observed EW of 59.01 mÅ and predicted EW of 53.50 mÅ. These supersede the
miscentred 43.42/48.40 mÅ measurements for this line. Each diagnostic aperture
and contaminant mask follows its adopted centre, and continuum sidebands
are refitted in the same prescribed manner.

This is a local correction over 1121.5–1129.2 Å, not a revised stellar RV
or a recalibrated FUSE spectrum. Neither the input coadd nor the intrinsic
model is changed. P V 1117 still has a residual offset; the 1128 correction
is not extrapolated to it or to the S IV windows. The original comparisons
are retained alongside the registered version.

### Checks, saved results and reproduction

208 distinct targeted tests passed, including new oxygen level mapping,
all-element restarts, retained Fe/Ni opacity and empirical wavelength
registration. All five final cases have finite positive fluxes,
represented-particle closure below 1e-12, identical H/He-only flux at shared
wavelengths, verified source snapshots, and roundtripped FITS abundances,
species, convergence flags and checksums. The first Ni attempt failed while
serializing a NumPy integer in metadata; its partial files are marked failed
and excluded. The subsequent run converts atom sizes to ordinary integers.

The later synthesis outputs cover separate diagnostic windows, not a
continuous 910–1990 Å spectrum. The common radiation mesh still includes
the iron-group UV opacity. No new levitation profile or metal contribution
to the atmospheric energy balance was introduced in these tests.

- [Machine-readable experiment history](history/hot-daz-g191-b2b-other-lines.json).
- `results/hot-daz/other-lines-summary/additional-elements.png` / `.pdf`.
- `results/hot-daz/other-lines-summary/carbon-silicon.png` / `.pdf`.
- `results/hot-daz/other-lines-summary/pv1128-registration.png` / `.pdf`.
- `results/hot-daz/fuse-local-registration/registration.json` and anchor plot.
- Per-case source snapshots, intermediate populations, and prediction FITS.

The additional runner switches are `--oxygen-op`, `--oxygen-levels
100 150 120 60 1`, `--oxygen-collisions results/hot-daz/chianti-oxygen`,
`--nlte-iron-group Fe` (or `Ni`), and `--published-alternatives N`.
With the pinned raw inputs in place and the environment variables above:

```bash
python research/register_fuse_diagnostics.py \
  --output results/hot-daz/fuse-local-registration
python research/compare_hot_composition.py \
  --model Control=results/hot-daz/composition-resume \
  --model Oxygen=results/hot-daz/composition-oxygen-collisions \
  --model IronNLTE=results/hot-daz/composition-iron-nlte \
  --model NickelNLTE=results/hot-daz/composition-nickel-nlte-v2 \
  --model LowerN=results/hot-daz/composition-nitrogen-alternative \
  --fuse-registration results/hot-daz/fuse-local-registration/registration.json \
  --output results/hot-daz/other-lines-summary
python research/summarize_hot_trace_repairs.py \
  --model Control=results/hot-daz/composition-resume \
  --model Oxygen=results/hot-daz/composition-oxygen-collisions \
  --model IronNLTE=results/hot-daz/composition-iron-nlte \
  --model NickelNLTE=results/hot-daz/composition-nickel-nlte-v2 \
  --model LowerN=results/hot-daz/composition-nitrogen-alternative \
  --output results/hot-daz/other-lines-summary \
  --verification results/hot-daz/other-lines-verification.json \
  --unregistered-comparison results/hot-daz/repair-completed-first \
  --history docs/development/history/hot-daz-g191-b2b-other-lines.json
```

## Continuing the seven-element solution to numerical convergence

The convergence runs keep the preceding lower-nitrogen case's physical
inputs fixed: the certified 52,500 K, log g 7.53 H/He host; C/Si/N/O/Al/P/S
in NLTE; Fe/Ni opacity in LTE; expanded O III--VII atoms with OP continua
and CHIANTI collisions; and N/H = 1.58e-7 from the published N IV inference.
The convergence threshold is a maximum **undamped** population-map residual
of 1e-4 across every explicit level, depth and solved element, with the
existing 1e-12 elemental-population floor. A small damped update or nearly
unchanged emergent line profile does not meet this criterion.

`research/explore_hot_composition.py` now accepts `--tolerance`, `--damping`,
`--acceleration-depth`, `--checkpoint-every` and `--require-convergence`.
Checkpoints retain evaluated states and their own radiation-field residuals.
The last switch preserves a failed run's output but raises an error if it
does not satisfy the requested tolerance. Each run snapshots its research
sources and verifies the numerical code/data identity before saving the
final result. `--elementwise-acceleration` optionally fits separate Anderson
histories for each element; the coupled transfer, statistical-equilibrium
map, conservation normalization and convergence test stay the same. This
research option is checked against the ordinary solver on a coupled C/Si
test problem; it does not replace the production default.
`--ionwise-acceleration` instead partitions the histories by ion stage,
while the represented-particle normalization still applies to whole
elements. The two grouping options are mutually exclusive. The ion-stage
option passes the same coupled C/Si fixed-point comparison.

The final audit uses a separate one-evaluation process seeded from the
converged populations, synthesizing continuous 910--1990 Å coverage. It
requires that freshly computed residuals pass the same threshold and checks
the host certificate, source snapshots, source-function closure, represented
particle conservation, unchanged saved populations, reproducible spectrum,
FITS checksums and convergence flags. The observational comparison then runs
without its exploratory convergence bypass.

This certifies the fixed-host trace-metal problem only. The optional
`--energy-audit` evaluates the H/He control and metal-containing state on a
common union of the thermal and metal-rate wavelength meshes. It measures
the flux and heating changes omitted by holding the structure fixed; it
does not solve the metal-blanketed atmospheric energy balance. The H/He
control on that same mesh exposes changes due to quadrature alone.

### Converged result and independent replay

The ion-stage continuation reached a maximum undamped population residual
of **4.97903e-5**. A fresh process, using the saved populations as its initial
guess, recomputed the radiation field and statistical equilibrium and
obtained **4.97903e-5** on its first evaluation. The continuous output has
136,161 wavelength points spanning 910--1990 Å, including the finer
diagnostic samples. The replayed spectrum agrees with the original at
shared wavelengths to 2.22e-15 relative; the H/He-only flux is bitwise
identical. Represented-particle closure is 2.30e-15 and the final spectral
source-function closure is 1.35e-15.

| Solved element | Maximum undamped population residual |
| --- | ---: |
| C | 3.2825e-5 |
| Si | 4.9790e-5 |
| N | 8.2656e-7 |
| O | 3.7267e-6 |
| Al | 1.2502e-5 |
| P | 4.8335e-8 |
| S | 1.9238e-7 |

The four evaluations in the last stage start from a nearly converged
checkpoint. The saved continuation chain contains 30 evaluations with a
joint history of 6, 20 with a joint history of 40, then 4 with histories of
16 per ion stage. The convergence figure includes this earlier work.
Grouping histories by element alone had not helped the initially limiting
Si populations; that trial was stopped with its checkpoints preserved.
Later, after six elements passed, isolating the slow C V excitation modes
by ion stage removed the remaining carbon bottleneck.

The separate joint-history-40 run also converged, after 30 evaluations,
at 9.33011e-5. Its diagnostic spectrum differs from the adopted solution by
at most **1.51e-6 relative flux** (1.5 ppm), with an RMS difference of
5.43e-8. Some sparsely populated C V states still differ by 1.38% when
weighted by the convergence floor. A fixed-point residual is not an error
bound on every weak population; the emergent UV spectrum is much less
sensitive to those differences. Neither comparison is an independent
atmosphere-code validation.

The strict comparison and export audit passes without allowing unconverged
models. FITS checksums, abundances, source snapshots, host certificate and
convergence flags were checked. `METCONV=T`, `HOSTFIX=T`, `MTHERM=F`,
`MCHARGE=F` and `VALIDATE=F` explicitly distinguish trace-population
convergence from thermal feedback, charge feedback and observational
validation. All 42 targeted trace-metal, restart, composition, comparison
and convergence-audit tests pass.

### Effect on the observed lines

The longer solve barely changes the previous spectrum. The two isolated
C III aperture EWs are **16.6693 / 19.7531 mÅ**, compared with observed
**17.0049 ± 0.6973 / 20.0419 ± 0.6081 mÅ**. The quoted observational errors
are statistical only. The normalized diagnostic RMS values are 2.986%
for C III, 3.713% for Ly-alpha, and 3.034% / 3.685% for the Si IV doublet.

| Diagnostic | Observed aperture EW (mÅ) | Converged prediction (mÅ) |
| --- | ---: | ---: |
| N V 1238 | 73.116 | 87.670 |
| N V 1242 | 53.776 | 77.625 |
| O IV 1338 | 16.452 | 4.683 |
| O IV 1343 | 22.306 | 7.150 |
| Al III 1854 | 17.062 | 13.599 |
| Al III 1862 | 11.132 | 7.433 |
| P V 1117 | 70.844 | 61.798 |
| P V 1128, locally registered | 59.006 | 53.931 |
| S IV 1062 | 15.404 | 12.748 |
| S IV 1072 | 18.966 | 18.302 |
| Fe V 1409 | 28.502 | 18.870 |
| Ni V 1306 | 19.167 | 6.315 |

All twelve additional-line EWs differ by less than 0.4% from the preceding
lower-N screen. N V remains too strong, and O IV, Fe V and Ni V remain too
weak in this model. The P V 1128 local-registration correction is retained;
P V 1117 still has a catalogue-to-coadd offset, so its profile mismatch
cannot yet be interpreted purely as a line-strength error. These are fixed
published-abundance predictions with conditional continuum fits, not a new
abundance determination.

### The fixed atmosphere is not in equilibrium after adding metals

The energy audit is a significant limitation on interpreting this result.
On the same 310,478-point union of host and metal-rate wavelength meshes,
the H/He-only control has surface F/(sigma Teff^4) = **1.0000752** and a
maximum all-depth flux residual of 9.81e-5. With metals added at the fixed
structure, the surface ratio is **0.8902491**, an approximately **11%
bolometric flux deficit**, and the maximum all-depth flux residual is
**0.402**. The maximum absolute net-cell-heating/emission ratio is 4.47,
compared with 0.00364 for the H/He control on this quadrature. Both formal
transfer solutions close to about 2e-15.

Thus the seven-metal trace populations are numerically converged, but the
metal-containing atmosphere is **not** in radiative equilibrium. The
common-grid control shows that the flux deficit is much larger than the
quadrature change. The populations were frozen during this energy audit;
it does not constitute convergence on a refined population grid. The next
physical development priority is to include metal opacity and emissivity
in the temperature-structure solve, updating the radiation field and
populations consistently, before drawing further abundance conclusions.
Fe/Ni are still LTE opacity sources in this benchmark, and incomplete
atomic data remain another limitation.

### Saved outputs and reproduction

- [Convergence audit and experiment record](history/hot-daz-g191-b2b-converged.json).
- `results/hot-daz/converged-summary/convergence.png` / `.pdf`.
- `results/hot-daz/converged-summary/strict-comparison/prediction.fits`:
  intrinsic rest-frame surface flux, without instrumental, RV or ISM effects.
- `results/hot-daz/converged-summary/strict-comparison/comparison.png`:
  C III, Ly-alpha and Si IV compared with STIS.
- `results/hot-daz/converged-summary/additional-lines/additional-elements.png`:
  the twelve additional diagnostics, with their arrays and metrics.
- `results/hot-daz/composition-converged-full/energy-audit/energy-balance.png`
  / `.pdf` / `.json` / `.npz`.
- `results/hot-daz/converged-method-comparison.json`: two converged methods.
- `results/hot-daz/converged-summary/study-summary.json`: artifact provenance,
  method trials, observational changes and validation summary.

With the pinned inputs and saved warm-start checkpoint, the final stage and
fresh replay can be reproduced using new output directories:

```bash
python research/explore_hot_composition.py --mode all \
  --seed results/hot-daz/composition-converge-history40/iteration-020 \
  --iterations 80 --tolerance 1e-4 --acceleration-depth 16 \
  --ionwise-acceleration --checkpoint-every 5 --require-convergence \
  --diagnostic-only --output results/hot-daz/ionwise-reproduction
python research/explore_hot_composition.py --mode all \
  --seed results/hot-daz/ionwise-reproduction \
  --iterations 1 --tolerance 1e-4 --require-convergence --energy-audit \
  --output results/hot-daz/converged-full-reproduction
python research/verify_hot_composition_convergence.py \
  --model results/hot-daz/ionwise-reproduction \
  --replay results/hot-daz/converged-full-reproduction \
  --previous results/hot-daz/composition-nitrogen-alternative \
  --output results/hot-daz/convergence-audit-reproduction
```

### Full UV comparison in wavelength panels

`research/plot_hot_daz_uv_atlas.py` compares the adopted converged trace
solution with the FUSE and STIS atlas over the available model interval,
910--1990 Å. Twelve 90 Å panels are saved as three four-panel PNG/PDF pages
and one combined three-page PDF in `results/hot-daz/full-uv-comparison`.
The E230H observations beyond 1990 Å are outside this synthesis and are
not plotted. E140H is preferred in the observational overlaps: FUSE is
shown below 1160 Å, E140H from 1160 to 1685 Å, and E230H above 1685 Å.

The comparison retains observed flux units and native observed bins. A
single surface-to-observed flux scale, **8.92655e-23**, is estimated from
29,527 continuum candidates in STIS E140H at 1300--1600 Å. Catalogue lines
are masked by ±20 km/s, with robust clipping of remaining flux-ratio
outliers. That same factor applies to every instrument and panel; no local
continuum normalization or abundance fit is performed. The model is
shifted by 23.8 km/s, convolved with the existing Gaussian instrumental
approximation, and averaged over observed bins. Six samples at the blue
edge have no model because convolution would require extrapolation.

The previously adopted H I Ly-alpha absorber is included. Other foreground
absorption and airglow remain unmodeled. The overview uses a common stellar
velocity, not the line-by-line FUSE catalogue velocities used for individual
diagnostics. The local P V 1128 catalogue-to-coadd correction is not a
global wavelength solution and is not extrapolated to this overview.
Input checksums, plot settings, the shared scale and sampled arrays are
saved alongside the figures. The model spectrum itself is unchanged.

```bash
python research/plot_hot_daz_uv_atlas.py \
  --model results/hot-daz/composition-converged-full \
  --output results/hot-daz/full-uv-comparison
```

The STIS detail version in `results/hot-daz/full-uv-stis-20a` uses 20 Å
panels from 1160 to 1990 Å (the final panel spans 10 Å): 42 panels across
11 pages in `stis-detail.pdf`. `full-uv-comparison.pdf` adds a three-panel
FUSE overview for 12 pages total. Each page also has a separate PNG and
PDF. STIS vertical limits zoom to the full local data/model range, while
the native observed bins, sampled model and single shared flux scale are
unchanged from the broad overview.

```bash
python research/plot_hot_daz_uv_atlas.py \
  --layout fine-stis --stis-panel-width 20 \
  --output results/hot-daz/full-uv-stis-20a
```

## Iron-group NLTE, convergence and speed (2026-10-03/04)

### Why the nine-element model missed G191-B2B

An audit of the converged nine-element solution traced most of the excess
UV line absorption, including the Fe IV forest near 1560--1620 Å, to the
iron group. With LTE Fe/Ni populations at the fixed host electron density,
Fe IV and Ni IV are about three times overpopulated relative to their NLTE
values. Promoting Fe/Ni IV--VI to explicit NLTE atoms (all bound levels of
the Stout/gfFUV99 ions; the ground of charge 6 as the top stage) exposed
two defects of the truncated atoms:

1. **Missing EUV resonance lines.** gfFUV99 is cut at 880--1990 Å, so the
   3d^(n-1)4p -> 3d^n resonance transitions are absent and excited levels
   have no radiative route to the ground term. They are supplemented from
   the Kurucz measured-level line lists (`gf2603.pos` ... `gf2806z.pos`,
   checksummed in `results/hot-daz/kurucz-pos/SHA256SUMS`). Only level pairs
   not already connected are added; existing lines are unchanged
   (`research/hot_trace_composition.py`, `--kurucz-positions`).
2. **Recombination deficit.** Radiative recombination into the explicit
   levels alone is 3--5 times below the CHIANTI total for Fe. The missing
   part (CHIANTI Shull & van Steenberg radiative plus Mazzotta dielectronic,
   `results/hot-daz/chianti-recombination`) is added as a pair between the
   parent ground and the recombined ground with its LTE inverse, which
   keeps exact detailed balance (`total_recombination_rate_coefficients` in
   the shared level solver; `research/hot_trace_recombination.py`). For the
   light metals the CHIANTI totals are dominated by low-density dielectronic
   recombination and are not applied.

### Convergence: line-weighted ALI and an opacity criterion

Plain lambda iteration stalled in the Fe/Ni forest. The rates are now
preconditioned with the diagonal approximate lambda operator of the current
radiation field (`accelerated_lambda=True`); each line's operator is
weighted by its share of the total extinction at line centre (Rybicki &
Hummer 1991), without which overlapping forest lines over-corrected and
diverged. Convergence is tested on the quantity that enters the transfer:
`convergence_criterion="opacity"` measures the change of metal absorption
and emissivity produced by the undamped proposed populations relative to
the total extinction and emission at every population wavelength and
depth. The population defect, dominated by levels that carry no opacity,
remains recorded. With both, the warm-started nine-element model converged
to an opacity defect of 9e-5 (`results/hot-daz/irongroup-converged`).

### Comparison with TMAP

At the TMAP abundances of the TheoSSA 52 kK / log g 7.55 model
(`results/hot-daz/tmap-reference`), the OpenWD fixed-host solution matches
the TMAP equivalent widths of C III, O IV, Al III, P V and Fe V to 15--25%
(`results/hot-daz/tmap-abundances`).

### Metals in the temperature structure (exploratory)

`research/hot_metal_structure.py` relaxes the H/He host temperature with
the metal opacities included (TMAP-style hybrid: local radiative
equilibrium with frozen metal departures, Unsöld--Lucy deep correction,
then H/He NLTE and metal NLTE re-solves). On the fixed host the nine metals
leave a 13.7% flux deficit; the relaxed structure is warmer by 1.5--2% in
the photosphere and 4--7% deep. Outer-layer heating above tau_Ross = 1e-5
comes from Fe VII+ lines kept in the LTE remainder (Saha populations with
an LTE source where J > B); it does not affect the emergent spectrum.
`research/plot_structure_comparison.py` compares both models with the data
in 20 Å panels (`results/hot-daz/structure-comparison`): most lines are
unchanged, while C III 1175, Al III 1855/1863 and S IV 1063/1073 weaken by
19--28% in equivalent width, toward the observed depths. Because this is
small compared with the other uncertainties and the coupled relaxation is
expensive, the fixed-host solution remains the default; the structure
coupling is optional.

### Speed

Profiling one iteration (40 depths, 1.17 M population wavelengths) showed
that ~80% of the time went to line-profile sums. Deep-layer Fe/Ni lines are
Stark broadened to ~1 Å, so the 100-HWHM windows (capped at 10% of the
wavelength) cover ~10^5 grid points each: 2.6e10 profile evaluations per
pass. The changes below leave the converged solution unchanged to the
stated tolerances; all of them are covered by
`tests/test_trace_metal_speedups.py`.

- **Fused line means** (`metal_line_profile_means`, `csrc/rt_core.c`). The
  photon occupation, its thermodynamic inverse and the lambda operator are
  averaged in one pass over each profile, on contiguous per-depth copies of
  the fields. Bit-identical to the former separate calls.
- **Block quadrature** (`profile_block_tolerance`, default 1e-4 in the
  trace-metal solver). Aligned blocks of 16--4096 grid intervals carry the
  trapezoid moments of each field; a block replaces its intervals where the
  profile is linear to the tolerance at both block ends. Line opacity uses
  the same test and deposits a value and slope per block, expanded onto the
  grid afterwards (`expand_metal_line_blocks`). Narrow cores keep the fine
  grid. Numerator and normalization share the quadrature (a constant field
  is returned exactly), and upward and downward rates share it (exact
  detailed balance). On the real problem 1e-4 changes the rate-equation
  populations by <4e-7. Emergent spectra are always computed exactly.
- **Cumulative Kramers continua** (`kramers_cumulative`). With
  sigma = sigma0 (E_i/E)^3 above each edge, the summed Kramers opacity and
  the photoionization/recombination integrals are E^-3 times prefix sums
  over levels sorted by edge, instead of one grid pass per level. Level and
  depth pairs where the per-level non-negativity clip can act keep the
  per-level accumulation. Same sums to rounding (2e-14).
- **Vectorized rate assembly and cached collisions.** Bound-bound rates for
  all transitions and depths are formed at once and added to each depth's
  matrix in the original per-element order (bit-identical). Electron-impact
  coefficients depend only on the fixed host and are cached per element for
  the iterations of a solve (`rate_cache`).
- **Shared thermal arrays.** B_lambda and exp(-h nu/kT) on the population
  grid are built once per coefficient evaluation rather than per element.
- **Gated undamped residual** (`opacity_check_gate=3`). The undamped
  opacity residual needs a second full coefficient evaluation; it is
  evaluated only once the free iterate-to-iterate opacity change is below
  three times the tolerance (and in the last allowed iteration).
  Convergence is still declared only on the undamped residual.
- **No closure check inside the iteration.** `transfer_field` repeats an
  independent fixed-source solution to report the source closure; inside the
  iteration only the mean intensity is used, so the check is skipped there
  (`check_source=False`) and kept for the emergent spectrum.
- **Cold-start update.** Starting from LTE, the first statistical-
  equilibrium solution is accepted in full. When the Anderson step is
  rejected, the damped fallback now moves falling populations halfway in
  log (geometric mean) instead of halving them; rising populations keep the
  arithmetic mean, and each element's particles are restored. Outer-layer
  O/C levels that must fall by 5--10 decades from LTE previously needed one
  iteration per factor of two.

### Cold start and the convergence criterion

At 40 depths the original code took 11--12 min per iteration
(single-threaded) and, from LTE, the opacity residual stayed near 1 for more
than ten iterations while outer-layer O/C populations halved per step
(`results/hot-daz/coldstart-40-fast`, stopped at iteration 11). With the
changes above an iteration takes 3.5--4 min after ~8 min of setup
(`results/hot-daz/coldstart-40-fast-v2`). Its checkpoint spectra, compared
with the warm-started solution converged to an opacity residual of 9e-5:

| Iteration | max \|dF/F\| 1150--1800 Å | 99th percentile | max \|dF/F\| 910--1150 Å | C III 1175.99 / 1176.37 EW (mÅ) |
|---|---|---|---|---|
| 1 | 3.6 | 0.24 | -- | 37.85 / 41.97 |
| 10 | 2.7e-3 | 3.8e-4 | 2.6e-2 | 17.129 / 20.030 |
| 20 | 1.7e-3 | 1.2e-4 | 3.8e-3 | 17.108 / 20.006 |
| converged | -- | -- | -- | 17.108 / 20.006 |

By iteration 20 the emergent spectrum agrees to 0.2% everywhere, while the
maximum-norm opacity criterion (largest relative change of absorption or
emissivity over all 47 million wavelength-depth cells) was still ~0.08:
it is dominated by cells that do not affect the emergent spectrum.
`convergence_criterion="flux"` therefore tests the observable directly:
the undamped proposed populations may change the emergent flux by at most
`tolerance` at every population wavelength in `flux_wavelength_range`
(default 900--1e5 Å), gated like the opacity residual. On the small
two-element test problem a 1e-3 flux tolerance converges in 19 instead of
31 iterations, with emergent fluxes within 1.7e-4 of the 1e-6 opacity
solution (`tests/test_trace_metal_speedups.py`).

### Production-resolution cold start (2026-10-04/05)

Single-threaded wall-clock times, one job on the machine:

| Step | Configuration | Time |
|---|---|---|
| H/He host, public cold start | 80 depths, 4 angles, NLTE fractions 0, 0.1, ..., 1 (67 Newton iterations) | 5.30 h |
| H/He host, adaptive continuation | `research/run_hot_adaptive_continuation.py`: fractions 0, 0.25, 0.5, 1 (39 iterations); same atmosphere to 1.4e-6 in T, 9e-7 in flux | 3.03 h |
| Metals, flux tolerance 1e-3 | 9 elements, Fe/Ni IV--VII + VIII ground (`--iron-group-top-charge 7`, CHIANTI 11.0.2 `fe_8`/`ni_8`), 3804 levels, 1.28e6 rate points | 4.82 h |

The metal run (`results/hot-daz/coldstart-80-flux`) needed 25 min of setup
and ~6.3 min per iteration, ~15 min once the undamped flux test ran every
iteration. It converged in 29 iterations; the undamped flux change stayed
at 1.2--1.6e-3 from iteration 22 to 28. Between iterations 10 and 20 the
emergent spectrum above 1150 A changed by at most 0.8% (0.1% at the 99th
percentile). A 3e-3 tolerance would have stopped at iteration 22 (about
3.1 h), so `FLUX_TOLERANCE = 3e-3` is now the default for
`convergence_criterion="flux"`. The 80-depth host has a 23 GB peak memory
and the metal run 24 GB; they cannot run concurrently on a 48 GB machine.
The 80-depth spectrum differs from the 40-depth model by 0.35% (median).

### Reproduction

```bash
python research/explore_hot_composition.py --mode all \
  --seed results/hot-daz/irongroup-converged \
  --host-directory results/hot-daz/g191-b2b-recertified-v2 \
  --kurucz-positions results/hot-daz/kurucz-pos \
  --chianti-recombination results/hot-daz/chianti-recombination \
  --accelerated-lambda --convergence-criterion opacity --cold-start \
  --iterations 60 --tolerance 1e-3 --acceleration-depth 16 \
  --ionwise-acceleration --checkpoint-every 10 \
  --output results/hot-daz/coldstart-40-fast-v2
```

The seed supplies only the host directory and atom sizes with
`--cold-start`. The host `g191-b2b-recertified-v2` was re-solved from the
same checkpoint after the C kernels changed; its atmosphere, populations
and spectrum are bit-identical to `g191-b2b-recertified`.

### Population-criterion limits and the ZTF J1539 grid (2026-10-05)

The 80-depth ZTF J1539 grid (`ZTFJ1539_fitting/joint-binary-fit/ni-cleaned-mcmc-80depth`,
C and Si only) used the relative-population criterion. Four hot nodes
(t45900-g8.35, t51900-g7.55/g7.95/g8.35) exhausted up to three 64-iteration
attempts at defects of 0.01--0.12. The limiting levels held 1e-13 to 1e-6 of
the carbon, mostly the C VI ground in the outermost layers, which oscillate
under acceleration but form no observed line. A single unaccelerated SE step,
used as an independent check, also overstated the remaining error where plain
iteration overshoots (t51900-g7.55: 1.36e-3 against 5.7e-4 for the same state).

The population test now ignores levels holding less than
`POPULATION_DEFECT_FLOOR = 1e-6` of the element at each depth, and layers
above `POPULATION_DEFECT_MINIMUM_TAU = 1e-6` (Rosseland). Re-evaluated from the
stalled states, t45900-g8.35 and t51900-g7.95 give 1.5e-4 and 5.1e-4, while
the genuinely unconverged t51900-g8.35 (a Si V level holding 2.3e-7 of Si at
tau ~ 4e-4) and t51900-g7.55 (Si VI holding 1.4e-5 of Si, depth 42) are still
flagged. A floor alone is not enough: at 1e-5 the outermost C VI ground still
gives 2--6e-3.

With the flux criterion (default tolerance 3e-3), t45900-g8.35 converged from
the original 24-depth seed in 10 iterations (3.7 min); its spectrum matches the
pipeline's accepted response to 0.6% at worst, with identical C III/Si III/Si IV
equivalent widths. `research/ztfj1539_metals.py` now defaults to the flux
criterion; the rerun of the full grid with that criterion, no unaccelerated-SE
gate and a 25-iteration host budget is in
`ZTFJ1539_fitting/joint-binary-fit/ni-cleaned-mcmc-80depth-flux`.
