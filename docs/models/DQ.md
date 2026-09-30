# DQ: refractive helium/carbon atmospheres

[Model guide](README.md) · [Getting started](../getting-started.md)

`DQConfig` predicts hydrogen-free, nonmagnetic, helium-dominated atmospheres
with trace carbon and C₂. The public calculation starts from its own
hydrostatic gray seed. It does not read a previous atmosphere, model spectrum,
observed spectrum, object-name lookup, or neighboring grid model.

## Run a model

Install OpenWD normally; the necessary constitutive tables are bundled. Model
generation performs no downloads. The DQ runtime requires NumPy, SciPy and
Numba, installed as package dependencies.

```python
from wd_spectra import DQConfig, run_model

run = run_model(
    DQConfig(
        effective_temperature=9347,
        logg=8.041,
        log_carbon_to_helium=-4.107,  # log10 N(C nuclei)/N(He nuclei)
        maximum_seconds=28800,
    ),
    "results/my-dq",               # must be a new directory
    require_convergence=True,
)
```

These are inputs, not fitting starting guesses: OpenWD does not adjust them.
Arbitrary finite positive temperatures and finite gravities are accepted,
with `-12 <= log_carbon_to_helium <= -3`; this is not a guarantee of
convergence or data-table coverage everywhere in that range. Only
`quality="standard"` is exposed for this protocol. Hot carbon-dominated DQs,
hydrogen-bearing DQs and DQp molecular distortion are outside its scope.

For an in-memory result and explicit diagnostics directory:

```python
from wd_spectra import compute_dq

result = compute_dq(
    DQConfig(9347, 8.041, -4.107),
    output_directory="results/my-dq-direct",
)
```

The returned spectrum is unscaled surface `F_lambda` in
`erg s^-1 cm^-2 Angstrom^-1`, on a vacuum-wavelength grid. An optional
`wavelength=` array affects only output synthesis; it cannot reduce the
structure sampling or bypass the independent flux test.

Calculations can take hours and several GiB. Each request runs in an isolated
worker using one numerical thread, so its process-local DQ solver adapters do
not modify another spectral class in the caller. Run separate objects as
separate requests, allowing for their memory requirements. Progress is printed
and written to `worker/progress.json` and `worker/automatic-phases.json` under
`run_model` (directly in the chosen directory under `compute_dq`).

## Paper comparison

The paper compares four SDSS stars at published temperature, gravity and
carbon abundance, without refitting. J0940 uses
[Koester & Kepler (2019)](https://doi.org/10.1051/0004-6361/201935946);
the other three use
[Blouin & Dufour (2019)](https://doi.org/10.1093/mnras/stz2915).

| Object | Teff (K) | log g | log10 N(C)/N(He) | Initialization of the plotted model |
| --- | ---: | ---: | ---: | --- |
| SDSS J0940+0210 | 7256 | 8.094 | -5.769 | Re-relaxed saved atmosphere |
| SDSS J1225+4706 | 6294 | 7.924 | -5.33 | Public cold start |
| SDSS J0804+1714 | 5364 | 7.885 | -7.19 | Re-relaxed saved atmosphere |
| SDSS J1803+2320 | 4400 | 7.694 | -7.79 | Re-relaxed saved atmosphere |

[![SDSS DQ spectra and predictions scaled using published masses and parallaxes](../assets/dq-paper.png)](../assets/dq-paper.pdf)

[Download the comparison (PDF)](../assets/dq-paper.pdf).
The models combine the dense-He EOS, coupled He/C/C2 chemistry, completed
Swan and C-A opacity, and refractive transfer described below. The saved
research starting states were re-relaxed with the paper's stride-four
structure sampling; all four plotted results passed the atmosphere checks
and the independent 0.2% bolometric-flux test. The warm-start rows are
spectral comparisons, not evidence of public cold-start convergence at those
points. Public `run_model` still starts from scratch.

Surface flux is converted to flux at Earth using `(R/d)^2`, with
`R^2 = G M / g` and the published masses and parallaxes. No flux factor is
fitted. Predictions are convolved with the wavelength-dependent SDSS
line-spread function, and both spectra are averaged in the same 5 Å bins
after invalid pixels are masked. Shading is the propagated pipeline
one-sigma error, without inter-pixel covariance. No extra velocity or
extinction correction is applied to the SDSS vacuum wavelengths.

The Swan-band pattern is recognizable, but continuum and band-profile
discrepancies grow toward the coolest object. J0804 and J1803 are peculiar
DQs: their inclusion exposes the limitations of the molecular profiles and
does not extend the module's qualified scope to DQp stars.

## What is required for success

The same shared nonlinear solver, ML2 helpers and screened-domain controller
are used with DQ material/coordinate/transfer adapters. The current protocol
uses current-state energy normalization, EOS-corner-aware thermal proposals,
automatic between-phase coordinate refreshes, and one-cell lower-boundary
extensions. Pseudo-time conditioning is only an initializer; its looser
transient contraction target is not a final atmosphere tolerance.

A successful public request must pass all five atmosphere checks: all-depth
flux, local energy balance, temperature stationarity, source closure, and
lower-boundary screening. It must then pass an independent **218520-point**
synthesis: 4000 logarithmic points from 1000 to 100000 Å, 150000 optical
midpoints at 0.02 Å spacing between 3800 and 6800 Å, and 64521 logarithmic
infrared nodes from 6800 to 100000 Å (R≈24000; one duplicate endpoint).
The added infrared grid resolves molecular structure that the older
154000-point qualification grid undersampled. All fluxes must be finite
and positive, and `abs(F_bol/(sigma Teff^4) - 1) <= 0.002`. No flux
normalization or parameter adjustment is applied. Source integrity and data
checksums are recorded.

Failed checks or exhausted budgets raise an error. Accepted diagnostic states
and any intermediate products remain available; an atmosphere certificate
alone does not cause an unqualified spectrum to be returned as successful.
The public API deliberately has no saved-state restart mode.

## Physics and limitations

The implementation includes He-REOS.3 host thermodynamics, coupled He/C/C₂
chemistry, dense-helium continuum corrections, and conservative refractive
transfer with bent and trapped rays, scattering, and a screened lower
boundary. Structure and synthesis use the same carbon-line declaration and
molecular opacity. UV sampling is determined without reference to a past
spectrum.

Carbon UV resonance-wing support is decided from the complete atmospheric
column before opacity work is divided into depth subsets. The same decision
is used for cached columns and temperature probes, with cache invalidation
if it changes. Carbon-line evaluation batches the union of retained lines
while preserving the conservative opacity-screening bound. Non-DQ callers
retain the shared line evaluator's original default behavior.

Resolved thermal Swan lines retain the 29,004 Hornkohl/Parigger lines, with one absolute
Brooke band-rate calibration, and add 1,232,442 ExoMol transitions outside their
per-band vibrational/angular-momentum coverage. No observed spectrum sets the
combination weights, and holes inside the supplied coverage envelope are not filled.
The density shift is integrated
over each depth cell; this is spatial quadrature, not extra collision
broadening. Other C₂ systems and the partition function use ExoMol 8states.
That list does **not** include the C¹Πg–A¹Πu Deslandres–d'Azambuja
system. DQs additionally include that system in both structure and synthesis,
using Lino da Silva's (2024) published Einstein coefficients for the same 63
bands as the previous historical estimate. Historical band origins and ExoMol
lower-A-state populations with the unchanged 8states partition are retained.
No strength multiplier is fitted to stellar observations. This is a finite-bin
rigid-rotor envelope, **not a modern resolved C–A line list**. High-v
perturbations and detailed rotational factors remain uncertain. The source is
a workshop presentation, not a fully validated modern line list. No unvalidated
C–A pressure shift is applied. The older Cooper/Nicholls table is bundled for
reproducibility, but is no longer selected by the default.

An exact-temperature LRU caches Swan strengths and cumulative sums, without
rounding temperatures, pruning lines, or changing arithmetic order. The cache
has an explicit 1 GiB cap for synthesis, reduced to 32 MiB during structure
iteration to leave room for response matrices; the overall
4 GiB process guard remains unchanged. This recovers much of the larger-list
synthesis cost, but does not promise faster cold convergence.
DQ helium-continuum work is evaluated in 1,024-wavelength batches using the
unchanged shared formula. This bounds temporary allocations and is tested
bit-for-bit against a full-grid call; it does not change the wavelength grid.

`DQConfig(include_c2_ca=False)` explicitly omits this contribution for
diagnostic A/B comparisons; it does not relax any qualification check or
change the Swan contribution. The isolated worker records the selection.
The C–A file is only constitutive opacity data, not an observed or fitted
spectrum. From a source checkout, rebuild the data using the audited sources:

```sh
PYTHONPATH=src python tools/build_dq_ca_2024.py \
  --states /path/to/12C2__8states.states.bz2 \
  --pdf /path/to/C2-deslandres-dazambuja.pdf --output results/ca-2024-rebuild
PYTHONPATH=src python tools/build_dq_swan_completed.py \
  --exomol /path/to/audited-exomol-branches.npz --output results/swan-rebuild
```

Structure quadrature first combines the atomic/molecular nodes with the
4000-point logarithmic continuum mesh over 1000–100000 Å, including the
infrared, then retains every fourth node plus both endpoints. This default
reduces structure cost; it does not remove opacity sources or thin the final
218,520-point independent spectrum. The same rule is applied afresh after
each automatic lower-domain extension. All five equilibrium gates and the
independent 0.2% luminosity gate remain mandatory, without flux rescaling.

This is an accepted speed/accuracy tradeoff, **not a guaranteed 1% spectral
error bound**. In fixed-parameter comparisons with the former dense structure
grid, J1225 (6294 K) completed from scratch in 17.6 minutes, with maximum
optical differences of 1.49% native and 0.29% after 3 Å FWHM smoothing.
J1311 (5529 K) stalled from scratch and was stopped at 27.5 minutes; a
separate converged warm diagnostic showed 8.13% native and 1.29% at 3 Å.
That warm diagnostic is not a cold timing result. Both converged spectra
passed the luminosity gate, which does not constrain individual line errors.
No paired cold speedup factor is established for these two objects. See the
[sampling release record](../development/history/dq-stride4-default-2026-09-18.md)
for validation and limitations.

Transfer and analytic temperature-response inner loops use allocation-free
scalar arithmetic, tested bitwise against the previous kernels. A roughly
3× response-kernel benchmark is not a claim of 3× end-to-end acceleration.
The continuum table and all molecular data are checksum-pinned; their
provenance is in the bundled data and [third-party notices](../../THIRD_PARTY_NOTICES.md).

This follows important ingredients of the Blouin approach but is not an exact
reproduction. Dense-mixture EOS, molecular collision profiles and the grey
refractive ML2 bridge remain approximations. Numerical convergence does not
guarantee observational agreement, abundance accuracy, or independent depth
convergence. These limitations are recorded in every result rather than
silently marked as validated.

In particular, the bolometric constraint fixes the integral of the emergent
spectrum, not its wavelength distribution. Refining a saved atmosphere
without restoring equilibrium measures a discretization residual, not the
error in the final converged luminosity. Spectral depth convergence requires
reconverging both meshes at the same stellar parameters and comparing their
spectra; neither a flux renormalization nor an unreconverged flux difference
is a substitute for that test.

See the [2024 C–A and completed-Swan default record](../development/history/dq-completed-default-2026-09-17.md),
the [historical C–A validation record](../development/history/dq-ca-default-2026-09-17.md)
and the [original release record](../development/history/dq-release-2026-09-17.md)
for tests and the distinction between research warm starts and packaged
cold-start qualification.

## Regression tests

The normal fast test suite covers chemistry, refractive conservation and
responses, controller transitions, and public cold-start/qualification
contracts. The separate fixed-atmosphere spectrum test is included in
`python tools/validate.py spectra --case dq-j1235`; this is not cold-start
evidence. To run the expensive public cold canary explicitly:

```sh
python tools/validate.py cold --case dq-j1235 --jobs 1 --timeout 28800 --output results/dq-cold-check
```

This test fixes the J1235 parameters, supplies no atmosphere or spectrum,
and retains its worker outputs in the validation artifacts. It is also part
of the scheduled/manual protected-canary workflow, not the fast push tests.
