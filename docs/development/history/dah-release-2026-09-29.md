# DAH paper prescription and release integration

[Development guide](../README.md) · [DAH user guide](../../models/DAH.md)

The public DAH default now reproduces the normalized Kurucz/Griem scalar
prescription used in the eight-object paper comparison. The earlier work
used temporary, process-global research patches to select the line profile
and normalize H2db strengths; the package now expresses both choices in
`DAHConfig` and `MagneticPhysics`, without patching another calculation.

## Default and compatibility

`atmosphere_structure="nonmagnetic"`, `balmer_profile="kurucz-griem"`,
`normalize_balmer_strength=True` and `polarized_transfer="scalar-stokes-i"`
are the defaults. Magnetic EOS, RWA photoionization, centered motion and
cyclotron absorption are off. The public request fingerprint has DAH
physics revision `dah-kurucz-griem-v4`. Low-level magnetic primitives retain
their explicit unified-profile defaults. The guide gives a complete
configuration for the previous mean-field magnetic calculation.

Both line-shape area normalization and H2db component normalization are
independent of the requested wavelength window. Component normalization
includes stimulated emission before ray-angle weighting. No stellar
parameters, widths, oscillator-strength multipliers or observed flux scales
were fitted during promotion. The output remains physical surface flux.
The equilibrium certificate describes the underlying nonmagnetic DA
atmosphere; magnetic synthesis is post-processing of that structure.

## Radiative convergence repair

Convection-free starts previously entered a convective-gradient conditioner
and deferred local-energy constraints. Flux rows in optically thin layers
can be nearly dependent, leaving their temperatures poorly constrained.
Radiative starts now enforce flux and local energy immediately with a fresh,
unregularized, equilibrated Newton tangent and frozen equation weights.
Broyden updates are disabled across changed energy-equation weights. Warm
starts that already bypass convective conditioning use the same fresh energy
tangents. Their old formal-flux continuation count is no longer meaningful;
the restart regression now checks the actual energy phase, rebuilt Jacobian
and rejection of an unfinished state by the physical certificate.

The 6680-K, log-g=7.96 G 76−48 atmosphere also requires more resolution at
its steep ionization transition. A failed production 100-layer DA attempt
can seed a 200-layer solve of the same physics, with a fresh certificate.
This numerical refinement does not alter stellar parameters, chemistry,
opacity, convection or physical tolerances. An unresolved fine model still
warns and fails mandatory-convergence workflows.

## Controls

- Physical unit checks cover the historical constants, positivity, symmetry,
  density scaling, Holtsmark tail and independently integrated profile area.
- Component-integral checks span 0.1–437.1 MG, including the low-field H2db
  continuation, and verify that normalized calls do not alter later calls.
- Eight immutable paper spectra test the public default on fixed input
  atmospheres, with structure iteration forbidden and no fitted rescaling.
  Their original source/checkpoint/spectrum hashes are preserved in the
  [fixture manifest](../../../tests/data/dah_paper/manifest.json).
- Independent public cold starts at J1007+1237 and J1254+5612 require all
  five equilibrium gates and absolute spectra within 0.2% of those controls.
  The latter has a local field maximum above 100 MG.
- A separate G 76−48 radiative DA canary protects the convergence repair.
  CLI dispatch uses `run_model`, including `--require-convergence`.
- The full validation plan and GitHub cold matrix include these cases and
  retain all existing DA/DB/DAB/DZ/DAZ/DQ/DO/DAO/PG1159/D6 controls.

The eight fixed-state comparisons establish reproduction of the accepted
paper curves, not magnetic radiative equilibrium or a fully validated
parameter grid. The smooth correction in one published figure variant
remains a comparison operation and is absent from the physical prediction.

## Qualification evidence

The [local evidence summary](dah-release-evidence-2026-09-29.json) records
commit `1a53fd4`, the validation input hash, and the completed cases from
`results/dah-release-final-v2`. Numerical inputs stayed unchanged during
that run.

- Fast and component checks: **1710 passed**, with one existing optional
  Koester-spectrum check skipped because its cached observation was absent.
- Fixed-atmosphere spectra: **21 passed**, including all eight paper objects
  at relative tolerance `2e-8`, with no fitted scales or parameter changes.
- Cold starts: **four passed** — J1007+1237, J1254+5612, radiative G 76−48,
  and the existing DZ PG1225 control. The two DAH spectra differ from the
  paper controls by at most `3.05e-14` and `2.27e-5`, respectively, in
  relative flux. Both pass all five required equilibrium checks on the
  underlying nonmagnetic structure. G 76−48 passes on the refined 200-layer
  grid with convection disabled.
- An isolated wheel built from numerical commit `2935f3a` passed **27 checks
  in each of the pure-Python and native installations**, including the
  J1007 paper spectrum, CLI, profiles, and bundled H2db data. Subsequent
  commits change only test bookkeeping and documentation.

The remaining local cold starts were intentionally interrupted or left
unstarted when validation moved to GitHub Actions. Accordingly, the local
runner records a failed overall status and `full_qualification=false`;
this is **not** a completed full-suite pass. The release PR requests the
complete GitHub matrix with the `full-validation` label, including all
25 cold cases and fresh fast/spectral preflight checks. Its GitHub
`qualification` check remains the release gate.
