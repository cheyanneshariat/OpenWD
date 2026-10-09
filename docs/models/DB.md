# DB module

[Model guide](README.md) · [Getting started](../getting-started.md)

[Precomputed DB grids and cold-model comparison](../grids/db-cold-2026-10-09/README.md):
a verified 5,000 K supplement adds three accepted parameter points at log g 7,
7.5 and 7.75, with log g 8 repeated for comparison. Current selected DB coverage
is 492/638. These experimental dense-He models retain their numerical qualification
and unresolved physical/depth/interpolation limits.

`compute_db` solves a homogeneous pure-helium LTE atmosphere. It combines a
Hummer--Mihalas/Q-MHD helium EOS, He I/II/III continuum opacity, corrected
Doppler-convolved Beauchamp25-LD He I Stark profiles, Schoening/SYNSPEC He II
profiles, Unsold neutral-He broadening, and ML2/alpha=1.25 convection.

```bash
python examples/one_shot_db.py --teff 20000 --logg 8.0 \
  --quality standard --output results/db-20000-8.0
```

Typical standard calculations take roughly 3--10 minutes on a current laptop;
cool neutral-line models can be slower. No previous atmosphere is needed.

## Paper comparisons

The paper compares pure-He models with the one-dimensional ML2/alpha=1.25
Montreal/Tremblay grid of
[Cukanovaite et al. (2021)](https://doi.org/10.1093/mnras/staa3684).
The six points are 10000, 16000, 22000 and 30000 K at log g = 8, plus log g
= 7 and 9 at 22000 K. OpenWD relaxes its own helium atmosphere at each point
and synthesizes the spectrum using the corrected B25 He I profiles and
neutral-He broadening. The reference grid is used only for comparison.

[![DB surface-flux spectra compared with the Montreal grid](../assets/db-paper-grid.png)](../assets/db-paper-grid.pdf)

[Download the grid comparison (PDF)](../assets/db-paper-grid.pdf).
The left panels cover 900--30000 Å and the right panels show optical helium
lines. The curves retain their absolute surface-flux scale; no continuum
factor is applied to make the displayed spectra agree.

The second figure uses five SPY/UVES DBs at the fixed temperatures and
gravities of [Voss et al. (2007)](https://doi.org/10.1051/0004-6361:20077285),
from 11002 to 27288 K. An OpenWD atmosphere is calculated at each star's
parameters, and the Montreal grid is bilinearly interpolated to the same
point. Both models are convolved to `R = 18500`; data and predictions use
the same local continuum-fitting procedure in the stellar rest frame.

[![Observed SPY He I profiles with Montreal and OpenWD predictions](../assets/db-paper-spy.png)](../assets/db-paper-spy.pdf)

[Download the SPY comparison (PDF)](../assets/db-paper-spy.pdf).
The sample follows the growth and subsequent weakening of the optical He I
lines. Noise, normalization and line-profile differences remain visible.
These atomic-helium comparisons do not test the cool dense-neutral extension
below, and normalized line agreement does not establish an absolute flux scale.

## Cool helium

Use `run_model(DBConfig(...), output_directory)` to select the experimental
dense-neutral helium treatment when indicated by the local material screen.
It combines the tabulated bulk EOS with approximate chemical potentials and
trace-ion chemistry, without inserting that closure into warm ionized helium.
The command-line example uses the same automatic selection as `run_model`.
The lower-level `compute_db` preset remains an explicit atomic-physics interface.
The dense driver propagates the requested `logg` through hydrostatic column mass,
ML2 initialization, the structure solver and the independent spectrum audit.
Saved structures record their gravity, and the audit rejects a mismatch with
the request. The production recipe and its numerical tolerances are unchanged.
The material-domain checks still apply at every depth; accepting a parameter is
not a claim that its atmosphere or spectrum has been qualified.

Protected cold starts cover the established prescription at 10000 and 22000 K
and the dense workflow at 5000 and 8000 K. See [tested points](../tested-temperature-ranges.md),
[cool-model setup](../getting-started.md#cool-helium-and-mixed-atmospheres), and
[physical limitations](../limitations.md#physical-approximations).

## Flux conservation (2026-10-01)

Two numerics settings are on by default:

- `photospheric_depth_concentration=1` concentrates the structure depths
  across 0.01 < tau < 10 at an unchanged point count.
- `synthesis_transfer_depth_refinement=4` subdivides each depth interval for
  the final formal solution.

Before, the 40-point standard structures emitted up to 2-3% more than
sigma Teff^4. The bare-grid formal solution partly cancelled this, so the
totals looked right while the structure was not. Standard-quality totals
are now within about 0.75% (see the [DZ guide](DZ.md) for the method). Setting
both to 0 and 1 restores the previous numerics.

## Dense spectrum reconstruction

The experimental mass-conservative dense-He spectrum uses the existing
positive-intensity reconstruction option in both its coupled transfer solve
and independent prescribed-source check. This avoids tiny negative source
values caused by numerical cancellation in faint radiation fields. It does
not clip negative values, normalize the flux, change the atmosphere, or relax
the qualification limits. The selected transfer option is saved in the
metadata and retained by the independent spectrum audit. Older saved runs
without that flag retain their previous audit option.

The regression fixture is one transfer row from a 5,000 K/log g 7 development
calculation. It checks the transfer operation. Dense DB runs preserve the
requested gravity as described in [Cool helium](#cool-helium); individual
parameter points still require atmosphere and spectrum qualification.

## A tested warm-model setting

At 27,000 K/log g 9.25, a fresh production calculation with the default
`photospheric_depth_concentration=1` failed the temperature-stationarity check.
Changing this existing setting to 2 passed all five required structure checks
on the same unchanged source, with the same physics and acceptance limits:

```python
from wd_spectra.models import DBConfig, run_model

result = run_model(
    DBConfig(
        effective_temperature=27000,
        logg=9.25,
        quality="production",
        photospheric_depth_concentration=2,
    ),
    "results/db-27000-9.25-depth2",  # fresh output directory
    require_convergence=True,
)
```

Both runs used 80 layers. The solved atmospheres had 37 and 43 layers,
respectively, between Rosseland optical depths 0.01 and 10. The measured
unrestricted log-temperature correction decreased from 0.00299 to
0.00000145; the acceptance limit stayed at 0.0003. The maximum optical
surface-flux difference was 0.1504% over 3,000–10,000 Å, without normalization.
The failed default result remains an unfinished reference despite that small
spectral difference.

[![Warm DB layer placement and convergence before/after](../assets/db-depth-concentration-before-after.png)](../assets/db-depth-concentration-before-after.pdf)

[Download the before/after figure (PDF)](../assets/db-depth-concentration-before-after.pdf).
The upper panels show temperature versus column mass and differences at the
same column mass. The latter use linear interpolation in log column mass
within the shared domain. The lower panels show the actual layer placement
and three residuals divided by their unchanged acceptance limits. Values below
one pass those displayed checks. Source closure and lower-boundary screening
also pass in both runs; the [evidence record](../assets/db-depth-concentration-evidence.json)
retains all five checks and the two exact configurations.

This is a demonstrated numerical setting for one model, not a new default or
automatic retry policy. Independent depth convergence and physical accuracy
at these parameters remain unverified. The opacity, equation of state,
convection prescription, solver limits, and production point count are unchanged.
