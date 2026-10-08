# DAZ: polluted hydrogen atmospheres

[Model guide](README.md) · [Getting started](../getting-started.md)

Use `DAZConfig` for a hydrogen-dominated atmosphere containing metals. Unlike
DZ/DBZ, its host EOS, hydrogen opacity, and convection are those of DA. Metals
participate in the common electron-density/charge-neutrality solution and in
the opacity used to solve the atmosphere, not only in the final spectrum.

```python
from wd_spectra import DAZConfig, run_model

run = run_model(
    DAZConfig(effective_temperature=11_820, logg=8.40, quality="standard"),
    "results/daz-g29-38",
)
print(run.convergence_verified)
```

The default abundances describe G29-38. Supply `abundances={"Ca": -8.0, ...}`
to replace the entire mixture; values are **log10 N(element)/N(H)**, not ratios
to helium. All required standard atomic and profile data are bundled. A new
run starts from scratch, and `require_convergence=True` requires all the
current atmosphere-certificate gates. Unqualified exploratory spectra warn.

The equivalent command is:

```bash
python examples/one_shot_daz.py --teff 11820 --logg 8.40 \
  --quality standard --output results/daz-example
```

`compute_daz` is also available for in-memory calculations. The automatic
interface never routes DAZ through a helium or pure-DA substitute on failure.

## Paper comparison

The paper's seven-object polluted-star figure contains two hydrogen-host
examples: **G149-28** (8600 K, log g = 8.10; DESI DR1) and
**GALEX J1931+0117** (20890 K, log g = 7.90; VLT/UVES), in the first and
last panels. Their parameters and metal abundances are fixed to
[Zuckerman et al. (2011)](https://doi.org/10.1088/0004-637X/739/2/101)
and [Vennes et al. (2011)](https://doi.org/10.1111/j.1365-2966.2011.18323.x),
respectively. The five middle panels use the helium-host
[DZ/DBZ module](DZ.md#paper-comparison).

[![Seven polluted white dwarfs, including the G149-28 and GALEX J1931 DAZ models](../assets/dz-daz-paper.png)](../assets/dz-daz-paper.pdf)

[Download the comparison (PDF)](../assets/dz-daz-paper.pdf).
Each DAZ atmosphere is relaxed at the displayed composition with metals
included in charge balance and structural opacity, then synthesized with the
more detailed metal line list. Number abundances in these two panels are
relative to hydrogen; the middle panels use helium. The plotted DAZ models
retain the paper's Stout strengths, Unsold neutral-H metal-line widths and
Stark-only Lyman setting described below.

For display, predictions are convolved to `R = 2000` for G149-28 and
`R = 40970` for GALEX J1931+0117. Observations and models are independently
pseudo-continuum normalized on rest-frame vacuum wavelengths. This compares
Balmer and metal-line shapes without fitting temperature, gravity or
abundances; it does not test absolute flux calibration. Remaining differences,
including Ca II cores and metal-line strengths, are visible in the panels.

## Physics and paper comparisons

The preset uses the shared adaptive DA solver, ML2/alpha=0.7, the DA hydrogen
line/molecular policy, Stout metal lines through charge 3, and Verner
photoionization. It retains the established Feautrier atmosphere solver and
formal-integral final synthesis.

As in DZ, the emitted spectrum is flux conserving. Before, the structure used
up to 8000 line-centered lines while the synthesis used 20,000, and the formal
solution ran on the bare structure depths. Totals were 0.984, 0.991 and 1.040
of sigma Teff^4 for G149-28, G29-38 and GALEX J1931+0117. They are now
1.005, 1.003 and 1.001. Three settings do this:

- The structure absorbs the synthesis line list (f >= 1e-4), opacity-sampled
  at R = 1000 (`structure_opacity_sampling_resolution`).
- Its depths are concentrated across the photosphere
  (`photospheric_depth_concentration=1`).
- The formal solution subdivides each depth interval four times
  (`synthesis_transfer_depth_refinement=4`).

Standard cold starts for these three objects took 220, 243 and 295 s. The
previous settings took 165, 267 and 771 s, so the heavily polluted GALEX
J1931+0117 is now much faster. `structure_opacity_sampling_resolution=None`,
`photospheric_depth_concentration=0` and
`synthesis_transfer_depth_refinement=1` restore the previous numerics.
Dense-helium ionization corrections and helium-perturber profiles are
not applied to hydrogen hosts.

Stout strengths and Unsold neutral-H metal-line widths preserve the paper's
metal-line prescription. The later `strong_line_atomic_data="nist-asd"` and
`metal_neutral_h_broadening="barklem"` options are explicit alternatives;
neither is selected by target name. Hydrogen Balmer self-broadening is a
separate prescription and follows the DA policy.

The fixed-atmosphere regression controls include G149-28 (8600 K) and GALEX
J1931+0117 (20890 K) from the paper's metal-polluted-star figure. These tests
use the paper's Stark-only Lyman setting (`lyman_profile_source="stark"`);
new calculations default to the DA Allard policy. Fixed-state comparisons
protect the plotted synthetic spectra, not cold-start convergence. See
[tested points](../tested-temperature-ranges.md) for the latter.

Fresh standard-resolution checks of G149-28, G29-38, and GALEX J1931+0117 use
`research/validate_daz_cold_start.py` and require all five physical atmosphere
gates. These are individual composition/gravity points, not a qualified
temperature or abundance grid. Atmosphere convergence does not certify depth
independence, exact reproduction of a paper atmosphere, or the integral of a
separately sampled final spectrum.

Metals remain trace contributors in the thermodynamic derivatives used by
ML2. Ca II H and K can use complete-redistribution source functions in the
final spectrum while the atmospheric populations and extinction remain LTE;
this is not a full metal-NLTE atmosphere. The shared metal-opacity revisions
of the 2026-09-30 audit (exact frequency Voigt profiles, `1/Z^2` Unsold
radii, structure line identity) also apply to DAZ models; see the
[DZ guide](DZ.md). A low-temperature pure-DA validation does not establish
the same validity range for arbitrary metal abundances.
