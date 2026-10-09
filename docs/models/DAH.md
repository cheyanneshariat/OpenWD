# DAH module

[Model guide](README.md) · [Getting started](../getting-started.md)

`DAHConfig` predicts disk-integrated optical spectra of magnetic,
pure-hydrogen white dwarfs. The default is the **normalized Kurucz/Griem
prescription used in the paper**: a nonmagnetic DA atmosphere, scalar
Stokes-I magnetic transfer, and the ordinary hydrogen EOS and continuum.
Convection is suppressed above a projected-area mean field of 0.05 MG.
The output covers vacuum wavelengths from 3400 to 8000 Å and is surface
`F_lambda` in erg s⁻¹ cm⁻² Å⁻¹.

## Quick start

This example uses the fixed published geometry of J1007+1237, whose mean
visible field is about 5.20 MG (the dipole polar parameter is 6.13 MG):

```python
import numpy as np
from wd_spectra import DAHConfig, run_model

config = DAHConfig(
    effective_temperature=18_687, logg=8.04,
    magnetic_field_megagauss=6.13,
    field_geometry="dipole", dipole_inclination_deg=71,
    dipole_offset_radius=(0.0, 0.0, 0.30),
    quality="production",
)
run = run_model(
    config, "results/j1007",  # choose a new directory
    wavelength=np.arange(3600.0, 7100.01, 0.5),
    require_convergence=True,
)
print(run.convergence_verified)
```

The atmosphere is computed from scratch. No downloaded model grid, research
patch, observed spectrum or previous atmosphere is required. To return a
`ModelResult` in memory, use `compute_dah(config, wavelength)`; inspect
`result.atmosphere.metadata["equilibrium_certificate"]` before using it.
`run_model` saves the configuration, spectrum, atmosphere and diagnostics
and can require a successful certificate.

The equivalent one-shot command is:

```sh
python examples/one_shot_dah.py --teff 18687 --logg 8.04 \
    --field-mg 6.13 --geometry dipole --inclination 71 --offset 0 0 0.30 \
    --quality production --require-convergence --output results/j1007-cli
```

`examples/dah_paper.py` accepts any of the eight paper object keys and holds
its temperature, gravity and geometry fixed. `quality="production"` is the
paper resolution (initially 100 depth points and four synthesis angles).
`standard` uses the ordinary smaller structure grid; `quick` is only a smoke
test and is not a convergence guarantee.

## Field geometry

| Input | Meaning |
| --- | --- |
| `field_geometry="uniform"` | One modulus over the disk; the default geometry. |
| `field_angle_deg=None` | Isotropic field directions for the uniform model; a number fixes the angle to the line of sight. |
| `field_geometry="dipole"` | Centered or displaced dipole integrated over the visible surface. |
| `magnetic_field_megagauss` | Uniform modulus, or polar field of the **undisplaced** dipole. |
| `field_strength_definition="visible-mean"` | Instead normalize a dipole to the projected-area mean modulus. |
| `dipole_inclination_deg` | Angle between the dipole axis and the line of sight. |
| `dipole_offset_radius=(ax, ay, az)` | Displacement in stellar radii in the magnetic-axis frame: z along the dipole axis, line of sight in the x–z plane. |
| `disk_field_bins=21` | Compress the resolved surface quadrature into field bins; `None` retains all cells. |
| `disk_component_drift_angstrom=None` | Opt-in dipole binning by estimated component drift; takes precedence over `disk_field_bins` (see below). |

The atomic regime is selected from a bound on the **continuous visible
surface maximum**, not a compressed bin mean or the polar input parameter.
Bounds are exact for axial offsets and conservative for transverse offsets;
the result records `visible_field_bounds_exact` and the field interval.
An offset dipole with a polar field below 100 MG can have local fields above
100 MG. The 100-MG scale is not a universal accuracy cutoff.

Each compressed bin is synthesized at one mean field. In the tested
high-field examples, components move rapidly between the default bins,
leaving discrete copies rather than a resolved field-spread profile.
Larger `disk_field_bins` values and `None` remain supported. They change
compression, not raw surface resolution; the new option also uses a denser
surface grid. Explicit dense grids are available at the lower level through
`dipole_surface_cells`.
`disk_component_drift_angstrom` builds bins from a 192 × 384 (or denser)
visible-surface grid. The bin edges lie at equal steps of the accumulated
drift ∫ max|dλ/dB| dB on a 4001-point field grid. The screen uses Balmer
components inside the output window (±100 Å) with at least 1% of their
parent line's strongest `energy × dipole-strength` proxy, not their
temperature-dependent LTE opacity. This is an estimated drift budget
within each interval, not a bound on adjacent bin means or spectral error.
Components outside the screen, broad wings, angular compression and finite
raw-surface resolution still require convergence checks. Requests needing
more than 4096 drift intervals fail rather than silently coarsen.

Each field interval also retains up to eight limb-cosine subgroups. Their
projected weights and mean limb cosines preserve disk integration when
component drift is negligible, including at zero field. The subgroups
share the interval's mean field, so they reuse its opacity and radiation
source calculation; their field–ray angles are averaged separately. The
result reports distinct field bins and limb rays separately.

With 16 Å, the archived J1018+0111 and J1351+5419 calculations use 130 and
267 field bins. These archived runs preceded the independent limb sampling;
their spectra and timings describe the earlier one-ray-per-field version.
The option is slower than the 21-bin default because every field bin needs
an opacity/source calculation and each limb subgroup needs an emergent ray.
See the
[disk-quadrature benchmark](../development/history/dah-disk-quadrature-2026-10-08.md)
for measured accuracy and cost. Start with 16 Å for these cases and tighten
it for your target and instrumental resolution; it is not a universal
accuracy setting. The option is off by default and does not change the
frozen paper predictions. `result.metadata["domain_notes"]` records the
scope of this numerical estimate and the unchanged physics approximations.

## Default physics

1. **Structure.** The shared adaptive Newton solver converges an ordinary
   nonmagnetic DA atmosphere. The zero-field DA opacities, including its
   usual unified Balmer profiles, determine the structure. The magnetic
   Kurucz/Griem spectrum is then computed on that temperature–pressure
   structure. The certificate establishes equilibrium of the **nonmagnetic
   structure**, not radiative equilibrium with magnetic opacities in every
   surface cell. This scope is explicit in the model metadata.
2. **Line shapes.** `balmer_profile="kurucz-griem"` implements the historical
   Kurucz (1970) / Griem Stark approximation (SAO Report 309, section 5.14;
   `STARK`, printed p. 243). Each parent-line shape is divided by its own
   infinite-frequency-detuning integral, independently of the output mesh,
   and multiplied by the ordinary LTE oscillator strength and HM/Q-MHD
   population and occupation factors. No width or opacity multiplier is
   fitted. The historical profile has no separate Doppler convolution,
   neutral self broadening or unbroadened central component.
3. **Magnetic components.** If every visible field is at most 1 MG, Hα–H22
   use normal Zeeman triplets, with Δν = eB/(4πm_e c). Otherwise Hα–H12 use
   H2db wavelengths and relative strengths, including Boltzmann populations
   of the split n=2 substates. H13 and higher are omitted in this regime.
   The reader restores the archived mirror components with the physical
   sign of m. Cells below the complete table's 0.0470103-MG floor use an
   analytic normal-triplet continuation.
4. **Line strength.** `normalize_balmer_strength=True` divides each H2db
   parent line's complete component weights by their depth-dependent sum,
   **including stimulated emission at each component frequency**, before
   angular weighting. The isotropic integrated strength equals that of the
   zero-field parent. The ray-specific integral need not be field independent.
   Components translate the parent profile in frequency; finite templates
   span 900–25000 Å and are never renormalized to the requested output window.
5. **Transfer.** `polarized_transfer="scalar-stokes-i"` integrates rays with
   their local π/σ angular weights and limb cosines. The coherent-scattering
   source is solved for each cell's angle-averaged opacity. Dipole quadrature
   starts at 8 limb × 16 azimuth nodes and refines to resolve displaced polar
   caps. Field-bin compression splits bins crossing 1 MG.
6. **Continuum and chemistry.** The defaults keep the nonmagnetic EOS,
   bound-free/free-free absorption and scattering. Molecular chemistry,
   H⁻ and H₃⁺ are included for weak-field models at or below 12000 K;
   the strong-field comparison prescription omits molecular chemistry.
   Magnetic EOS, RWA photoionization, centered-motion corrections and
   cyclotron absorption are disabled by default.

The normal convection policy uses ML2/α=0.7 below 0.05 MG and no convection
above it. An explicit positive `mixing_length_alpha` overrides this policy.
For a strictly radiative DA control at any field, use
`DAConfig(mixing_length_alpha=None)`. Cool radiative production atmospheres
can refine a failed 100-layer attempt to 200 layers with unchanged physics
and fresh convergence checks; see the [DA guide](DA.md#radiative-atmospheres-with-convection-disabled).
G 76−48 (6680 K, log g=7.96) motivated this repair. It does not establish
an entire cool radiative grid or remove its observed line-core discrepancies.

## Results at published parameters

The paper comparison uses these fixed values, ordered by dipole polar field.
The first four have continuous visible maxima below 100 MG; the remaining
objects test the prescription at higher fields.

| Object | Teff (K) | log g | Bp (MG) | i (deg) | (ax, ay, az) |
| --- | ---: | ---: | ---: | ---: | --- |
| J1007+1237 | 18687 | 8.04 | 6.13 | 71 | (0, 0, 0.30) |
| J1034+0327 | 15756 | 8.80 | 11.17 | 77 | (0, 0, 0.09) |
| J1154+0117 | 29316 | 8.90 | 35.53 | 87 | (0, 0, −0.23) |
| J2149−0728 | 22642 | 8.37 | 45.09 | 66 | (0, 0, 0.17) |
| J1254+5612 | 12870 | 8.58 | 60.43 | 62 | (0, 0, 0.18) |
| J1018+0111 | 10500 | 8.00 | 108.12 | 60 | (0, 0.07, 0.10) |
| J1351+5419 | 13937 | 8.43 | 368.52 | 34 | (0, 0, 0.07) |
| J2247+1456 | 19000 | 8.00 | 437.10 | 10 | (0, 0, −0.15) |

[![Observed and predicted spectra with a single flux scale](../assets/dah-paper-unscaled.png)](../assets/dah-paper-unscaled.pdf)

[Download the single-scale comparison (PDF)](../assets/dah-paper-unscaled.pdf)
or the [paper's smoothly rescaled comparison (PDF)](../assets/dah-paper-rescaled.pdf).

The figure above uses one flux scale per object. The
[smoothly rescaled version](../assets/dah-paper-rescaled.png) used in the
paper additionally applies the disclosed continuum correction; neither
version changes the stellar parameters. Observations are SDSS/BOSS spectra
and the Hardy et al. comparison data. J1007 uses SDSS plate 5328, MJD 55982,
fiber 66; its spectrum is not velocity shifted or dereddened. J1034 uses
the previously photometrically recalibrated observation.

The archived spectra underlying the figure are immutable controls in
[tests/data/dah_paper](../../tests/data/dah_paper/README.md), with original
source and spectrum hashes. Fixed-state tests exercise the public defaults
with atmosphere iteration forbidden. Separate cold-start canaries for
J1007 and J1254 create new structures and require all five equilibrium
gates as well as absolute-flux agreement with the paper controls.

```sh
python tools/validate.py spectra --case dah-j1007+1237
python tools/validate.py cold --case dah-j1007+1237 --case dah-j1254+5612
python tools/validate.py cold --case da-radiative-g76-48
```

Bundled observational controls are available through
`wd_spectra.validation.magnetic_da`; `research/validate_dah_observed.py`
runs their public models and saves comparisons. Its fitted velocity and
single 5200–6100 Å scale belong to the comparison, never the atmosphere or
spectrum calculation. Smooth rescaling used in one paper-figure variant is
also a plotting operation: a positive quadratic correction in log flux.
It can absorb flux-calibration error **or** a physical continuum error.
The unscaled variant is necessary for assessing continuum agreement.

## Other physics options

The previous coupled magnetic treatment remains explicitly selectable:

```python
from dataclasses import replace

coupled = replace(
    config,
    atmosphere_structure="mean-field",
    balmer_profile="unified", normalize_balmer_strength=False,
    polarized_transfer="full-stokes-iquv",
    include_magnetic_eos=True, include_rwa_photoionization=True,
    include_centered_motion=True,
)
```

Above 1 MG this solves one structure at the projected-area mean field with
angle-averaged H2db Balmer opacity, the selected continuum and optional
magnetic Saha/HM chemistry. Below 1 MG it retains the DA-structure
approximation. `balmer_profile="unified"` selects the ordinary zero-field
Stark and neutral-broadening templates; `normalize_balmer_strength=False`
retains the field-dependent H2db total oscillator strength.

`include_rwa_photoionization=True` selects stationary-state rigid-wavefunction
H I photoionization through n=8 with shifted dissolved-level edges.
`include_magnetic_eos=True` selects Landau-quantized charged particles and
H2db stationary energies in the Saha/HM closure. `include_centered_motion`
controls the transverse-mass weighting in both the EOS and RWA populations.
These are separate diagnostic options, not ingredients of the paper default.

Full IQUV transfer includes dichroism and magneto-optical dispersion at each
cell, but only disk-integrated Stokes I is returned. Full IQUV transfer works with both
line prescriptions: each magneto-optical profile is the Kramers–Kronig partner of
its absorption profile, and template nodes that round to the same frequency
are merged before the transform. The compressed geometry
does not retain the azimuths needed for Q/U, and V has not been validated
against polarimetry. `include_cyclotron_absorption=True` additionally uses
magneto-ionic free-free and Thomson coefficients with a Doppler-broadened
cyclotron resonance. Its broad optical absorption at hundreds of MG is not
supported by all comparison objects and remains off by default.

## Limitations and references

This is an approximate LTE spectral prescription, not a simultaneous
Stark–Zeeman calculation or a reproduction of unpublished Jordan/Moss code.
H2db substitutes for their component data. The accepted input range
(5000–40000 K, log g 7–9.5 and fields within the atomic tables) is wider
than the tested sample; table coverage does not establish physical accuracy.
At high fields, stationary components can have widths quite unlike a
translated zero-field profile. Magnetic Lyman/Paschen/Brackett data,
thermally decentered atoms and polarimetric output are absent. Strong-field
molecular chemistry is absent, which limits cool-star applications.
Agreement with eight objects does not guarantee an unbiased parameter fit.
At zero field the default still uses Kurucz/Griem profiles; select unified
profiles to recover the ordinary DA line-profile limit.

Scientific use should cite the relevant ingredients and observational sources:

- [Kurucz (1970), SAO Special Report 309](https://articles.adsabs.harvard.edu/pdf/1970SAOSR.309.....K), section 5.14 and `STARK` on p. 243: historical Griem-based shape.
- [Moss et al. (2024)](https://doi.org/10.1093/mnras/stad3825), section 4.2: scalar, nonmagnetic-atmosphere, normalized-component prescription.
- [Schimeczek & Wunner (2014)](https://doi.org/10.1016/j.cpc.2013.09.023) and [H2db data](https://doi.org/10.18419/DARUS-2118): component wavelengths, dipole strengths and energies.
- [Hardy, Dufour & Jordan (2023)](https://doi.org/10.1093/mnras/stad196): offset-dipole parameters and comparison spectra.
- [Vera-Rueda & Rohrmann (2024)](https://doi.org/10.1051/0004-6361/202449627): field geometry and the J1018/J2247 comparisons.

[Release notes and regression controls](../development/history/dah-release-2026-09-29.md)
describe the integration and radiative convergence repair.
Data attribution and redistribution details are in
[third-party notices](../../THIRD_PARTY_NOTICES.md).
