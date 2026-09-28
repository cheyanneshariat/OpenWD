# DAH module

[Model guide](README.md) · [Getting started](../getting-started.md)

`compute_dah` solves a magnetic pure-hydrogen atmosphere and its optical
spectrum, 3400–8000 Å, from about 1 kG to several thousand MG. A spectrum
depends on the field over the visible disk, so the field geometry is part
of the input:

```python
from wd_spectra import DAHConfig, run_model

# Uniform 0.325 MG field of unknown direction (GD 9).
weak = DAHConfig(effective_temperature=16_700, logg=8.07,
                 magnetic_field_megagauss=0.325)

# Offset dipole, Hardy et al. (2023) convention: polar field, inclination
# of the dipole axis to the line of sight, offset along the axis.
dipole = DAHConfig(effective_temperature=22_642, logg=8.37,
                   magnetic_field_megagauss=45.09, field_geometry="dipole",
                   dipole_inclination_deg=66.0,
                   dipole_offset_radius=(0.0, 0.0, 0.17))

run = run_model(dipole, "results/j2149")
print(run.convergence_verified)
```

`dipole_offset_radius=(ax, ay, az)` is in stellar radii in the magnetic-axis
frame of Vera-Rueda & Rohrmann (2024): z along the dipole axis, the line of
sight in the x–z plane. `field_strength_definition="visible-mean"` instead
fixes the projected-area mean field. A uniform field may be given a fixed
angle to the line of sight with `field_angle_deg`.

## Physics

- **Line opacity, B ≤ 1 MG everywhere on the disk.** The complete DA Balmer
  opacity (Tremblay–Bergeron Stark profiles, neutral broadening, HM/Q-MHD
  populations, H3–H22) is split into the normal Zeeman triplet
  (Δν = eB/4πm_e c). At B = 0 it is the DA opacity exactly.
- **Line opacity, stronger fields.** Hα–H12 component wavelengths, relative
  strengths and field-dependent total strengths come from the public
  Schimeczek & Wunner H2db calculation, with Boltzmann factors for the split
  n = 2 substates. The release reader restores the Δm = 0 components from
  the loosely bound 2p(m = +1) substates, which the archive stores only once
  per |m|, and uses the physical sign of m, so Δm = +1 is always the blue
  component. Each component carries the zero-field unified profile of its
  parent line, translated in frequency with its own stimulated-emission
  factor; there is no general theory of
  simultaneous Stark and Zeeman broadening. H13 and higher are omitted
  above 1 MG: their upper levels are already field-mixed and H2db has no
  data for them.
- **Continuum.** Above 1 MG the n = 1–8 H I bound-free opacity is the
  polarization-resolved stationary-state rigid-wavefunction (RWA)
  calculation of Rohrmann (2026) with H2db energies and the atmosphere's own
  level populations. The dissolved-level pseudo-continuum of the DA model is
  attached to each shifted RWA edge instead of the zero-field edges.
  Free–free, H⁻ and scattering are as in DA.
- **Free electrons (`include_cyclotron_absorption`, off by default).** When
  enabled, the free–free absorption, Thomson scattering and their dispersion
  become the cold magneto-ionic coefficients of the three circular modes,
  `R_q = (ω² + Γ²)/((ω − qω_c)² + Γ²)` times their zero-field values, with the
  co-rotating cyclotron resonance Doppler broadened. The collision frequency
  is taken from the free–free opacity itself, so B → 0 recovers DA exactly,
  the resonance has the classical sum-rule strength, and its thermal
  fraction is ν/(ν + γ_rad). At 200–600 MG this predicts a broad depression
  (tens of percent near λ_c = 10712 Å × 100 MG/B, consistent with Martin &
  Wickramasinghe 1979) that the J2247+1456 spectrum does not show, so it is
  not yet a default.
- **Chemical equilibrium.** Above 1 MG the magnetic Saha equation of
  Vera-Rueda & Rohrmann (2020): Landau-quantized electron and proton
  translation, H2db bound-state energies (converted with the reduced-mass
  Rydberg so that B → 0 recovers the ordinary EOS exactly) and the
  centered-motion transverse-mass factor, coupled to HM/Q-MHD occupation
  probabilities. Thermally decentered atoms are not included, because no
  matching radiative cross sections exist. Molecules and negative ions are
  absent from the magnetic EOS; H⁻ opacity is still evaluated.
  `include_centered_motion=False` disables transverse-mass weighting in
  both the EOS and the RWA substate populations.
- **Structure.** At or below 1 MG the structure is the DA structure. Above
  1 MG a radiative structure is solved at the projected-area mean field
  with the shared adaptive Newton solver and its equilibrium certificate,
  using the magnetic chemistry and angle-averaged H2db and RWA opacity (and
  the magnetized free electrons when enabled) on a mesh that resolves the
  displaced components and the cyclotron resonance. Convection is
  suppressed at 0.05 MG and above (Tremblay et al. 2015); weaker fields keep
  ML2/α = 0.7. One structure is shared by all surface cells. The maximum
  visible cell field selects the atomic regime for both structure and
  synthesis, even when the mean field is below 1 MG.
- **Transfer and geometry.** The disk is divided into surface cells, each
  with its own field modulus, field–ray angle and limb cosine: 21
  equal-weight field bins for a dipole, and 4 limb × 3 field-direction nodes
  for a uniform field. Each cell uses the local-field chemistry on the shared
  temperature–pressure structure. The coherent-scattering source is solved
  exactly for the cell's angle-averaged opacity. One ray per cell is solved
  with the coupled IQUV equation (matrix exponential with a linear
  equilibrium source, preserving constant-temperature LTE) including line and
  continuum dichroism and magneto-optical dispersion (Kramers–Kronig
  partners of the same opacities). The flux is the projected-area sum of
  Stokes I. `polarized_transfer="scalar-stokes-i"` solves Stokes I only,
  with the ray-specific π/σ opacity.

## Validation

Validation data are bundled. `research/validate_dah_observed.py` runs cold
starts for ten DAHs at their published parameters and scores them. Eight
are from the CDS release of Hardy, Dufour & Jordan (2023): six well-fit
offset dipoles and two Table 6 stress tests. The other two, J1018+0111 and
J2247+1456, are from Vera-Rueda & Rohrmann (2024). No parameter is fitted;
one radial velocity and one 5200–6100 Å flux scale are nuisance
operations. `broad_rms` is the RMS of the 12-pixel-smoothed model/data ratio
over 3820–6950 Å. `feature_rms` compares locally normalized spectra, so it
measures the magnetic components independently of the continuum slope.

### Results at published parameters

Standard-quality (40-layer) cold starts with the defaults (full IQUV,
cyclotron absorption off), single-threaded. Every structure below passes
all equilibrium checks. "Old" is the best hand-configured development-tree
calculation for each star, scored with the same code; Hardy is the
digitized Hardy et al. (2023) model.

| Target | B_p (MG) | Teff (K) | broad RMS | feature RMS | old broad / feature | Hardy broad / feature | minutes |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| J1034+0327 | 11.2 | 15756 | 0.083 | 0.038 | 0.085 / 0.038 | 0.147 / 0.042 | 21 |
| J2149−0728 | 45.1 | 22642 | 0.031 | 0.046 | 0.031 / 0.045 | 0.036 / 0.045 | 9 |
| J1154+0117 | 35.5 | 29316 | 0.066 | 0.066 | 0.034 / 0.061 | 0.036 / 0.062 | 8 |
| J0908+0921 | 51.6 | 29293 | 0.023 | 0.035 | 0.026 / 0.036 | 0.033 / 0.034 | 7 |
| J1254+5612 | 60.4 | 12870 | 0.069 | 0.143 | 0.063 / 0.143 | 0.053 / 0.141 | 14 |
| J1018+0111 | 108.1 | 10500 | 0.069 | 0.036 | 0.045 / 0.036 | — | 16 |
| J0732+3646 | 111.5 | 25904 | 0.065 | 0.074 | 0.054 / 0.074 | 0.035 / 0.074 | 9 |
| J1033+2309 (GH Leo) | 230.5 | 26025 | 0.092 | 0.026 | 0.037 / 0.029 | 0.019 / 0.024 | 8 |
| J1351+5419 | 368.5 | 13937 | 0.037 | 0.040 | 0.051 / 0.038 | 0.050 / 0.043 | 12 |
| J2247+1456 | 437.1 | 19000 | 0.062 | 0.049 | 0.059 / 0.048 | — | 10 |

Line features (feature RMS) now match the older calculations and Hardy's
within noise almost everywhere; the continuum (broad RMS) is best at
40–60 MG and at J1351. The GH Leo, J0732 and J1154 continua are 8–14% too
blue; at least for GH Leo this comes from the synthesis, not the structure
or the published Teff (a zero-field DA spectrum on the same structure has the
observed level), and is attributed to the zero-field Stark profiles of the
components (Limitations). For J1154 (inclination 87°) scalar Stokes-I
transfer on the same structure gives 0.048/0.062: with components this
broad, polarized transfer limits their Stokes-I depth. The old GH Leo value
used thermally decentered atoms, which were removed as unphysical.

For the weak-field SPY/UVES targets (mean locally normalized Hα–Hδ RMS),
GD 9 (0.325 MG, 16700 K) gives 0.0382 with full IQUV, 0.0405 with scalar
Stokes I on the same structure and quadrature, and 0.0477 for the old model.
G 76-48 (0.09 MG, 6680 K) is not yet supported: the strictly radiative DA
structure the field requires does not converge at this temperature.

The runner writes per-target `scores.json` files and a combined
`summary.json`. Check each model's convergence status alongside its spectral
residuals; a score from an unconverged atmosphere is exploratory.

For comparisons that track individual experiments, use
`research/dah_observational_followup.py`. It also supports the bundled
GD 9 and G 76-48 UVES observations, with resolved Hα–Hδ profiles and
one fitted velocity per epoch. Each directory records the configuration,
source hashes, iteration history, model, scores and comparison plot:

```sh
PYTHONPATH=src python research/dah_observational_followup.py gd9 results/gd9-cold
PYTHONPATH=src python research/dah_observational_followup.py j0732+3646 results/j0732-relaxed \
    --checkpoint path/to/atmosphere.npz --relax
PYTHONPATH=src python research/dah_observational_followup.py gd9 results/gd9-hotter \
    --set effective_temperature=17500
```

Omitting `--relax` when supplying a checkpoint holds the structure fixed;
this is a synthesis diagnostic, not a newly converged atmosphere. For strong
fields `compute_dah` rebuilds the mean-field magnetic EOS before fixed
synthesis, including the densities that determine the shared Stark profiles.
When relaxing a strong-field checkpoint at a different quality, its temperature
profile is interpolated onto the requested depth grid; the checkpoint's old
depth count does not override the requested resolution. Fixed synthesis keeps
the supplied grid.
Parameter overrides are exploratory fits and should be reported separately
from comparisons at published parameters. `research/propose_dah_weak_fit.py`
uses a converged baseline and separate temperature/gravity trials to propose
a bounded local step from Hα and Hβ, holding out Hγ and Hδ. Its linear
prediction must be checked by computing a fresh atmosphere at the proposed
parameters; it supplies no parameter uncertainties.

## Limitations

- One structure at the mean field is shared by the whole disk.
- Components use zero-field Stark profiles. Once the Zeeman splitting
  exceeds the linear Stark shifts, the Stark effect of a component becomes
  quadratic and one-sided; exact calculations (Friedrich et al. 1994, A&A
  282, 179) show the width of stationary Hα/Hβ components collapsing by
  factors of 65–125. The translated zero-field profiles therefore smear
  stationary features (e.g. GH Leo's 4130 Å Hβ 2s0→4f0 component) into
  broad, shallow wings and remove blanketing near them. This is the likely
  cause of the 9–12% blue excess of GH Leo and J0732: the same synthesis on
  a zero-field structure keeps the excess, while a zero-field DA spectrum at
  the published parameters has the observed continuum level. Published fits use a
  global empirical factor C = 0.1 on an Unsöld width (Jordan 1992; Vera-Rueda
  & Rohrmann 2024), which OpenWD does not adopt. A per-component
  second-order treatment needs intra-manifold dipole elements that H2db
  does not provide.
- Cyclotron absorption (above) is off by default pending that validation.
- Weak-field DAHs below about 7000 K need a strictly radiative DA structure
  (convection is suppressed), which the shared DA solver does not yet
  converge (G 76-48).
- There is no magnetic Lyman, Paschen or Brackett line data. These series,
  and H13+ above 1 MG, keep zero-field or no line opacity.
- The magnetic EOS has no molecules or H⁻ in the chemistry; below ~8000 K
  this is a real omission.
- Only disk-integrated Stokes I is returned. The cell compression discards
  the sky-plane azimuth needed to sum Q and U, and the absolute sign of V
  has not been checked against polarimetry.
