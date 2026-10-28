# Metal-polluted atmospheres: DZ, DBZ and DAZ

[Model guide](README.md) · [Getting started](../getting-started.md) ·
[Observational comparisons](../validation/dz-observational-comparisons.md)

OpenWD has two configurations for metal-polluted white dwarfs. They share the
metal physics described below and differ in the host atmosphere:

| | `DZConfig` (DZ, DBZ) | `DAZConfig` (DAZ) |
| --- | --- | --- |
| Host | Helium, with optional trace hydrogen | Hydrogen |
| Abundances | `log10 N(Z)/N(He)`; trace H as `log_hydrogen_abundance = log10 N(H)/N(He)` | `log10 N(Z)/N(H)` |
| Host EOS, opacity and lines | DB/DAB helium physics, with the [dense-helium physics](#cool-helium-dominated-dzs) of cool stars | DA hydrogen physics, including its line and molecular policy |
| Convection | ML2/alpha = 1.25 | ML2/alpha = 0.7 |
| Neutral metal-line broadening | Unsöld widths for He; unified He profiles for the strongest lines | Anstee–Barklem–O'Mara widths for H where tabulated, Unsöld otherwise |
| Default mixture | GD 40 | G29-38 |
| In-memory function | `compute_dz` | `compute_daz` |
| Command-line example | `examples/one_shot_dz.py` | `examples/one_shot_daz.py` |

In both, metals contribute electrons to the common charge-neutrality
solution, and their bound-free and line opacity enters the relaxed
atmosphere, not only the final spectrum. They remain trace contributors in
the thermodynamic derivatives used by ML2; metal-rich mixtures need the
composition-complete EOS of the [D6 module](D6.md). Ca II H and K can use
complete-redistribution source functions in the final spectrum while the
atmospheric populations and extinction remain LTE; this is not a full
metal-NLTE atmosphere. Atmospheres are plane parallel and abundances are
fixed inputs; the models do not fit observations.

## Running a model

```python
from wd_spectra import DAZConfig, DZConfig, run_model

daz = run_model(DAZConfig(effective_temperature=11_820, logg=8.40, quality="standard"),
                "results/daz-g29-38")
dz = run_model(DZConfig(effective_temperature=5_830, logg=8.09, log_hydrogen_abundance=-4.4,
                        abundances={"Ca": -10.14, "Mg": -8.6, "Fe": -9.3, "Si": -8.6,
                                    "Na": -9.6, "Cr": -10.83, "Ni": -10.97},
                        quality="standard"),
               "results/dz-vma2")
print(daz.convergence_verified, dz.convergence_verified)
```

Supplying `abundances` replaces the entire default mixture. All required
standard atomic and profile data are bundled. A new run starts from scratch,
and `require_convergence=True` requires all the current atmosphere-certificate
gates; unqualified exploratory spectra warn. The automatic interface never
routes a polluted atmosphere through a pure-host substitute on failure.

The equivalent commands are:

```bash
python examples/one_shot_dz.py --teff 15300 --logg 8.0 \
  --abundance O=-5.61 --abundance Mg=-6.24 --abundance Si=-6.76 \
  --abundance Ca=-6.88 --abundance Fe=-6.48 --log-h-he -6.16 \
  --quality standard --output results/gd40
```

```bash
python examples/one_shot_daz.py --teff 11820 --logg 8.40 \
  --quality standard --output results/daz-example
```

The model does not refit `Teff`, `log g` or composition. Standard line-rich
helium atmospheres can take from about 30 minutes to several hours; the cool
DZ comparison models below took 3–35 minutes, and the DAZs in the
[seven-object comparison](#seven-polluted-white-dwarfs) 220–295 s.

The [DZ fitting notebook](../../examples/fit_dz_spectrum.ipynb) fits Ca, Mg and
Fe in four public VLT/UVES spectra of PG 1225-079, with Teff, log g and the
remaining composition held fixed. The observed coadd is included in the
repository. Each trial uses `relax_atmosphere=False` on one certified
cold-start structure. These are diagnostic fits, not a final abundance
solution. See the [fitting notes](../examples/fitting-dz.md) for data
provenance, error assumptions and the structural-feedback check.

## Metal physics (both hosts)

The default metal database uses Stout v3.00b4 levels and transitions through
charge 3, adding bundled Kurucz transitions with independent atomic-data
corroboration under a common selection rule when those ions are present; see
the [atomic-data notes](../../src/wd_spectra/data/atomic/README.md). The metal
physics was revised after a 2026-09-30 audit (physics revision
`metal-audit-2026-09-30`); the revisions apply to both hosts.

The [population-consistency repair](../validation/metal-population-consistency.md)
uses matching occupation factors in native line and photoionization
populations. It retains the cool-DZ Rydberg-only neutral prescription and
low-level reservoir. Identified external levels use the native population
atom; unresolved grouped levels retain an explicitly recorded effective-level
approximation. Dated comparisons below retain their original source versions.

- Bound-free opacity is level-resolved Opacity-Project photoionization from
  every level of the bundled TLUSTY/SIROCCO model atoms (C I--II, O I--II,
  Mg I--II, Al II, Si I--II, S II, Fe II), including excited-level continua
  such as Mg I 3P and Si I 1D. Identified model terms are placed at their
  observed Stout energies. Other ions keep Verner et al. (1996) fits, which
  are applied to the whole LS ground term rather than its lowest ``J``
  level. `level_resolved_photoionization=False` restores Verner fits only.
  Verner's phfit2 fits supply ground-state edges for iron-group ions absent
  from `photo.dat` (Ti, Cr, Mn, Ni, ...). Bautista (1997) Fe I and NORAD Cr I
  level-resolved cross sections are bundled (xz-compressed, checksummed) and
  required; they are used for terms identified with observed levels within
  0.5 eV.
- Metal partition functions use Hummer--Mihalas occupation probabilities
  (Q-MHD charged microfields, and in helium hosts the neutral-He excluded
  volume), iterated with charge neutrality; the line opacity applies the same
  level survival. `occupation_probability_metal_partitions=False` restores
  the fixed 0.1-eV cutoff. Since 2026-10-12 the line-opacity survival counts
  every ion in the charged microfield, including the host ions. Before, it
  counted the metal ions only, which in GD 61 is 10⁻³–10⁻⁵ of the electron
  density, so the line opacity applied almost no dissolution and was
  inconsistent with the partition functions. Spectra change by at most 3%
  (one pixel in the J0738 control).
- Metal lines use an exact Voigt profile evaluated in frequency (Humlicek
  W4), so impact wings keep their physical asymmetry. Unsöld widths use the
  hydrogenic ``<r^2>`` with its ``1/Z^2`` core-charge factor, and levels
  above the first ionization limit are measured from their parent limit.
- The structure solve computes opacity for exactly the lines its frequency
  grid samples.
- The `strong_line_atomic_data="nist-asd"` option replaces every exactly
  matched Stout transition down to f = 1e-4 with evaluated NIST ASD values
  (accuracy C or better), including 115--300 nm (Mg I 2852, Mg II h/k, Si II,
  C II, Al II, Fe I/II). The default retains existing Stout transition data
  and uses the [independently corroborated Kurucz catalogue](../validation/independent-kurucz.md)
  for missing transitions. The same rule covers 30 neutral/singly ionized
  catalogues of 15 elements, admitting 19,966 absorption additions in seven
  ions, including 14,891 Cr II and 4,613 Fe II transitions. Original Stout
  oscillator strengths and damping records are retained; qualified weaker
  radiative branches also contribute to inferred natural widths. The two
  previously added Cr II/Fe II levels are retained; no further levels are
  imported. This replaces the earlier unfiltered Cr II and selected Fe II/Si II
  supplements. Selection and provenance are recorded in the
  [atomic-data notes](../../src/wd_spectra/data/atomic/README.md).
- The common catalogue admits 17 Si II absorption additions, including
  1309.45/1309.73 Å. It holds out 1305.59 Å because its upper level is slightly
  above the ground ionization limit. This calls for an autoionization/profile
  treatment beyond the present bound-bound rule; it does not establish that
  the observed resonance is spurious. When the earlier 1471-line Kurucz Si II
  import was introduced (2026-10-12), the whole-spectrum COS χ² per pixel fell
  by 25% for GALEX J1931+0117, 24% for WD 1226+110, 6% for PG 0843+516 and
  2% for PG 1015+161, with no change in the optical windows.
  `si_ii_kurucz_transitions` is retained for configuration compatibility with
  that import; it cannot broaden the independently selected default atom. For
  a bare-Stout diagnostic, supply an atom read with
  `include_default_supplements=False` and disable that legacy flag.
- Most Stout transitions carry no electron-impact (Stark) width, and the
  classical SYNSPEC fallback stays off. By default (`kurucz_electron_stark=True`,
  since 2026-10-11) OpenWD attaches Kurucz's tabulated `log Gamma_4/n_e` at
  10^4 K, scaled as (T/10^4 K)^(1/6), to every E1 transition that has no width
  of its own and matches a Kurucz line (wavelength, level energies and
  statistical weights). Measured or previously tabulated widths are kept, and
  no other atomic data change. The constants come from
  `src/wd_spectra/data/stark/kurucz_stark_constants.csv.xz` (153,641 Stout
  transitions for charges 0--3, vacuum 900--11000 A), built by
  `scripts/build_kurucz_stark_table.py` from Kurucz `gfZZCC` lines between
  observed levels. The table is keyed by Stout level indices, because Stout
  has distinct levels with equal energy and weight, and the energies and
  weights are checked again on use. The source URLs and SHA-256 checksums
  are in the file's trailer. Lines outside that range still get no Stark
  width. For example, Si II 1260/1264/1265 gets `log Gamma_4/n_e = -5.75`,
  the same value VALD3 lists. In fixed-structure comparisons at published
  parameters, the mean window residual fell by 1.2% both for 54 windows of
  GD 40, G241-6 and seven further optical DZ/DAZ targets (20 better, 5 worse)
  and for 34 windows of eight DAZs at 5900--23100 K (6 better, 1 worse by
  0.2%). The gains are in Si II lines where electron densities are high: the
  COS Si II 1260 residual of GD 40 and G241-6 fell 16--18% (the lines remain
  weaker than observed), GALEX J1931+0117 Si II 5041/5056 by 18% and 4128/4131
  by 11%, and WD 1226+110 Si II 5042/5057 by 4%; Ca and Mg windows did not
  change. `kurucz_electron_stark=False` restores the previous behaviour.
- Ca II H and K use complete-redistribution source functions solved
  exactly with the linear formal Lambda operator, with line centres taken from
  the atomic transitions and a fixed integration grid. Collisional transfer
  between the two 4p levels feeds the partner line instead of thermalizing
  the photon. Populations and extinction stay LTE.
- The emitted spectrum is flux conserving: 1.000--1.003 of sigma Teff^4 for
  DZs at 9000--15600 K (0.84--0.91 before), and 1.005, 1.003 and 1.001 for
  the DAZs G149-28, G29-38 and GALEX J1931+0117 (0.984, 0.991 and 1.040
  before). Three numerical changes do this:
  - The structure absorbs the synthesis line list (f >= 1e-4, the same
    20,000-line budget), opacity-sampled on a uniform R = 1000 grid
    (`structure_opacity_sampling_resolution`). The former line-centered
    structure samples (1000 lines for DZ, up to 8000 for DAZ) left the extra
    synthesis blanketing unbalanced and overweighted narrow cores.
  - The structure depths are concentrated across 0.01 < tau < 10,
    taking points only from the optically thin top
    (`photospheric_depth_concentration=1`).
  - The final formal solution subdivides each depth interval four times
    (`synthesis_transfer_depth_refinement=4`), interpolating opacity
    log-log in column mass. On the bare structure grid the piecewise-linear
    formal solution lost 5--6% of the flux of steep convective atmospheres.

  Together these add about 15--20% to a standard DZ cold start. The DAZ
  cold starts of G149-28, G29-38 and GALEX J1931+0117 took 220, 243 and
  295 s, against 165, 267 and 771 s before.
  `structure_opacity_sampling_resolution=None`,
  `photospheric_depth_concentration=0` and
  `synthesis_transfer_depth_refinement=1` restore the previous numerics.

## Helium hosts (DZ, DBZ)

`DZConfig` solves a helium-dominated polluted LTE atmosphere. The GD 40
preset includes levels through charge 3, dense-helium ionization shifts, and
the unified metal–He profiles below. Observable helium and trace-hydrogen
lines use the same policies as DB and DAB. Trace-hydrogen Lyman lines retain
the paper figure's charged-particle Stark treatment by default;
`lyman_profile_source="allard"` explicitly selects the later unified-profile
option.

- The Blouin et al. (2018) dense-He shift lowers the first ionization
  potentials of C, Na, Mg, Ca and Fe in the Saha equation, with the full
  partition functions, so it vanishes in the dilute limit. Outside the
  authors' tested 2000--10000 K and 0--1.5 g cm^-3 range the fit is held at
  its edge instead of being switched off.
- The measured Hammond (1975) Ca II H/K widths in helium are extended in
  temperature with impact theory for his own Lennard-Jones (12,6) fits to
  those widths and shifts. His two-shot exponent, which he advises against
  using, is no longer applied. At 15000 K the width is 1.58 times its
  5200-K value, against 1.27 for the old power law.
- Inside the published Ca I--He density range the unified 4227-A profile
  replaces the impact profile; below it, the impact core plus the
  density-scaled non-impact part of the unified profile is used. Mg I 2852
  uses the Allard et al. (2018) red-wing cross sections wherever they exceed
  the ordinary profile; above their 1e21 cm^-3 validity boundary they are held
  rather than extrapolated.
- `si_ii_he_unified_profile=True` (the default since 2026-10-11; `False` restores
  the Unsöld Lorentzian) replaces the Lorentzian
  neutral-He broadening of the Si II 1260/1264/1265 Å triplet
  (3s² 3p ²P°–3s² 3d ²D) with a unified binary-collision profile. The bundled
  table `src/wd_spectra/data/profiles/si_ii_1260_he_unified.npz` (checksum
  verified) covers 4000–40000 K. It was computed from one-electron
  model-potential Si⁺–He adiabatic potentials, with dipole-weighted Λ
  components, straight classical paths, a Maxwell average and the
  Anderson–Talman autocorrelation. The same method reproduces Allard et al.
  (2016) Mg II–He widths to 0.3–2% and Hammond's (1975) laboratory Ca II H&K
  width to 13%. Electron and radiative damping still come from each line's own
  data, and the Doppler core from its Voigt profile, merged by the existing
  Si II envelope rule. Nothing is fitted to a stellar spectrum.
  With the stellar parameters, abundances and structures fixed, the COS
  1258–1269 Å residual falls from 0.221 to 0.153 for GD 40 and from 0.166 to
  0.104 for G241−6, and in three further DBZs computed from cold starts at
  their published parameters from 0.095 to 0.082 (GD 61), 0.069 to 0.056
  (GD 378) and 0.066 to 0.056 (WD 1425+540); every other COS window is
  unchanged. The triplet wings
  stay weaker than observed. Known approximations: LS potentials (no fine
  structure), no 3s² 3d–3s 3p² mixing, a constant transition dipole, and
  independent ab initio Si⁺–He potentials that give a smaller improvement.
- Neutral trace H contributes its impact broadening to ordinary metal
  lines using the neutral-H density from the same metal/H charge closure.
  For the Si II--He unified profile, this H width joins radiative and
  electron widths in `gamma_other`; the unified profile supplies the He
  contribution. Pure-He and fully ionized trace-H cases receive no added
  neutral-H width. No abundance or damping multiplier is fitted.
- Trace hydrogen keeps charged-particle (Q-MHD) level dissolution only. An
  audit variant also applied the HM88 neutral-He hard-sphere term with
  `r_n = n^2 a_0`. That removed almost all n >= 6 population in DBA
  photospheres and erased the observed Hdelta of WD J1013+0259, so it was
  reverted.

Paper-spectrum regression and cold-start convergence are separate checks.
See [tested points](../tested-temperature-ranges.md) and
[reference-comparison limitations](../limitations.md#spectrum-accuracy-and-reference-comparisons)
for the status of PG 1225 and SDSS J0738+1835.

### Cool helium-dominated DZs

Below about 8000 K, helium-dominated photospheres are dense fluids
(0.005–0.4 g cm^-3 at tau_R = 1 in the cool comparison below). The cool-DZ
release (2026-10-23) therefore selects these defaults:

| `DZConfig` option | Previous default | Default |
| --- | --- | --- |
| `dense_helium_eos` | `ideal` | `reos3` |
| `dense_helium_free_free` | off | `rogers-young` |
| `helium_minus_free_free` | `automatic` | `partial-wave` |
| `mg_i_b_he_unified_profile` | `False` | `True` |
| `mg_ii_he_unified_profile` | `False` | `True` |
| `ca_ii_he_unified_profile` | `False` | `True` |
| `na_i_he_unified_profile` | `False` | `True` |

- **Equation of state.** Inside the He-REOS.3 table (Becker et al. 2014) the
  mass density and internal energy come from the ab initio table, and the
  ionization equilibria are solved at the resulting helium density. Only the
  departures from an ideal monatomic gas are interpolated between the widely
  spaced isotherms, smoothly (PCHIP), so the ideal temperature dependence is
  exact and all first derivatives are continuous. Linear interpolation gave
  ~30% jumps in nabla_ad at each isotherm and stalled cool-DZ solves.
  `dense_helium_eos="ideal"` retains the preceding chemical-picture EOS for
  controlled comparisons.
- **He⁻ free–free.** The dense-fluid reduction of Iglesias, Rogers & Saumon
  (2002) multiplies the dilute coefficient, with the He structure factor from
  the thermodynamically consistent Rogers–Young closure (within 1.2% of Monte
  Carlo for the same potential). Below John's 0.5063-µm edge, the Carbon et
  al. (1969) fit, 16–24% above John at the edge, is replaced by John's edge
  value times the wavelength shape of a first-principles e–He partial-wave
  calculation (2–3% from John 1994 at 0.5–15 µm).
- **Unified line profiles.** Mg I b, Mg II h/k, Ca II H/K and Na I D use
  unified He profiles computed for OpenWD from published ab initio potentials
  and transition dipoles, for 3000–12000 K and n_He = 1e19–1e23 cm^-3. They
  keep the physical band area that R-dependent dipoles imply, and the ionic
  and Na I profiles are corrected for correlated He perturbers with a Monte
  Carlo simulation of the fluid. At n_He >= 1e20 cm^-3 the table is the
  complete line; below that, the larger of the table and the ordinary Voigt
  profile is used. The paper's appendix describes the construction. The older
  Mg I 2852 and Ca I 4227 tables have different, more restrictive domains.
- **Populations.** The neutral-He excluded-volume factor acts only on
  Rydberg levels (mean radius beyond that of n* = 4 in a neutral atom). A He
  atom inside a compact orbital forms a quasi-molecule whose effect is the
  pressure-broadened line; applied to every level at n_He ~ 1e22 cm^-3, the
  factor left the Ca II partition function below its ground-state weight and
  overcounted Ca and Mg by up to 9×.
- **Convergence.** The lower-boundary controller deepens the domain earlier
  after screening failures and restarts the phase sequence; the physical
  certificate tolerances are unchanged.

These are model prescriptions; numerical convergence alone does not
establish their accuracy for a particular star. The
[release record](../development/history/cool-dz-release-2026-10-23.md) lists
the regression changes, comparisons with the previous defaults and
limitations. Setting the seven options back to their previous values is a
diagnostic of the default choices; the EOS interpolation, population,
Ca-source and RAP-reader repairs still apply.

## Hydrogen hosts (DAZ)

`DAZConfig` uses the shared adaptive DA solver, ML2/alpha = 0.7, the DA
hydrogen line and molecular policy, Stout metal lines through charge 3 and
Verner photoionization. It retains the established Feautrier atmosphere
solver and formal-integral final synthesis. Dense-helium ionization
corrections and helium-perturber profiles are not applied to hydrogen hosts.

Metal-line broadening by neutral hydrogen uses the Anstee-Barklem-O'Mara
(ABO) cross sections where they are tabulated (`metal_neutral_h_broadening="barklem"`,
the default since 2026-10-12), with Unsöld widths for the remaining lines.
Unsöld widths alone underestimate this broadening, which matters only in cool
hosts where hydrogen is mostly neutral. In a fixed-structure comparison of
eight DAZ targets (34 windows), the mean window residual fell by 1.1%: 6
windows improved and 1 got worse by 0.2%. The gains were in the coolest star,
NLTT 43806 (5900 K): Ca II H&K -13%, Al I 3945/3962 -15%, Ca I 4228 -6%.
G149-28 Ca II K improved by 2.5%, and the hot DAZs did not change.
`metal_neutral_h_broadening="unsold"` restores the paper's prescription.
No option is selected by target name. Hydrogen Balmer self-broadening is a
separate prescription and follows the DA policy.

`DAZConfig(h2_lyman_werner_lines=True)` adds photospheric H2 Lyman and
Werner lines with fluorescent source functions to the final spectrum. The
physics and data are the same as for DA; see the [DA guide](DA.md). In the
COS G130M spectra of GD 133 and G29-38, at fixed published parameters, it
lowers chi^2 (1153-1437 Å) from 2.37 to 1.47 and from 3.26 to 2.71. The fitted
line-strength scale `k` is 1.14 and 0.78. The option is off by default; the
[UV comparisons](#ultraviolet-comparisons) enable it.

The fixed-atmosphere regression controls include G149-28 (8600 K) and GALEX
J1931+0117 (20890 K) from the paper's seven-object figure. These tests use the
paper's Stark-only Lyman setting (`lyman_profile_source="stark"`) and pin the
Unsöld widths; new calculations default to the DA Allard policy. Fresh
standard-resolution checks of G149-28, G29-38 and GALEX J1931+0117 use
`research/validate_daz_cold_start.py` and require all five physical
atmosphere gates. These are individual composition/gravity points, not a
qualified temperature or abundance grid. Atmosphere convergence does not
certify depth independence, exact reproduction of a paper atmosphere, or the
integral of a separately sampled final spectrum. A low-temperature pure-DA
validation does not establish the same validity range for arbitrary metal
abundances.

The observations, archived figures and reproduction commands are collected
in [DZ/DAZ observational comparisons](../validation/dz-observational-comparisons.md).
The links below preserve the original section addresses.

## Paper comparisons

See [Paper comparisons](../validation/dz-observational-comparisons.md#paper-comparisons).

### Seven polluted white dwarfs

See [Seven polluted white dwarfs](../validation/dz-observational-comparisons.md#seven-polluted-white-dwarfs).

### Cool DZs and a cool DAZ

See [Cool DZs and a cool DAZ](../validation/dz-observational-comparisons.md#cool-dzs-and-a-cool-daz).

## Xu et al. (2019) HIRES comparisons

See [Xu et al. (2019) HIRES comparisons](../validation/dz-observational-comparisons.md#xu-et-al-2019-hires-comparisons).

### WD 1232+563

See [WD 1232+563](../validation/dz-observational-comparisons.md#wd-1232563).

### WD 1551+175

See [WD 1551+175](../validation/dz-observational-comparisons.md#wd-1551175).

### WD 2207+127

See [WD 2207+127](../validation/dz-observational-comparisons.md#wd-2207127).

## Ultraviolet comparisons

See [Ultraviolet comparisons](../validation/dz-observational-comparisons.md#ultraviolet-comparisons).

### GD 40

See [GD 40](../validation/dz-observational-comparisons.md#gd-40).

### G241−6

See [G241−6](../validation/dz-observational-comparisons.md#g2416).

### PG 1225−079

See [PG 1225−079](../validation/dz-observational-comparisons.md#pg-1225079).

### GD 61

See [GD 61](../validation/dz-observational-comparisons.md#gd-61).

### GD 378

See [GD 378](../validation/dz-observational-comparisons.md#gd-378).

### WD 1425+540

See [WD 1425+540](../validation/dz-observational-comparisons.md#wd-1425540).

### G29−38

See [G29−38](../validation/dz-observational-comparisons.md#g2938).

### GD 133

See [GD 133](../validation/dz-observational-comparisons.md#gd-133).

### GD 362

See [GD 362](../validation/dz-observational-comparisons.md#gd-362).
