# Caveats and limitations

[Documentation home](README.md) · [Getting started](getting-started.md)

OpenWD is research software under active development. A useful-looking spectrum, numerical
convergence, agreement with a reference spectrum, and adequate physical
approximations are separate questions.

## Tested temperatures and compositions

Tested cold-start points reach **3000 K for DA, 4500 K for pure-He DB, and
7500 K for a DAB mixture with N(H)/N(He) = 0.01**, at log g = 8. These are
individual tested points, not validity ranges. The cool DB/DAB workflows are
experimental and require the setup described in the [user guide](getting-started.md#cool-helium-and-mixed-atmospheres).

DQ has a cold-start-qualified point at J1225 (6294 K, log g = 7.924,
log(C/He) = -5.33); it is not a qualified DQ temperature/abundance grid. See
the [DQ release evidence](tested-temperature-ranges.md#dq-release-qualification).
DO/DAO and PG 1159 are qualified at the individual stars and temperatures in
the [tested-point table](tested-temperature-ranges.md).

The [tested-point table](tested-temperature-ranges.md) is the detailed record
of temperatures, compositions, settings, and convergence evidence. In particular:

- No abundance or gravity grid has been demonstrated.
- DAB at 7250 K remains unqualified, and 5000 K DAB experiments remain far
  from equilibrium. There is no validated 5000–10000 K mixed-atmosphere grid.
- Historical continuation calculations are not evidence for a fresh run.
  Public generation requires no prior model.

## What convergence means

`run.convergence_verified` reports qualification for the selected workflow's
declared equations and numerical checks. Established presets require actual
all-depth flux conservation, local energy balance, a measured unrestricted
temperature correction, scattering-source closure, and lower-boundary screening.
The experimental dense-helium workflow has a dedicated checker and retained
audit records; metal-free DABs use the in-process molecular closure and the
standard certificate. A solver's terminal success flag alone is insufficient.

For established presets, unqualified completed spectra remain available with
a warning for exploration; use `require_convergence=True` to require numerical
qualification. DQ always requires its atmosphere certificate and a finite,
positive independent 218520-point spectrum with absolute bolometric ratio
error at most 0.002. It raises on failure even without the strict flag.
PG 1159 distinguishes `spectrum-qualified` (flux, local energy, population,
source and boundary checks that protect the emergent spectrum) from
`converged` (which also requires a full-rank temperature certificate);
`require_convergence=True` accepts only the latter.
A failed calculation never selects an alternative physics prescription automatically.

## Spectrum accuracy and reference comparisons

Atmosphere qualification does not by itself establish wavelength, angle, or
depth-grid independence, nor the accuracy of a separately synthesized spectrum.
The former piecewise-linear spectrum method has known transfer-consistency limits:
a broad-wavelength audit integrated to 0.98613 of the expected stellar flux for
DA 3000 K and 0.99314 for DB 10000 K. These historical numbers are not measurements
of the new cubic DA default. Spectra are not renormalized to hide flux errors.
See the [numerical report](development/history/cold-start-numerics-2026-09-07.md#known-spectrum-consistency-limits-unfinished-changes-excluded).

For established presets, different atmosphere and synthesis methods are
intentional, not by themselves a convergence failure. The established Feautrier
atmosphere and formal-integral
synthesis remain the defaults; the experimental matched-transfer and forced
fine-wavelength atmosphere calculations are not enabled. Convergence checks
and spectral-accuracy checks remain separate. Default DA regression tests cover
absolute flux and Balmer cores/wings without selecting an alternative method.

DQ instead uses conservative refractive transfer in both structure and
synthesis, with independent wavelength grids. Its bolometric qualification
does not establish independent depth/angle convergence or agreement with
observations. The DQ spectrum regression protects absolute flux from a saved
state; only the separate cold canary tests construction of a new atmosphere.

DA calculations now default to `synthesis_transfer="formal-pchip"`
for higher-order source interpolation on the same atmosphere. Explicit
`synthesis_transfer="formal-linear"` retains the former interpolation. The checked
12000-K standard model's broad sampled flux ratio improves from about 0.980
to 1.0002 without flux rescaling. This is not a universal flux-conservation
guarantee. Separate checked cold-state controls now protect the new DA default
at 12000 and 20000 K; the historical spectral controls remain unchanged.
See the [DA guide](models/DA.md#spectrum-synthesis).

Regression controls protect previously calculated spectra; they are not
independent observational validation. The repository includes the observed
[PG 1225-079 UVES coadd](../examples/data/README.md) used by the fitting
example, with source records and a checksum. Other comparison observations
and published model grids are not bundled as a complete validation dataset.
The fresh production SDSS J0738+1835
atmosphere passes static checks but does not pass the paper-spectrum comparison.
PG 1225's production cold-start check is also distinct from exact reproduction
of the lower-resolution paper model. These distinctions are recorded in the
[tested-point details](tested-temperature-ranges.md#limits-and-preservation-of-established-results).

The undoubled Q-MHD critical-field correction intentionally changes some
warm-model higher-series features beyond the old spectral regression bounds.
Reviewed corrected outputs are now protected by separate regression controls;
the historical files and numerical tolerances have not been overwritten or
relaxed. This is not a claim of improved agreement at every wavelength.
See the [microphysics audit](development/history/microphysics-audit-2026-09-10.md)
for the distinction between corrected equations, cold-start convergence, and
preservation of historical spectra.

## Physical approximations

All modules are plane-parallel and static. DA, DAZ, DB, DAB/DBA, DZ/DBZ and
DQ use LTE populations; DO/DAO and PG 1159 use NLTE (see below). DAB/DBA
assumes a homogeneous mixture, not a stratified hydrogen layer; DAZ assumes a
hydrogen-dominated host, and DZ/DBZ a helium-dominated host, with fixed input
abundances. DQ assumes hydrogen-free, nonmagnetic helium with trace carbon and
C₂. The public [DAH](#dah) and [D6](#d6) models have the separate
assumptions and limits described below.

Convection uses 1D mixing-length theory. For cool DAs (6000--13,000 K, log g
near 8) the optional [3D line-profile mode](models/DA.md#3d-line-profile-mode-opt-in)
applies a published mean-3D temperature correction for Balmer line-profile
analysis. Its spectra are not flux-conserving; use the default 1D spectra for
spectral-energy distributions and photometry. The switch from full to
photospheric-only correction at 9000 K is empirical.

The dense pure-He DB treatment combines tabulated bulk thermodynamics with
approximate chemical potentials and trace-ion chemistry. It includes a
Rogers–Young dense-fluid correction to He-minus free–free absorption, but
does not include refraction; see the [DB guide](models/DB.md).
DQ includes refraction and dense-helium continuum corrections, but
its dense-mixture EOS, molecular collision profiles and grey refractive ML2
bridge remain approximations. It is not an exact reproduction of Blouin's
implementation or qualified for precision abundance fitting; hot carbon-rich,
hydrogen-bearing and DQp atmospheres are excluded. See the [DQ guide](models/DQ.md).
Molecular H/He mixtures still lack
a consistent dense-mixture free energy, nonideal dissociation, some molecular
ions, and pressure-distorted CIA. The [full limitations list](tested-temperature-ranges.md#limits-and-preservation-of-established-results)
and [model guides](models/README.md) describe the scope in more detail.

Automatic selection identifies the relevance of implemented physics; it does
not establish convergence or supply missing physics. Unsupported overrides,
missing data, or invalid material domains are reported explicitly.

## DO/DAO

[DO/DAO models](models/DO-DAO.md) have seven cold-start-qualified configurations,
not a validated hot-star grid. Their electron density and gas pressure retain
LTE closure; NLTE ionization does not feed back into charge balance. Metals,
winds, radiative acceleration and convection are omitted. One DO target still
fails the strict legacy profile comparison, and GD153's H-alpha profile remains
worse than the TMAP reference. A numerical certificate does not resolve these
physical and observational limitations. The required CCC/TLUSTY inputs are
installed with OpenWD.

## D6

[D6 models](models/D6.md) are LTE and qualified at one star, SDSS J1637+3631 (15680 K, log g = 6.3). Molecules and negative ions are omitted, so they do not yet apply to the cool D6 stars; hot D6 stars need NLTE. Known line-data differences from the Hollands et al. Koester model are listed in the guide.

## DAH

[DAH models](models/DAH.md) default to normalized Kurucz/Griem Balmer
profiles and scalar magnetic transfer on a nonmagnetic DA structure.
Above 1 MG the default synthesis uses stationary-state RWA photoionization,
the magnetic Saha/HM EOS and centered-motion populations on that structure.
The equilibrium certificate applies to that underlying structure; magnetic
opacities do not feed back on its temperature profile. The eight fixed-parameter
paper comparisons span dipole polar fields from 6.13 to 437.1 MG, but neither
those examples nor the input bounds establish a validated parameter grid.

The historical Stark approximation has no separate Doppler convolution or
neutral self broadening, and translating a zero-field profile is not a
simultaneous Stark–Zeeman calculation. Magnetic Lyman/Paschen/Brackett data,
decentered atoms and strong-field molecular chemistry are absent. Opt-in
disk-integrated Q/U/V use signed, uncompressed cells and a declared observer
basis, but have not been validated against polarimetry. The public
`compute_dah(..., surface_cells=...)` interface accepts custom maps;
the caller supplies their phase dependence and checks surface resolution.
The coupled mean-field magnetic structure, full IQUV transfer and cyclotron opacity
remain explicit options. `paper_dah_config` retains the previous zero-field
EOS and bound-free continuum prescription.
The default 21 field bins can under-resolve rapidly moving magnetic line
components. The opt-in drift-resolved disk reduces this numerical error in
two fixed-atmosphere examples; it does not repair continuum, line-width or
atmospheric-structure approximations. See the
[disk-quadrature benchmark](development/history/dah-disk-quadrature-2026-10-08.md).
The [DAH guide](models/DAH.md) distinguishes the current defaults, paper preset
and optional transfer/structure treatments and explains why smooth continuum
corrections belong to observational
comparisons rather than the physical prediction.

## PG 1159

[PG 1159 models](models/PG1159.md) are qualified from a cold start for three
stars: PG 1707+427 (85000 K), PG 1424+535 (110000 K) and PG 1159-035
(140000 K, with radiative acceleration), all with the upper-atmosphere
refinement. They are individual points, not a grid. Only He, C and O define
the atmospheric structure; the trace elements (N, Ne, F, Si, P, S, Ar, Fe) are
NLTE passengers in the final line formation and do not feed back on it.
Radiative acceleration is off by default; upper-atmosphere refinement is
on by default. Convection, winds and diffusion are omitted.

Known differences from the observed spectra, at the published stellar
parameters, are:

- the O VI 1032/1038 damping wings are too deep in PG 1424+535 and
  PG 1159-035 with the Dimitrijević & Sahal-Bréchot / Elabidi et al. Stark
  width;
- He II 4686 and several optical C IV absorption lines are too deep in
  PG 1424+535 and PG 1159-035;
- PG 1159-035 shows spurious O V emission at 6462–6502 Å (and a weak residual
  bump at 6001 Å), a
  too-strong C IV 5801 emission core, and lacks the observed emission cores of
  He II 4686, C IV 4658 and O VI 5291.

PG 1707+427 fits its FUSE spectrum about as well as the published TMAP model,
and its SDSS line windows to reduced chi-square of 1.2 or better. In the refined upper layers the radiative heating and
cooling nearly cancel (net ~1e-4 of the gross terms), and in PG 1159-035 local
radiative equilibrium is thermally unstable between `tau_Ross` 2e-5 and 4e-4,
so the upper-layer temperatures are the least certain part of the structure.
