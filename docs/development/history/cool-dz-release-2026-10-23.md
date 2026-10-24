# Cool-DZ release preparation, 2026-10-23

[DZ model guide](../../models/DZ.md) · [Development guide](../README.md)

This candidate builds on PRs 35 and 36 and the local cool-DZ physics work.
Full protected PR qualification is requested with this release; the local
comparisons below do not stand in for that qualification.

## Physics and defaults

| DZConfig option | Previous default | Release default |
| --- | --- | --- |
| dense_helium_eos | ideal | reos3 |
| dense_helium_free_free | off | rogers-young |
| helium_minus_free_free | automatic | partial-wave |
| mg_i_b_he_unified_profile | False | True |
| mg_ii_he_unified_profile | False | True |
| ca_ii_he_unified_profile | False | True |
| na_i_he_unified_profile | False | True |

The shared He-REOS.3 material interpolation now interpolates the nonideal
part smoothly between isotherms. The computed partial-wave e–He shape
continues He-minus free-free absorption below the 0.5063-micron John-table
edge. Computed unified Mg I b, Mg II h/k, Ca II H/K and Na I D profiles
include the dense-fluid correction and preserve their physical band areas.
The new tables cover 3000–12000 K and n_He = 1e19–1e23 cm^-3.

The neutral-He excluded-volume prescription acts on Rydberg levels rather
than suppressing the low-level reservoir. Table-path lower populations use
the corresponding occupation convention. The lower-boundary controller can
deepen the domain earlier after screening failures and restart the phase
sequence; the physical certificate tolerances are unchanged.

Two repairs are taken from cheyanneshariat's draft PR 37:

- Ca II CRD uses atomic transition centers and a fixed integration grid.
- RAP tables retain their absolute excitations and photon grids, including
  supported zero-to-positive ramps. The Fe ground-term centroid is counted
  once, and the tabulated cross sections are not clipped at a mean binding.

The other PR 37 population-normalization and MALI changes are outside this
candidate. This release retains its separately tested Rydberg-only neutral-He
prescription.

## Observational comparisons and limits

The preparation workspace reports the following broad-band comparison scores
at published stellar parameters. The score is the RMS of log(data/model),
after normalization over 6500–9000 Angstrom, with no clipping. These are ten
parameter sets, including two parameter choices for WD 2251-070. They do not
establish a uniformly accurate abundance grid.

| Parameter set | Previous defaults | Release candidate |
| --- | ---: | ---: |
| WD 2251-070, Blouin et al. (2019a) | 0.600 | 0.067 |
| WD 2251-070, O'Brien et al. (2026) | 0.899 | 0.257 |
| LP 658-2, O'Brien et al. (2026) | did not converge | 0.104 |
| vMa2 | 0.139 | 0.046 |
| SDSS J0744+4649 | 0.284 | 0.189 |
| SDSS J0916+2540 | 0.176 | 0.131 |
| SDSS J0956+5912 | 0.068 | 0.050 |
| WD J0255+0237 | 0.031 | 0.035 |
| SDSS J1227+6330 | 0.046 | 0.062 |
| G137-24, Hollands et al. (2017) | 0.087 | 0.153 |

Several objects improve substantially, while three parameter sets worsen.
G137-24 still has excess absorption around 3700–4150 Angstrom. The candidate
does not include the proposed He-induced Ca I 1S–1D absorption near
4500 Angstrom. The older Mg I 2852 and Ca I profile-table limits remain
relevant to cool dense atmospheres. Published-parameter comparisons and
equation certificates do not by themselves establish depth convergence or
validate extrapolation outside a table's physical domain.

## Frozen spectral controls

The new DZ spectra are separately versioned in
[the dated approval directory](../../../tests/data/approved_regressions/cool_dz_release_2026_10_23/manifest.json).
Atmospheric inputs and wavelength grids remain fixed. Maximum single-sample
relative changes are 21.33% for PG1225 and 55.28% for J0738, primarily in
line/UV samples. Band-integrated changes are at most about 0.31% for PG1225;
J0738 changes by about -3.8% over 1000–3500 Angstrom and at most 1.15%
redward. These are differences between model versions, not errors against
observations. Earlier DZ approvals remain unchanged and are reproduced in
test-only scopes using the preceding defaults and physics.

The DQ J1235 and J1311 controls retain their stored atmospheric structures.
Their manifests append the preceding hashes and the reason for reapproval:
the shared smooth REOS3 interpolation. J1235 changes by at most 0.015%;
J1311 has band-integrated changes below 0.8%, with one sample changing by
about 6.4%.
Both DQ controls run in the fixed-spectrum CI tier, including the completed
J1311 default calculation.

The fresh PG1225 cold canary retains its stellar parameters and production
quality while selecting current DZ defaults, including REOS3. Its historical
fixed-spectrum configuration still explicitly selects the ideal EOS; the
cold canary no longer inherits that preceding choice.

The J1637 Fe RAP approval is separately versioned as
`tests/data/d6_regressions/warm_v4_rap_absolute`. It uses the same archived
48-layer cold-source capsule, abundances, prescribed perturbation and
25,000-line budgets as v3. Earlier capsules and all trajectory/spectrum
tolerances remain unchanged. Historical scopes restore the preceding RAP
reader as well as their original atomic policies. The current version
recomputes two full-resolution warm updates and the fixed optical spectrum.
Its archived cold certificate belongs to its original source; bounded warm
updates do not certify present-code cold convergence.
The fixed-spectrum change from v3 is at most 0.021918%, with an integrated
optical-flux change of -0.005255%, within the explicitly approved 0.022%
maximum. The replay spectrum tolerance remains 2e-6.

The offline capture records hashes, source revision, runtime, full-profile
differences and the spectral change. Tests only compare frozen artifacts;
they never rebuild them. The local preparation audit and the eventual PR
checks record their actual outcomes separately.

The first protected CI run passed all 24 cold-model cases and the normal
test/spectral jobs. Its three historical J1637 trajectories passed in about
97, 99 and 88 minutes, leaving insufficient time for the fourth version
under the shared 330-minute command guard. The warm checks now run as four
separate CI jobs, each retaining the same two updates, full resolution,
reference arrays, tolerances and 330-minute command guard. Qualification
requires all four; no prior-head pass is substituted for a fresh run.
