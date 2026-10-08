# Reviewed microfield and DA synthesis controls (2026-09-11)

These are synthetic OpenWD outputs, not observations or published model grids.
They protect the user-approved undoubled Q-MHD critical field and the monotone
cubic **DA** synthesis default. DB, DAB, DZ and DAZ retain linear synthesis.
No stellar parameters, convergence tolerances or spectral tolerances were fitted.

The historical files in `../spectral_regressions` and `../daz_regressions` are
unchanged. They remain the structure inputs for `fixed/`; approved flux arrays
are separate. `cold/` supplies comparison outputs only: tests always calculate
a fresh atmosphere before reading them, and still require all independent
equilibrium checks. All cold reference structures come from separately verified
corrected cold states, including same-run lower-boundary extension where needed.
No atmosphere from another model is required as a solver input. Individual
origins and certificates are recorded in the manifests. Reconstructing a
reference for synthesis is never itself a new convergence certificate.

Each manifest records the source/data hashes, historical-input and output
hashes, interpolation choice, source closure and (for fixed-state controls)
the change from historical flux. The capture script
`research/capture_release_controls.py` refuses existing output directories and
is never invoked by tests or CI. An explicit physics/numerics review and user
approval are required for future updates, not just a failing test.

The fixed-state checks retain the 2e-6 array comparison (0.1% for the DAZ
controls) and common band limits. Cold-model checks retain their 0.3% spectral
energy-scale limits and 2e-5 pressure/mass limits. Independent numerical
quadrature, occupation-probability and buffer-lifetime tests do not depend on
these captured outputs. The separate `../da_cubic_regressions` controls also
protect normalized Balmer profiles and broad sampled stellar-flux agreement.

See the [microphysics audit](../../../docs/development/history/microphysics-audit-2026-09-10.md)
for the reviewed cold-start and literature/observational comparisons, including
remaining limitations. Historical spectra are preserved as evidence, not
asserted to be unchanged by the approved physics correction.

## Update 2026-10-01: flux-conserving numerics (physics revision `openwd-0.1.3-qmhd-undoubled-v5-flux-conserving`)

User-approved recapture after a numerics review. The emergent spectra of every
family conserve flux to 0.0-0.75% of sigma Teff^4. Before, the errors were up
to 16% (DZ) and 2-3% (DA/DB/DAB). Two settings are now on by default in DA,
DB, DAB, DZ and DAZ:

- `synthesis_transfer_depth_refinement=4` subdivides each structure depth
  interval for the final formal solution.
- `photospheric_depth_concentration=1` concentrates the structure depths
  across 0.01 < tau < 10 at fixed point count.

DZ/DAZ structures also absorb the opacity-sampled synthesis line list. The
fixed controls change by the refined formal solution (about 1-3% for DB/DAB,
up to 7% blue continuum for the coarse historical G149-28 checkpoint). The
cold controls change by the new depth grid. Every replacement cold structure
is a fresh public cold start with a verified certificate (see the
manifests). The DA and DB canary work guards were raised from 45 to 70 and
from 60 to 90 iterations. These are work limits, not tolerances. The
spectral and structure tolerances are unchanged.
