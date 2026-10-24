# J1637 absolute-RAP reference

This version records the explicitly approved Fe RAP photoionization repair.
Its reference source is the unchanged, checksum-verified v3 capsule. The
original cold-source provenance, abundances, temperature arrays, 48-layer
grid, 25,000-line budgets and prescribed 1.5% perturbation are preserved.

The fixed spectrum changes by at most 0.0219181% relative to v3, at
4883.3 Angstrom, with -0.0052548% integrated-flux change on the stored grid.
The repair counts the supplied excitation once and retains the absolute
photon/cross-section support. No atmosphere is relaxed for this spectrum.

Two measured full-resolution warm updates are captured separately. They
exercise current equations and do not establish cold convergence. The
manifest retains the archived source certificate with its original scope
and records actual full-profile differences, runtime and source/data hashes.
The two updates change temperature by at most 2.0035 K (0.009049%), mass
density by 0.017100% and electron density by 0.007489% relative to v3;
gas pressure is identical. Their capture took 2206.20 seconds, plus
107.87 seconds for the fixed spectrum.

Earlier v1/v2/v3 inputs, outputs and manifests are unchanged. Historical
tests recover the preceding RAP reader in a test-only scope; current tests
use the repaired reader. Source-convergence, transfer, profile and spectral
tolerances are unchanged, including the fixed-spectrum relative tolerance
of 2e-6.
