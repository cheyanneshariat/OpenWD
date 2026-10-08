# Corrected DA cubic-default controls

These new snapshots protect the user-approved monotone cubic default after
the undoubled Q-MHD correction and the cold-start paper comparisons. The
older `../spectral_regressions` files have not been replaced.

- 12000 K, log g = 8: 40-layer public cold start with no synthesis override.
  The equilibrium certificate passes; broad synthesized flux is 1.00015858
  of sigma T_eff^4 (linear synthesis on the same state gives 0.97966294).
- 20000 K, log g = 8: the certified 100-layer corrected Figure 1 cold state,
  with the previously checked, independently closed cubic synthesis.

`manifest.json` records hashes of the original calculations. The capture
script only selects existing evaluated wavelengths and copies the saved
states; it never calculates a reference spectrum from the test invocation.
The same 0.1% absolute-spectrum and continuum-normalized Balmer-profile guards
are retained. The tests exercise `compute_da` with no method override.

These are synthetic regression data, not observed or Koester spectra.
Fixed-state reconstruction is a fast regression check, not a new cold-start
convergence certificate. Full-grid and physical adequacy remain separate.
