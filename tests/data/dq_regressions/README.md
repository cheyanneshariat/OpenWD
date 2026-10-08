# DQ fixed-state spectral control

`j1235-fixed.npz` contains the 41-layer atmosphere and 79 unscaled flux samples
from the independently qualified J1235 research cold run of 2026-09-17.
`manifest.json` records parameters and source/output checksums.

This fixture is used only by `test_dq_spectral_regression.py` to detect changes
in refractive synthesis and material physics. It is not observed data, is not
proof of physical accuracy, and never initializes a cold-start test.
`test_j1235_public_true_cold_and_independent_spectrum` starts from parameters
alone and is separately marked as a slow canary.

`j1311-completed-fixed.npz` preserves all 30,000 optical samples from the
pre-promotion 2024 C–A + completed-Swan warm model. Its independent broad-band
check passed the unchanged luminosity tolerance; production packaging reproduces
its spectrum bit-for-bit. `completed-manifest.json` records the source hashes.
This separate fixture protects the new default without replacing the historical
control. Rebuild with `tools/build_dq_completed_regression.py --source
results/dq-continuum-followup/j1311-combined-warm --output NEW_DIRECTORY`.

## Re-approval 2026-10-01 (physics revision `openwd-0.1.3-qmhd-undoubled-v5-flux-conserving`)

Both flux arrays were recomputed on the unchanged stored structures with the
current code and re-approved by the user. The previous hashes are recorded
under `reapprovals` in each manifest. The changes come from the shared
metal-line physics of the 2026-09-30 DZ audit, attributed by reverting one
change at a time on the J1311 structure:

- J1311 (5529 K): flux falls by 20% at 3800-4000 A and 12% at 4000-4500 A,
  and by about 1-2% redward of 4500 A. This is entirely the compiled
  line-kernel change from a wavelength-symmetric pseudo-Voigt to an exact
  Voigt profile in frequency, as impact theory defines it. The far red wings
  of strong UV carbon lines carry (lambda/lambda0)^2 more opacity (about 2.6x
  at 4000 A for C I 2479 A). Reverting the dense-He ionization shift or the
  ground-term photoionization fix changes nothing. Against the SDSS spectrum
  (Blouin & Dufour 2019 extraction, red-normalized, 10 A smoothing) the RMS
  improves from 0.086 to 0.062. The published Blouin & Dufour model reaches
  0.050. The impact approximation in such distant wings remains the main
  uncertainty.
- J1235 (9347 K): under 1% at 2000-100000 A, plus 25-30% less flux at 1000-1100 A.
  The far-UV change is consistent with C I Verner photoionization now
  applied to the whole ground term.
