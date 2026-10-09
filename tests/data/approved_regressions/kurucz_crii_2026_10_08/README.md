# Default Cr II fixed-atmosphere controls (2026-10-08)

These controls accompany the requested promotion of the full Kurucz Cr II
missing-transition supplement to the default line list, following the
WD 1232+563, WD 1551+175 and WD 2207+127 HIRES comparisons in the DZ guide.
They are synthetic regression outputs, not observations or independently
qualified atmosphere structures.

Only the two protected DZ controls contain chromium. Their original
structures, configurations, wavelength grids and previous flux controls are
retained unchanged. The new default adds Cr II absorption and its one missing
level, so an equality test against the old flux would reject the requested
physics change. The current tests use these separate flux arrays with the
same 2e-6 tolerance and band limits as before. In the same test, disabling
only the supplement must still recover each previous control to its original
tolerance. Thus the update cannot hide unrelated changes to the old spectra.

| Fixed historical structure | Largest fractional pixel change | 1150--3000 Å band change | 3500--7000 Å band change | 7000--300000 Å band change |
| --- | ---: | ---: | ---: | ---: |
| PG 1225-079 | 18.67% | -0.2804% | -0.004045% | +0.000044% |
| SDSS J0738+1835 | 88.21% | -2.2737% | -0.06328% | +0.003920% |

The large pixel changes occur in added absorption lines. This capture uses
the full Kurucz file with SHA-256
`6f0c4d0e01421fb0549ddbcf5649af391ca9b930e4e83e693e662d5874505205`,
preserves all original Stout records, and is exactly equivalent to the explicit
missing-transition import used for the HIRES comparisons. Tests separately
check the recovered Cr II 3368.049, 3433.309 and 3585.294 Å lines, preservation
of other ions, and the added level's negligible partition-function change.
Both fixed formal solutions converge their source closure. These historical
structures have not been re-relaxed for the new line list; this capture is
**not a cold-start or flux-equilibrium qualification**.

`manifest.json` records the input/output hashes, request configuration,
atomic-data provenance, source/data identities and measured changes. The
source/data identities include installation paths by the model's standard
fingerprint convention; compare their hashes only within that installation.
The capture used the compiled backend, Python 3.9, NumPy 1.26.4 and SciPy
1.11.1, on the source based on OpenWD `33176c8`. The recorded numerical-source
hashes identify the Cr II implementation used for the capture.

To recapture for an explicitly reviewed future physics change, load each
unchanged `../../spectral_regressions/dz-*.npz` historical structure using
`load_atmosphere_checkpoint`, reconstruct its saved `DZConfig`, and call
`compute_dz(..., initial_atmosphere=atmosphere, relax_atmosphere=False)` on its
saved wavelength grid. Preserve the old controls and record the raw-Stout
comparison before accepting new arrays. Tests and CI never recapture them.
