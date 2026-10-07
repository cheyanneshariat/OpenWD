# Reviewed Stark density-boundary references — 2026-10-05

These three fixed-atmosphere outputs cover DA 4000 K, DA 5000 K, and DAB 9000 K.
The user explicitly approved separate versioned references after reviewing the
density-scaling correction in PR #7. The previous `../fixed/` outputs, all
historical structure inputs, and all cold-start controls remain unchanged.

The tables already include thermal Doppler broadening. Previously, a density
below the table range selected the boundary shape but scaled its wavelength
width with the lower density. That mismatch compressed the thermal core.
The correction holds the entire physical profile at the lower density limit.
Selected profiles inside and above the density grid remain bit-for-bit
unchanged against the pre-fix source at `a1bf936`.

The captures use the existing `research/capture_release_controls.py` machinery
and the public default synthesis on the original fixed structures. The same
gas-pressure and column-mass arrays are retained. The changed samples relative
to the preceding controls are 19/3000, 4/3000, and 7/3000, all in the far UV.
Maximum optical relative changes are 2.02e-9, 1.44e-9, and 5.19e-9 respectively.
No atmosphere was solved, flux rescaled, wavelength grid changed, or parameter
fitted during capture.

The array comparison remains `rtol=2e-6` and
`atol=1e-12 * max(expected_flux)`. The significant-flux and band-integral checks
also retain their existing tolerances. The new density/thermal-width tests
independently protect the correction. Tests and CI only read these frozen
outputs; they never regenerate them.

`fixed/manifest.json` records output, historical-input and previous-control
hashes, numerical source/data identity, synthesis configuration and source
closure. These records certify the provenance of a fixed-state comparison.
Full cold-start qualification remains a separate requirement.
