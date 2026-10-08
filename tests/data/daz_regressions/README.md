# DAZ paper controls

These contain only OpenWD synthetic spectra and atmosphere arrays for G149-28
and GALEX J1931+0117, selected from the September 1, 2026 paper-figure products
over 3700–4500 Angstrom. No observed spectra are redistributed. Original-file
SHA256 values and provenance are stored in each NPZ.

`tools/import_daz_controls.py WORKSPACE` records the provenance and copies
the original samples without rescaling, smoothing, fitting, or synthesizing
a replacement reference. The configuration records the figure's Stout line
strengths, Unsold metal-line widths, and Stark-only Lyman profiles, not the
later research defaults. The 40-layer structures have no current certificate.
Their historical convergence claims are deliberately not imported.

The tests are fixed-atmosphere spectral controls. Independent fresh DAZ tests
must pass the current physical gates; these fixtures are never their inputs.
