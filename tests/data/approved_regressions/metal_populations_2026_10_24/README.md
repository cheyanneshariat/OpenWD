# Metal population normalization, 2026-10-24

These fixed-atmosphere DZ outputs follow the reviewed population repair on
main `6e664591` (PR #38). They retain every original structure array,
configuration, wavelength and comparison tolerance. Existing approved
references remain untouched, and the regression test also recovers the
PR #38 spectra under the scoped previous population convention before
checking the entire older approval chain.

The largest relative changes are 0.0049231% for PG1225 at 2659.76 A and
0.364745% for J0738 at 1222 A. Flux integrals over the stored wavelength
ranges change by +0.0002631% and +0.0168186%. Component reversions in the
manifest attribute nearly all of this to TOPbase populations, rather than
the line/profile changes already released in PR #38. The complete prior
spectra are recovered exactly on the capture runtime.

The correction uses the same occupation-weighted level sum in the numerator
and partition denominator. Identified native fine-structure and LS terms
have normalized populations. Unidentified aggregate model levels retain a
documented effective-level approximation; this does not recover unknown
superlevel constituents. Low levels retain the released neutral reservoir.

Capture is offline via `research/capture_metal_population_controls.py`.
The tests never run the builder. The manifest includes input/output hashes,
source hashes, request identities, runtime, measured differences and separate
component reversions. The capture includes no atmosphere relaxation and
provides no cold-start certificate. Protected PR qualification remains
required. No J1637 reference was changed: all four fixed-spectrum controls
remain within their original tolerances.
