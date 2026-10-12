# Default unified Si+-He profile fixed-atmosphere controls (2026-10-11)

These controls accompany the promotion of `si_ii_he_unified_profile` to the DZ default.
It replaces the Lorentzian neutral-He broadening of Si II 1260/1264/1265 with the bundled
unified profile. See the DZ guide for the COS comparisons of five DBZs behind the change.
These are synthetic regression outputs, not observations or qualified atmosphere structures.

Only the two DZ structures are affected. DAZ hydrogen atmospheres contain no helium and are
unchanged. The structures, configurations and wavelength grids are unchanged. The previous
controls in `../kurucz_stark_2026_10_11/fixed/` are kept. The tests compare the new default
with these arrays using the same tolerances as before. In the same test, disabling only
`si_ii_he_unified_profile` must recover each previous control to 2e-6; at capture it did so
exactly.

| Fixed historical structure | Largest fractional pixel change | 1150--3000 Å band change |
| --- | ---: | ---: |
| PG 1225-079 | 0 (its grid does not sample the triplet) | 0 |
| SDSS J0738+1835 | 72.7% (coarse-grid pixel in the triplet wing) | -0.82% |

`manifest.json` records the input, previous-control and output hashes, the request
configuration and the source hashes, including the profile table. Capture used
`research/capture_si_he_unified_controls.py`, which refuses an existing output directory.
These structures were not re-relaxed; this is **not a cold-start qualification**.
