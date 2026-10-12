# Default Kurucz Si II missing-transition fixed-atmosphere controls (2026-10-12)

These controls accompany the promotion of `si_ii_kurucz_transitions` to the DZ and DAZ
default. It adds Kurucz Si II transitions between levels that Stout already contains but does
not connect, including the 3s3p2 2D - 3s3p(3Po)3d 2Fo lines at 1305.59, 1309.45 and
1309.73 Å. See the DZ guide for the COS comparisons of 11 polluted white dwarfs behind the
change. These are synthetic regression outputs, not observations or qualified atmosphere
structures.

The structures, configurations and wavelength grids are unchanged. The previous controls in
`../si_he_unified_2026_10_11/fixed/` (DZ) and `../kurucz_stark_2026_10_11/fixed/` (DAZ) are
kept. The tests compare the new default with these arrays using the same tolerances as before.
In the same tests, disabling only `si_ii_kurucz_transitions` must recover each previous
control; at capture it did so exactly.

| Fixed historical structure | Largest fractional pixel change | Band change |
| --- | ---: | ---: |
| PG 1225-079 (DZ) | 0 | 0 |
| SDSS J0738+1835 (DZ) | 5.4% | -0.006% (1150--3000 Å) |
| G149-28 (DAZ) | 0 | 0 |
| GALEX J1931+0117 (DAZ) | 2.3% | -0.001% (3500--7000 Å) |

`manifest.json` records the input, previous-control and output hashes, the request
configuration and the source hashes, including the bundled Kurucz Si II subset. Capture used
`research/capture_si_kurucz_controls.py`, which refuses an existing output directory.
These structures were not re-relaxed; this is **not a cold-start qualification**.
