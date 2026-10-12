# Default Kurucz electron Stark fixed-atmosphere controls (2026-10-11)

These controls accompany the requested promotion of `kurucz_electron_stark`
to the DZ and DAZ default. With it on, Kurucz electron-impact Stark constants
are attached to Stout E1 lines that have no Stark width of their own; see the
DZ and DAZ guides for the observational comparisons behind the change. They
are synthetic regression outputs, not observations or independently qualified
atmosphere structures.

The four protected DZ and DAZ structures, configurations and wavelength grids
are unchanged. The previous flux controls are kept:
`../kurucz_crii_2026_10_08/fixed/dz-*.npz` and `../fixed/daz-*.npz`. The
tests compare the new default with these arrays, using the same tolerances as
before: 2e-6 plus band limits for DZ, and 0.1% for DAZ. In the same test,
disabling only `kurucz_electron_stark` must recover each previous control to
its original tolerance, so the update cannot hide unrelated changes. At
capture the disabled option reproduced the previous arrays to between 1e-13
and 7e-9.

| Fixed historical structure | Largest fractional pixel change | 1150--3000 Å band change | 3500--7000 Å band change | 7000--300000 Å band change |
| --- | ---: | ---: | ---: | ---: |
| PG 1225-079 (DZ) | 0.21% | -0.0049% | -0.00001% | -0.00002% |
| SDSS J0738+1835 (DZ) | 98.6% | -1.13% | -0.078% | -0.10% |
| G149-28 (DAZ) | 0.71% | -- | -0.0002% | -- |
| GALEX J1931+0117 (DAZ) | 14.2% | -- | -0.010% | -- |

The large J0738 pixel changes are single pixels of the coarse grid at
939--979 Å, among strong far-UV metal lines of this heavily polluted star. The
GALEX J1931 changes are Si II and other ionic line profiles: absorption moves
from line cores into wider wings. These historical structures have not been
re-relaxed for the broader lines, so this capture is **not a cold-start or
flux-equilibrium qualification**.

`manifest.json` records the input, previous-control and output hashes, the
request configuration, atomic-line provenance, the source hashes (including
the bundled Stark table) and the measured changes. Capture used
`research/capture_kurucz_stark_controls.py`, which refuses an existing output
directory. Tests and CI never recapture.
