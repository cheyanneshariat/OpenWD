# Precomputed model grids

OpenWD calculates an atmosphere and spectrum for a chosen temperature, gravity
and composition. These saved grids let users reuse those calculations. The
selected grid table contains **2,368 numerical completions from 2,499 requests**.
Each completion passed the atmosphere and spectrum checks recorded for its
source version. Numerical completion alone does not establish physical accuracy
or accurate interpolation between grid points.

DAB/DBA has **637 accepted coordinates from 656 requests**. The October 11
refinement adds 32 newly calculated knots and eight reused centers in eight
selected patches. Its 32 independent test spectra remain outside the predictor
mesh. Together with three earlier validation coordinates, there are
35 separate DAB test coordinates. Earlier settings, attempts and archives remain
unchanged.

![Current grid progress](grid_progress.png)

Each title gives numerical completions divided by requests. Green circles passed
the recorded checks. Red crosses failed a numerical check or encountered an
invalid numerical depth grid. Grey squares reached a time limit. Grey triangles
could not run their requested configuration. Grey circles were not started in the saved selection. Dark grey crosses mark worker crashes.
The [four-DO/four-DAO batch](hot-2026-10-12/README.md) finished with four numerical
completions and four six-hour timeouts. The figure includes those outcomes in
place of the eight previously unattempted rows.
Timeouts remain unfinished. Different compositions can share temperature and
gravity, so symbols can overlap. Blank regions were not requested.

| Family | Numerical completions | Requests | Latest selected source |
|---|---:|---:|---|
| DA | 808 | 852 | October 8 + native validation supplement |
| DB | 592 | 638 | October 8 + cold, validation and retry supplements |
| DZ | 317 | 322 | October 8 + October 11 retries |
| DAO | 3 | 15 | Three verified spectra; remaining gaps retained |
| DO | 5 | 7 | Five verified spectra; two timeouts retained |
| DAB/DBA | 637 | 656 | October 9/10 + October 11 retries and refinement |
| DAZ | 3 | 3 | October 8 |
| DQ | 2 | 2 | October 8 |

The eight-family figure covers 2,495 requests. The full table retains 2,499,
including four D6, DAH and PG1159 cases omitted from the figure. D6 now has
one accepted coordinate from one request. The production
DAB bank replaces the five earlier standard-quality DAB examples in these counts.
Four repeated DB comparison models are counted once at their original coordinates.
All previous selections remain in [dated snapshots](history/README.md).

The [final DO/DAO tier and phase overview](hot-extension-2026-10-12/README.md)
add the 60,000 K/log g 9 DO coordinate and a byte-identical DAO source check.
All five phase allocations are settled, with **7.0207 charged CPU node-hours**.
The phase produced 73 numerical completions and seven timeouts from 80 fresh
attempts. Repeated coordinates and independent test spectra are counted separately
from the selected mesh.

## Downloads and spectrum examples

[Contributor draft releases](https://github.com/cheyanneshariat/OpenWD/releases)
hold the original DA/DB/DZ/DAZ/DQ archives, the 326-spectrum DAB archive, the
four-model cold DB supplement, the 118-spectrum native validation supplement,
the 261-spectrum DAB abundance supplement, the 38-spectrum timeout-recovery
supplement and the 104-model DAB refinement archive. The last archive includes
64 new calculations and 40 earlier predictor inputs. The hot-batch archive
retains four accepted DO/DAO spectra and all eight attempts.
Draft assets require write access to the fork. They are not public downloads until
the releases are published. The original archives remain unchanged.

- [DAB/DBA coverage and spectrum sequences](dab-2026-10-09/README.md) vary
  abundance, temperature and gravity separately. The current selection has
  637/656 accepted points after the [temperature and gravity refinement](refinement-2026-10-11/README.md);
  the original archive contains 326.
- [New DAB knots and 32 independent refinement tests](refinement-2026-10-11/README.md)
  show how halving temperature and gravity spacing changes the errors.
- [New grid points and DAB interpolation tests](validation-2026-10-09/README.md)
  give all 136 follow-up requests, their full settings and remaining gaps.
- [Six-hour retries and the first DAO spectrum](timeout-recovery-2026-10-11/README.md)
  show all 68 outcomes, the nine recovered DAB points and all eight measurable
  earlier DAB interpolation tests. Eight of those targets are now predictor inputs
  in the new refinement.
- [New DO temperature sequence and matched source control](do-continuation-2026-10-12/README.md)
  add the 70,000 K/log g 8 coordinate under current source; the repeated
  60,000 K control has an identical native spectrum.
- [Four new DO/DAO spectra and four six-hour timeouts](hot-2026-10-12/README.md)
  retain native structures, full inputs and source-check scope.
- [Accepted D6 spectrum and three eight-hour diagnostic timeouts](diagnostics-2026-10-12/README.md)
  retain the matched source controls and unresolved DB cases.
- [Two DZ benchmarks after the October 12 physics update](dz-validation-2026-10-12/README.md)
  separate matched fixed-structure spectra from two fresh cold starts. The cool
  UV disagreement remains unresolved; selected grid counts are unchanged.
- [Cold DB and Montreal comparison](db-cold-2026-10-09/README.md) preserve the
  earlier four-model demonstration and its unresolved spectral disagreement.

Wavelengths are vacuum Angstroms. Flux is surface Fλ in
`erg s^-1 cm^-2 Angstrom^-1`, without normalization or resampling.
Different families can use different wavelength grids.

The [failure and timeout review](failure-review-2026-10-11/README.md) lists
the unresolved coordinates, failed checks and bounded diagnostic follow-ups.

## Source versions and remaining limits

The October 8 grids use `cfe2d2ff99a434cd502696c1eb7b2f5daf8d11c5`.
The production DAB grids and their matching interpolation tests use
`0a74fdbe06596fc145f5169a3b239ddd67053141`. The earlier cold DB supplement
preserves its `0a74fdbe` and `54e401e2` records. The new DB, DA and hot pilots
use merged source `c8329d86c98f956617ec1bb3746181350492ce66`. The latest
DO continuation and DO/DAO extension use `70bca78e82e7b7dd5860ca3e337bd1f83919e09a`.
The D6 diagnostic uses `14e206729d06f0989e03a0d2f1744a9cc518a136`.
[Dataset records](dataset.json) and the full request tables identify each source.
Updating OpenWD does not recompute a saved archive.

Accepted models passed the required native atmosphere and spectrum checks.
The new DA point uses the existing `multigrid_initialization=True` option.
The repaired DAB point uses `photospheric_depth_concentration=2`; its earlier
failed default-mesh attempt remains recorded. Native physics, iteration limits
and acceptance tolerances were unchanged.

DB has 24 nonconvergence cases, five depth-grid errors, five material-domain
stops and 12 six-hour timeouts. DA has 44 numerical failures. DZ has four
numerical failures and one SIGSEGV crash. The selected D6 example now passes the native checks under source 14e2067.
Its original-source control and both DB diagnostic attempts reached eight-hour
limits; [all four diagnostic outcomes](diagnostics-2026-10-12/README.md) remain archived.
All nine DAB timeouts passed; its 19 remaining gaps are 17 unsupported
configurations and two earlier SIGSEGV crashes. The first DAO passed after 261.24 minutes.
The [latest hot batch](hot-2026-10-12/README.md) adds two DO and two DAO spectra.
Its other four cases reached their six-hour caps; these remain unfinished.

The DAB abundance mesh has one-dex spacing. In eight selected patches,
temperature spacing falls from 5,000 to 2,500 K and gravity spacing from 0.5 to
0.25. On 32 fresh independent spectra, linear interpolation improves from
10/32 to 22/32 passing both optical error targets. Ten still miss at least one
target. [The full comparisons](refinement-2026-10-11/README.md) retain these limits.
H/He is fixed within each patch; abundance interpolation and the full grid
remain unqualified for precision fitting.
DB uses the merged gravity repair in the retry, so its improvement cannot be
attributed to a longer time limit alone. The cold DB/Montreal discrepancy remains
unresolved. DZ expansion remains paused for matched external validation models.

Numerical completion does not establish physical accuracy, independent atmosphere
depth convergence, interpolation accuracy or calibrated parameter recovery.
Cool dense-helium DB models retain their experimental-physics labels.
The standalone DA mass-domain experiment remains separate from accepted grid models.

## Reproduce the figures

[The selected status table](progress.csv) preserves detailed outcomes alongside
plot categories. Run in the existing NumPy/Matplotlib environment:

```bash
python docs/grids/plot_grid_progress.py --output output/grid-progress
```

The stable model_id identifies the requested coordinate. The retry table joins
on case_id and records the actual attempt identifier and source/data digest.
Those digests are separate from the original bank identity. New refinement rows
record their native model IDs, full configurations and source/data digest in the
[refinement model table](refinement-2026-10-11/native-models.json).

The [DAB plotting script](plot_dab_spectra.py) reads the original archive and
optionally uses the current selection table for coverage. The
[validation plotting script](plot_grid_validation.py) reproduces the interpolation
figures and metrics from checksummed saved arrays. The [one-dex mesh script](plot_dab_refinement.py)
reproduces the October 10 abundance sequence and its dated coverage. The
[retry plotting script](plot_timeout_recovery.py) reproduces the current DAB
coverage, all eight interpolation comparisons and the first DAO spectrum.
These scripts perform no atmosphere calculation.

The [temperature and gravity refinement script](plot_dab_temperature_refinement.py)
reads the checksummed 104-model archive. It reproduces the new-knot map, paired
errors and two direct-spectrum examples, and rechecks all 256 saved predictions.

The [hot-spectrum script](plot_hot_spectra.py) checks the eight-attempt archive
and reproduces the four accepted spectra and the batch outcome figure.

The [final hot-spectrum script](plot_hot_extension.py) reproduces the two final
spectra from their checked archive.

The [DO temperature script](plot_do_temperature.py) reproduces the new
fixed-gravity sequence from its checked archive.

[The updated DZ rerun](dz-rerun-2026-10-12/README.md) is queued for an eight-model
pilot. The remaining 314 requests await verified pilot results and resource
measurements. The selected DZ counts above are unchanged.
