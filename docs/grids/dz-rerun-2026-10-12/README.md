# DZ grid rerun with updated physics

The first **eight production-quality DZ reruns passed** their native atmosphere
and independent spectrum checks. Their downloaded products passed source,
configuration, cold-start, array and file-hash verification. The selected grid
still has **317 numerical completions, four nonconvergences and one infrastructure
interruption from 322 requests**. These pilot spectra are separate from that
saved selection; original products and unsuccessful attempts remain intact.

## Verified pilot

Pilot **59720137** completed successfully after 7,751 allocated seconds.
Required fixed-spectrum checks for DA 5,000 K, DB 22,000 K, DZ J0738 and DZ PG1225
passed before model dispatch. All eight requests used their original production
settings and physics-selection policy, public cold starts, one numerical thread,
unchanged native tolerances and iteration limits, a three-hour time limit and a
3 GiB memory limit. All 96 remote/local file hashes and 72 arrays in eight native
NPZ files were verified. The compact 2,648,677-byte pilot archive matches its local
and private persistent copies. The final combined draft download is still pending.

| Metal pattern | Temperature (K) | log g | Runtime (minutes) | Peak RSS (GiB) |
|---|---:|---:|---:|---:|
| PG1225 | 6,000 | 8.0 | 46.61 | 0.875 |
| PG1225 | 10,000 | 8.0 | 109.44 | 0.858 |
| PG1225 | 60,000 | 8.0 | 28.61 | 0.844 |
| J0738 | 20,000 | 8.0 | 58.12 | 0.887 |
| J0738 | 40,000 | 8.0 | 38.18 | 1.046 |
| J0738 | 7,000 | 7.25 | 85.02 | 0.921 |
| GD40 | 10,000 | 8.0 | 87.08 | 0.845 |
| GD40 | 25,000 | 8.5 | 119.81 | 1.000 |

[Full inputs and native certificates](PILOT_RESULTS.json) ·
[Archive verification](PILOT_ARCHIVE_RECEIPT.json)

One J0738 request retains its original held-out role. These are eight existing
coordinates, not eight new grid points. Passing their numerical checks does not
establish independent depth convergence, physical accuracy or an abundance fit.

## Bounded continuation

Bulk job **59725915** started at **01:29:39 Pacific on October 13** on
`nid006992`, in ordinary CPU QOS `regular_1`. It covers **310 remaining
requests**. The allocation has 256 logical CPUs, 448 GiB and billing256/256. Four requests are deferred because their full recorded time limits
do not fit the conservative planning window. They are unattempted, not numerical
failures. A fresh accounting check after bulk settlement will determine whether
a small completion batch can cover them. No extra allocation is submitted now.

The original five-hour plan failed its timing check. The revised forecast uses
measured pilot ratios separately for each metal pattern, never credits speedups,
and retains padding for runtime and resource caps. It predicts about 5.05 hours
for the selected batch and reserves **5.75 hours**. Eight pilot cases do not
cover every abundance, gravity or packing effect, so this is not a completion
guarantee. Late requests without enough time for their full cap will remain
explicitly not started. There is no automatic retry loop.

The bulk uses 128 physical workers, one thread each, a 448 GiB allocation and
3 GiB memory limits per worker. Recorded model time limits range from one to
three hours. Native tolerances and iteration limits remain unchanged. Required
fixed-spectrum checks passed again before dispatch. Scientific cold starts began
at **01:37:34 Pacific**. The first owned read verified 128 exact requests
running on distinct physical cores with one numerical thread each; 182 planned
requests await worker slots. The four deferred records are separate unattempted
requests and are not counted as scientific attempts. The allocation cap ends
at **07:14:39 Pacific**; that is a resource limit, not a completion estimate. Numerical failures, unsupported
cases, timeouts, memory stops and requests that could not start remain separate.

Both stages use source **1386029e9a911d9d6f7977cfb30af2f24e5f7545**, which includes
[PR25](https://github.com/kareemelbadry/OpenWD/pull/25). Every explicit old request
value is retained. Three added source options are recorded: Kurucz electron
Stark broadening, unified Si II–He profiles and additional Si II Kurucz
transitions. Comparisons with the original spectra span several source updates
and cannot identify one cause. Accepted final products and old/new comparisons
will be added after collection; the rerun does not expand the grid.

## Cost and remaining limits

The prior five-job phase cost **7.0207074653 CPU node-hours**. The pilot adds
**0.15138671875**, from 7,751 seconds and an allocated billing fraction of 18/256.
The actual settled total is **7.1720941840**. Adding the full bulk reservation gives
an upper bound of **12.9220941840 / 13**, leaving **0.077905816** unreserved.
The bulk's actual charge is not yet known. Accounting includes allocated setup,
source checks and idle time. [NERSC charging policy](https://docs.nersc.gov/jobs/policy/)
describes fractional shared-node charges.

The earlier two collaborator benchmarks used standard quality and took about
9.5 minutes. Production models are more expensive. The original first 293 accepted
production cases had a median runtime of 65.09 minutes, a 90th percentile of
106.59 minutes and a total of 346.08 core-hours. Current pilot runtimes span
28.61–119.81 minutes. A full 322-model completion within this budget is not promised.

The [separate two-model comparison](../dz-validation-2026-10-12/README.md) still
has a large cool-model UV disagreement with the collaborator's synthetic
reference. Rerunning the grid does not resolve that disagreement by itself.

## Recent upstream work

[PR26](https://github.com/kareemelbadry/OpenWD/pull/26) is merged. It adds optional,
off-by-default H₂ Lyman/Werner fluorescence for DA and DAZ. Inspection of the
patch found no DZ-specific physics changes, so this rerun keeps its tested source.
[Draft PR27](https://github.com/kareemelbadry/OpenWD/pull/27) adds nine HST/COS
comparisons at fixed published parameters, with continuum, velocity and kernel
choices recorded. It still shows strong Si II wing residuals in GD40/G241−6.
Those reported checks have not been independently reproduced here and do not
establish physical accuracy across this grid. PR18 remains unmerged and all
model releases remain unpublished drafts.
