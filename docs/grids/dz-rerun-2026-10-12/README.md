# DZ grid rerun with updated physics

An updated-source rerun is authorized for the **322 existing DZ requests**.
The selected grid remains **317 numerical completions, four nonconvergences
and one infrastructure interruption**. The rerun has not changed those selections.
Original spectra, settings and unsuccessful attempts remain available.

Pilot job **59720137** is queued in the ordinary shared CPU queue. Its eight
production-quality requests cover all three existing metal patterns and
6,000–60,000 K. Four fixed-spectrum checks run before any new atmosphere starts.
The remaining **314 requests** are prepared but have not been submitted. They
require eight verified pilot completions, measured runtime and memory, and
fresh accounting that covers the full allocation.

The rerun uses source **1386029e9a911d9d6f7977cfb30af2f24e5f7545**, which includes
[upstream PR25](https://github.com/kareemelbadry/OpenWD/pull/25). It keeps every
explicit old request value, including production quality, gravity, H/He,
metal abundances, numerical sampling and physics-selection policy. Three
added source options are recorded: Kurucz electron Stark broadening, unified
Si II–He profiles and additional Si II Kurucz transitions. Changes relative
to older spectra can involve several source updates and cannot identify one fix.

All new models use public cold starts and require native convergence. Numerical
tolerances and iteration limits are unchanged. Each pilot has a three-hour
wall limit and a 3 GiB memory limit. Bulk model time limits will be set from
measured pilot and historical runtimes, between one and three hours. Resource
stops, numerical failures, unsupported cases and requests that could not start
will remain separate. There is no automatic retry loop.

## Cost and limits

The completed five-job phase used **7.0207 charged CPU node-hours**. The new
pilot has a conservative maximum of **0.40625 node-hours**. A later bulk job
would reserve at most **5 node-hours**, for a combined phase upper bound of
**12.42696 / 13**. That bound is not an actual new charge. Final accounting will
include setup, source checks and idle time, using allocated Slurm resources.
[NERSC charging policy](https://docs.nersc.gov/jobs/policy/) describes fractional
shared-node charging.

The earlier two collaborator benchmarks used standard quality and took about
9.5 minutes. The archived production grid is more expensive: its first 293
accepted cases had a median runtime of **65.09 minutes**, a 90th percentile of
**106.59 minutes**, and a total of **346.08 core-hours**. The pilot will test how
well those measurements predict the updated-source run. No completion time for
all 322 is guaranteed.

This rerun does not expand the DZ grid or establish independent depth
convergence, physical accuracy or an observational abundance fit. The
[separate two-model comparison](../dz-validation-2026-10-12/README.md) still
has a large cool-model UV disagreement with the collaborator's synthetic
reference. Rerunning the grid does not resolve that disagreement by itself.
Verified products and old/new comparisons will be added after collection.
The PR remains unmerged, and releases remain unpublished drafts.
