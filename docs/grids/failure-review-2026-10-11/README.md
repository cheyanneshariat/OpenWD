# Failures and timeout follow-up

The saved selection has **81 numerical failures or worker crashes**: 73 failed
native equilibrium checks, five numerical depth-grid failures and three crashes.
These are investigation flags, not 81 demonstrated code bugs. The
[parameter table](flagged-models.csv) also retains 13 timeouts, 22 unsupported/domain
cases and two limited magnetic profiles. It excludes the 20 not-started entries
in that saved snapshot; the separate hot batch has since attempted eight of them.
[The summary](SUMMARY.json) records the exact selection hash and denominators.

## Priority cases

- DAB/DBA SIGSEGV at 15,000 K/log g 9/log(H/He) +3 and
  20,000 K/log g 9.5/log(H/He) −5. Inspect the native stack and data access.
- DZ SIGSEGV at 40,000 K/log g 7.5. Earlier partition warnings do not prove
  the crash cause. Keep the full abundance pattern and raw attempt.
- DB local-energy and temperature-stationarity failures. All 13 failed DB
  attempts in the detailed six-hour review failed both checks; ten also failed
  lower-boundary screening. Good total-flux balance alone is insufficient.
- DA at 4,750 K/log g 7.75 stalled and failed flux, local energy, stationarity
  and lower-boundary checks. The other saved DA failures need their raw logs
  inspected before a cause is assigned.
- Numerical depth-grid errors and unsupported inputs remain separate.
  The 17 unsupported DAB requests select a cool molecular workflow restricted
  to log g 8. Do not substitute atomic physics to fill these points.

The [cold DB/Montreal discrepancy](../db-cold-2026-10-09/README.md) remains a
separate physical-validation question. The ten [DAB interpolation misses](../refinement-2026-10-11/README.md)
also remain separate: their direct atmospheres passed the native checks.

## Does more time look useful?

The [timeout table](timeout-priority.csv) measures the last hour of all 13
ended six-hour resource stops. Most DB cases have nearly stationary or worsening
local energy. They do not justify a blanket time extension.

Two cases merit an uncertain, bounded diagnostic:

| Family | Teff, K | log g | Evidence and limitation |
|---|---:|---:|---|
| DB | 7,000 | 6.0 | Changed phase and reduced energy residual from about 1 to 0.254; the last 20 updates worsen |
| D6 | 15,680 | 6.3 | Only 18 preconditioner updates fit six hours; flux improves, but local energy ends at 0.0403 and grows |

Job **59699007** is queued with four fresh cold attempts: an original-source
control and a current-upstream run at each coordinate. Each has an eight-hour
resource cap. [Full requests and resource limits](diagnostic-requests.json) preserve
the physical inputs, native tolerances and iteration limits. The controls test
extra wall time; updated source/data is a separate comparison. Required
fixed-spectrum checks withhold any source branch that fails preflight.

Original sources are c8329d86 for DB and cfe2d2ff for D6. Updated source is
14e2067. The complete source changes include D6 Rosseland-probe reuse and
updated shipped data; a result cannot be attributed solely to the newest thermal
change. No old attempt or accepted product was modified.

A conservative reservation bound of 4.25 node-hours gives a total phase bound
of **10.755**, below the authorized 13. Models must pass their native checks
before acceptance. A timeout, queued job or worker crash is not a converged
spectrum. Numerical completion remains distinct from physical accuracy and
independent depth convergence.

[Current grids](../README.md)
