# LTE preconditioner handoff and thermal-conditioning steps

The shared LTE solver can spend additional iterations on its approximate
convective-gradient objective after actual total flux is balanced. Those
iterations need not improve local radiative energy balance. The
`use_preconditioner_flux_handoff` argument to `solve_adaptive_lte_structure`
transfers that accepted atmosphere to the existing formal-flux completion
phase earlier.

The option is enabled by default (see
[Default enablement](#default-enablement-with-faster-thermal-conditioning)).
Pass `use_preconditioner_flux_handoff=False` to recover the previous
trajectory. It acts on convective LTE calculations with local energy
enforcement, including DQ, which calls this shared adapter; DO/DAO and PG1159
use other structure paths. The sections before that one record the evidence
gathered while the option was still off by default.

## Progress policy and final checks

The hook applies only to the first convective-gradient preconditioner. Two
consecutive genuinely accepted Newton updates must each have both actual
all-interface flux error and provisional residual below the existing flux
tolerance, while directly measured local energy error remains at or above
that tolerance. No new physical tolerance is introduced.

The initial residual cannot count as an update. A stationary callback, material
correction, nonfinite diagnostic or unchanged state cannot advance the streak.
Material and stationary callbacks reset it; rejected trials neither advance nor
reset it. A backtracked Newton direction counts only at its actual accepted
anchor. The token for each accepted callback is consumed once.

A triggered handoff returns the provisional segment as **unconverged**. It
retains the accepted state and changes no Newton acceptance, tangent, material
physics or coordinate mapping. The original formal-flux phase, bounded
continuations and local energy completion remain in place. The final
certificate still requires solver completion and the five existing checks:
all-depth flux, local energy, unrestricted temperature correction, scattering
source closure and lower-boundary screening.

Requested runs record `convective_preconditioner_flux_handoff` diagnostics,
including whether the policy was used and triggered, accepted Newton update
counts and per-anchor observations. Disabled runs remove an inherited record
for this current-solve diagnostic; they preserve other seed provenance.

## Contained evidence

The pilot used complete reduced-grid solves with the native extension and
six numerical thread settings pinned to one, at merged-main commit
`39664f1aa38d8b0a5f455b50ad6dea7f93c1de24`. Its only candidate change was the
accepted-state hook; the implementation benchmark exercises the production
option itself. Both arms share grids, material callbacks, budgets and each
composition's actual convergence tolerances. D6 uses a sampled J1637 accepted
checkpoint with materials rebuilt on the diagnostic grid; DB and DAZ use
public-preset initial material fixtures.

| Fixture | Preconditioner observations | Complete-solve residual requests | Complete-solve Jacobian requests | Largest relative diagnostic spectrum difference |
| --- | ---: | ---: | ---: | ---: |
| D6, 16 depths / 128 metal lines | 8 → 6 | 25 → 23 | 5 → 4 | 4.18e-6 |
| DAZ, 8 depths / 16 metal lines | 8 → 6 | 32 → 21 | 9 → 5 | 5.56e-7 |
| DB, 8 depths | 8 → 8 | 26 → 26 | 7 → 7 | 0 (bitwise equal) |

Counts include shared-solve initial selection, all phase work and thermal
sweeps; each arm also requests one separate endpoint Jacobian diagnostic at the exact returned
state. Exact-state cached transfer/tangent data may be reused; this is a
separate unpenalized-correction measurement, not an independent physics solver. Callback
observations can include a stationary check and must not be treated as Newton
update counts. DAZ's earlier handoff also avoids the thermal/local-energy
completion route used by its baseline; its final energy error is smaller.

All pilot endpoints pass the production certificate, separate endpoint measurements of
all five physical checks and finite-positive material checks. D6 and DAZ were
repeated in both execution orders; the saved endpoint arrays, histories and
work counts reproduce exactly for each variant. DB follows an identical
numerical trajectory. Reduced endpoints differ between D6/DAZ variants within
these checks; this is an algorithm change, not a bitwise-preserving cache fix.

The implemented option was then exercised in three additional baseline/candidate
pairs. Every retained per-arm state, R/J, unrestricted correction and spectrum
matches the pilot bitwise. Baseline metadata is identical; candidate metadata
adds only the current-solve handoff record. All six endpoints retain the same
certificates and work counts. A focused suite of 163 checks covers shared driver
acceptance, initial/stationary/material/rejected observations, accepted-state
streak reset, phase isolation, thermal completion, certificate gates, disabled
metadata routing and the actual DQ dispatch omission.

D6's three paired solve-time reductions were approximately 10–11%. DAZ's
were approximately 11% in the first pair and 42–45% in the alternating-order
repeat. Timings include Python profiling and per-segment report writes;
setup, separate endpoint diagnostics and fixed synthesis are separate.
The varying DAZ timings do not establish a stable full-model speedup. The
repeatable eliminated work is the stronger result.

Fixed public synthesis covers about 200 optical wavelengths and selected
line windows. Reduced structures do not match a standard public request;
the saved `fixed_synthesis_request_verified=False` is retained. These are
structure-grid convergence and narrow spectral comparisons, not independent
resolution validation or released-quality cold-start qualification.

The previously completed full J1637 baseline is motivation only: its recorded
steps 9 and 10 satisfy this progress predicate, while the preconditioner
continues through step 20 and local energy error grows. A changed cold solve
at the default 48-depth/25,000-line resolution has not been run. The broader tests below measure completion work on their
explicitly declared grids.

## Broader grids and public cold starts

The next serial screen extended DB to 16 depths/160 continuum points/128
lines, and checkpoint-derived D6 to 24 depths/160 continuum points/128 lines.
Both completed pairs passed all five checks. DB retained identical arrays,
histories, work counts and narrow spectra; the candidate handed off at the
same eight-iteration boundary as the existing cap and saved no work. D6
reported seven rather than eight preconditioner iterations, with shared-solve
R-only requests 29 → 28 and Jacobian requests 5 → 4. Its narrow-spectrum
maximum pointwise relative difference was 1.30e-7. This D6 test used the
retained checkpoint, not a cold initializer.

A separate DB control with 8 depths/80 continuum points/16 lines and a
two-iteration preconditioner budget exercised an enabled policy that never
triggered. Both complete trajectories and spectra were bitwise equal. A
16-depth/160-continuum/128-line DAZ pair exhausted a shared 420-second
pair deadline after baseline completed. Its first 15 durable observations
matched, but candidate had no final endpoint or spectrum. That attempt is
retained as inconclusive; it establishes no complete timing ratio or final
qualification.

Two public cold cases were then run in both variants, first with global
profiling and then without it. Each arm had its own 900-second cap. Both
protocols ran serially with the six numerical thread settings pinned to one;
the profiled pairs ran baseline first and the lower-overhead pairs candidate
first. Baseline omitted the option, and candidate injected only
`use_preconditioner_flux_handoff=True` at the shared solver entry. Every arm
started without an external atmosphere, temperature or mass state and used
the composition's actual continuum initializer, with no warmup call.

G149-28 used the unchanged standard DAZ preset: 8600 K, log g 8.10, its
canonical metal abundances, 40 depths, 300 continuum points, three angles,
the public default line policy and the full 18,901-point output spectrum.
J1637 used a genuine cold D6 request with an explicit research resolution:
24 depths, 2,000 structure lines, 2,000 formal lines, native 450 continuum
points, three angles and the full 43,001-point public spectrum. Its structure
transfer grid contained 22,230 wavelengths. The low-adapter depth override
is outside the nominal config fingerprint: this D6 pair does not qualify
the default 48-depth/25,000-line model.

The complete public-call measurements without global profiling are:

| Cold case | Baseline seconds | Handoff seconds | Descriptive reduction | Reported preconditioner iterations | Maximum pointwise relative full-spectrum difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| G149-28, standard DAZ | 226.877 | 211.543 | 6.76% | 18 → 11 | 6.611e-4 |
| J1637, reduced D6 | 317.072 | 262.575 | 17.19% | 12 → 7 | 3.866e-8 |

These timers include actual initialization, all solver phases and final
certification, public synthesis and saving. They exclude import/provenance
setup. Lightweight observational wrappers, copies and report/checkpoint I/O
remain inside the timers. One pair per protocol is descriptive, not a
statistical performance estimate. The separately profiled public times were
832.402 → 780.714 seconds for DAZ and 775.231 → 612.124 seconds for D6;
global profiling adds substantial overhead and its cost need not cancel
between variants. The measured I/O subset does not bound that overhead.

All eight cold endpoints completed through `require_convergence=True` and
passed the five actual original gates: flux, local energy, unrestricted
log-temperature correction, source closure and boundary screening. The
DAZ flux/energy/boundary tolerance remained 2e-3 and its temperature
tolerance 2e-4; D6 retained 3e-3 and 3e-4, respectively. Source closure
remained 1e-6. Both initial states, existing initial R/J arrays and all
source/data/native/runtime identities matched between variants. Before/after
identities remained unchanged. Saved public arrays and certificates agree
with the shared solver endpoint. These audits reconstruct recorded canonical
checks; they do not independently rerun the physical transfer diagnostics.

For each variant, the lower-overhead run exactly replayed its corresponding
profiled run: all retained callback atmosphere arrays, full metadata and
diagnostics, phase controls, shared seed and initial R/J, final atmosphere,
full saved metadata and spectrum text. Separately retained initializer-return
seeds also matched across the two lower-overhead arms. Initializer-internal
iterations were not observed by either protocol.

| Cold case | Total callbacks | Profiled shared-solve R-only requests | Profiled shared-solve Jacobian requests | Profiled true-absorption calls |
| --- | ---: | ---: | ---: | ---: |
| Standard DAZ | 33 → 28 | 51 → 58 | 22 → 17 | 90 → 87 |
| Reduced D6 | 20 → 15 | 34 → 29 | 6 → 4 | 40 → 33 |

The reference counts include shared-solve initial selection, thermal work
and final certification outside the nonlinear driver. They are observations
from the separately profiled, exactly replayed runs. The lower-overhead
worker directly counts driver requests only; its outer and constitutive
counts remain explicitly unobserved. Continuum-initializer entries and time
are recorded separately, without a claim to its internal work counts.

Reported preconditioner iterations and callbacks are not accepted Newton
update totals: DAZ baseline had 16 accepted updates, one rejected direction
and one stationary iteration. Candidate had 11 accepted preconditioner
updates and added six formal updates, with fewer thermal sweeps (15 → 10). D6 ended its baseline
preconditioner at its existing 12-iteration cap; candidate used seven
accepted updates before the handoff. Formal completion accepted seven updates in
both D6 arms. Earlier preconditioner exit therefore need not reduce every
category of later work.

The standard DAZ final temperature maximum relative difference was 5.756e-5;
its full-spectrum maximum difference normalized to the baseline peak was
1.580e-6. Reduced D6 corresponding differences were 1.203e-6 and 5.454e-9.
Passing identical finite-grid checks and comparing two spectra does not
establish independent mesh or spectral accuracy. The frozen local validation
bundles are `solver-handoff-broader-2026-10-07` and
`solver-handoff-unprofiled-2026-10-07`; they retain plans, workers, provenance,
checkpoints, capped attempts and standalone offline audits.

## DAH, DAB and cool DA/DB cold controls

A further serial screen tested public cold requests at log g 8, with original
standard module physics, 120 iterations, three angles and default full output
spectra. Each arm had an independent 300-second cap. The explicit diagnostic
starting grid was 16 depths and 80 continuum points; native boundary retries
retained their requested depths. These are not qualifications of the native
40-depth/300-continuum standard grids. The worker used no global profiler and
injected only the opt-in argument in the candidate.

| Cold case | Baseline public seconds | Handoff public seconds | Observed driver requests / Jacobians | Interpretation |
| --- | ---: | ---: | --- | --- |
| DAH, 8000 K, 0.02 MG | 45.844 | 30.240 | 57 / 13 → 34 / 6 | Earlier handoff saves work; 34.0% less time in this pair |
| DA, 5000 K, molecular physics | 49.131 | 52.027 | 34 / 5 → 53 / 6 | More formal work; 5.9% longer in this pair |
| DA, 4000 K, molecular physics | 49.007 | 49.950 | 17 / 4 in both | No trigger; identical numerical trajectory and spectrum |
| DB, 10000 K, first pair | 148.196 | 268.609 | 40 / 11 in both | No trigger; identical numerical work and outputs despite timing variation |
| DB, 10000 K, reversed repeat | 184.597 | 183.996 | 40 / 11 in both | Exact replay; the large initial timing difference does not reproduce |
| DAB, 12000 K, log H/He = -2 | 219.219 | 169.667 | 34 / 9 in both | Trigger coincides with existing eight-iteration cap; no work reduction |

All twelve endpoints in this table pass their actual original five-gate
certificates. Across each pair, the actual cold continuum initializer arrays,
first shared-solve seed and existing initial R/J match, as do config,
source/data/native/runtime and controls. Before/after identities remain
unchanged. Total driver requests include residual-only and Jacobian-bearing
requests; the Jacobian count is an included subset. Counts include all observed
parent shared-solve attempts;
outer initial selection, thermal/certification requests, constitutive work and
initializer-internal counts remain unobserved. Public timers include those
phases, full synthesis and saving, plus the observational wrappers and I/O.
Single pairs do not establish statistical speedups.

Weak-field DAH uses the original automatic ML2 structure with full magnetic
synthesis. Its preconditioner changes from 32 reported iterations (31 accepted
and a stationary check) to ten accepted updates and handoff. Thermal callbacks
fall from 16 to nine. Both 19,205-point spectra use identical wavelengths;
their maximum pointwise relative difference is 5.836e-6.

The 5000 K DA case shows why the current rule should remain experimental.
Its baseline preconditioner reports 16 iterations, including 14 accepted
updates, one rejected direction and one stationary check; candidate hands off
after ten accepted updates. Baseline formal completion needs no update, while
candidate adds seven accepted formal updates and 22 rejected trials. Thermal
callbacks fall from ten to seven, but total driver work rises. Final energy
and unrestricted temperature corrections are smaller in candidate and both
pass the original checks. Maximum temperature difference is 0.357 K. The
full-spectrum peak-scaled difference is 3.810e-7; the largest pointwise relative
difference, 0.003336, is at 900 Å where baseline flux is 2.313e-11 of its peak.

The 4000 K DA control retains the actual below-5000 K interface-transport
initialization, asymptotic conditioning and molecular/H-minus/H3-plus EOS.
It never reaches two eligible accepted updates, and all 27 callbacks, final
physical arrays and spectra are exact between variants. DB similarly never
triggers: both variants retain the native 16 → 18 depth boundary expansion,
36 callbacks and identical final gates, arrays and spectra. Each DB repeat
exactly replays its corresponding first variant, including policy metadata,
all nested seeds/R/J, controls and observed work. The initial timing gap occurs
partly before policy entry; large UTC/performance-clock gaps are retained.
Host execution conditions remain unresolved, and the repeated times provide
no evidence of an algorithmic DB slowdown.

DAB at 12000 K follows the ordinary atomic public selector. Its handoff at
accepted update eight coincides with baseline's existing preconditioner cap.
All 22 callbacks, physical endpoint arrays, spectrum and observed work agree.
Apart from the policy record, final metadata differs only in the first
segment's terminal reason. The shorter measured time therefore does not
demonstrate a solver work saving.

Two negative controls are also retained. DAB at 20000 K, log H/He = -2,
reaches both independent 300-second caps during improving thermal completion.
The first 141 callbacks match, but neither arm has a qualified endpoint or
complete timing ratio. A DAH 8000 K, 0.325 MG control uses the original
automatic convection suppression. Both variants fail the same coarse-grid
solver-completion, flux and energy checks; temperature stationarity remains
unmeasured. In both cases,
`require_convergence=True` rejects the exploratory saved models. The handoff
is inactive. This is an unqualified diagnostic-grid control, not evidence of
default-grid failure or an algorithm-induced regression.

Very cool dense-DB and molecular-DAB public routes use their separate isolated
physical-only recipes. Those recipes disable both the convective-gradient
preconditioner and built-in local energy enforcement, so this hook cannot
install even if its flag were forwarded. This is source/guard coverage, not
a new cold convergence qualification of those expensive recipes. DQ remains
excluded, and hot NLTE paths are unchanged.

The retained local bundle is `solver-handoff-other-modules-2026-10-07`, with
both plans, failed/capped outputs, checkpoints, identities and standalone
stdlib audits. Local energy error was decreasing at the 5000 K DA handoff
anchors and increasing in the successful DAH/DAZ/D6 anchors. Waiting for stalled
energy progress is a possible future criterion; it has not been implemented
or validated. These mixed results support keeping the option default-off.

## Additional named-object cold tests

Four more named objects were tested serially with the same frozen option and
current module physics. No historical atmosphere/checkpoint or warmup solve
was supplied. Every arm retains the original standard 120-iteration budget,
three angles, strict five-gate certificate and full public spectrum request.
Initial depth overrides preserve later native retry/refinement depths.

| Object and numerical controls | Baseline public seconds | Handoff public seconds | Observed driver requests / Jacobians | Result |
| --- | ---: | ---: | --- | --- |
| DAZ G29-38, native 40 depths / 300 continuum / default 20,000 lines | 402.492 | 346.022 | 39 / 8 → 44 / 7 | 14.0% lower time; standard-grid endpoint qualified |
| DAZ GALEX J1931+0117, 16 depths / 80 continuum / default 20,000 lines | 158.094 | 98.637 | 78 / 17 → 24 / 6 | 37.6% lower time; reduced-grid endpoint qualified |
| D6 J1235−3752, 16 depths / 160 continuum / 512 lines | 73.716 | 68.566 | 21 / 5 → 19 / 4 | 7.0% lower time; reduced-grid endpoint qualified |
| DZ GD 40, 16 depths / 80 continuum / 128 lines | 300-second cap | 300-second cap | 36 / 8 in both durable snapshots | No trigger; identical 25-callback prefix, no qualified endpoint |

All six completed endpoints pass their actual original five checks and save
full spectra. Config, cold initializer, first shared seed and existing initial
R/J are exact across each pair; source/data/native/runtime identities and
controls agree. The first 11 G29, seven GALEX and seven J1235 callbacks match
exactly before their respective handoffs. These are single serial pairs,
not statistical timing estimates. Driver counts omit outer adaptive,
thermal/certificate/material and initializer-internal work; public timers
include those phases, synthesis/save and lightweight observer I/O.

G29 illustrates why total completion matters. Its preconditioner uses 24
accepted updates in baseline and 11 in candidate; thermal callbacks fall
from 14 to seven. Candidate adds eight accepted formal updates and 15 rejected
trial evaluations before the original rejected-step handoff. Observed driver
requests therefore increase even though one Jacobian and substantial thermal
work are avoided. Final mass and pressure arrays are exact, temperature differs
by at most 0.744 K, and recomputed Rosseland labels differ. The full-spectrum
maximum pointwise relative difference is 8.883e-4; the peak-scaled difference
is 1.248e-5. The original saved-grid checks do not independently establish
resolution or spectral accuracy.

GALEX's preconditioner reports 58 observations (57 accepted updates plus a
stationary check), compared with seven accepted candidate updates. Thermal
callbacks fall from 15 to five. Both local-energy segments report two
iterations. The spectra agree to 1.813e-11 maximum pointwise relative
difference. J1235 hands off after seven accepted updates instead of baseline's
eight-step preconditioner budget; both formal segments report seven
observations, including six accepted updates and a stationary check. Its
maximum pointwise spectral difference is 1.576e-5. Neither reduced-grid pair
qualifies the corresponding full default-resolution calculation.

GD 40's hook installs but does not trigger: its actual all-depth flux error
at the existing eight-step cap is 0.003840, above the original 0.003 tolerance,
despite a smaller provisional residual. Both capped trajectories record eight
preconditioner and 17 formal callbacks, with identical physical arrays,
diagnostics, metadata apart from the policy record and observed driver work.
Their last local-energy error is about 0.593. No full endpoint, complete
timing ratio or complete no-effect proof is inferred from these prefixes.

The first G29 and GALEX pairs used a 128-line cap and both variants failed
public synthesis with `source_function must be non-negative`, after passing
the structure checks. Those four failed arms remain retained and unqualified.
G29's follow-up restores every standard numerical control. GALEX retains its
original 16/80 grid and removes only the line cap; both reruns synthesize
successfully. Static inspection identifies a consistency hazard when capped
metal absorption omits Ca II resonance opacity that is selected separately,
uncapped, for replacement emissivity. The exact offending lines/coordinates
were not measured, so this is a source-based explanation rather than a
verified root cause. No physics prescription was disabled or changed.

The local `solver-handoff-more-objects-2026-10-08` bundle retains all 12 arms,
both plans and supervisors, six frozen stdlib audits, full saved outputs and
an aggregate report. The numerical implementation, unit tests and native
binary are unchanged from the previously validated bundle. The option remains
default-off; the cool-DA work regression and outstanding DZ/full-D6
qualification still apply. DQ and hot NLTE algorithms remain unchanged.

## Default enablement with faster thermal conditioning

The handoff is now combined with a change to `thermal_condition`, the
pseudo-time phase between formal-flux completion and the local-energy Newton
solve. Its first sweep keeps the previous cautious control (0.04 ln T bound,
doubling the pseudo-time step only after a temporal defect below 0.0625). If
that sweep lowers the largest local energy defect, later sweeps use fast
control: every accepted step doubles the pseudo-time step, the ln T bound is
the shared solver's 0.12 maximum trust radius, and the handoff to the Newton
solve waits for a quarter of the local tolerance. The first fast sweep that
does not lower the defect discards the fast sweeps and resumes from the state
and pseudo-time step after the first sweep with the original cautious control
and tolerance, so a non-monotone case follows the previous trajectory exactly.
Acceptance (temporal defect at most 0.25), admissibility checks and every
final certificate are unchanged.

Simpler step rules failed and were rejected: always-fast steps (DB 10000 at
production depth then exceeds its 90-iteration canary guard), reverting to
cautious control after a rise (same failure), and handing off to Newton at
the first rise (DAZ GALEX J1931 then fails certification).

The two changes act on different phases and reinforce each other: stopping
the preconditioner before it drifts away from local balance leaves the thermal
phase a monotone problem that fast steps can finish. The handoff alone also
reduces thermal sweeps, which the earlier driver-request counts above do not
include.

Validation, all serial on one machine from true cold starts with public
`run_model` defaults, comparing against the parent commit:

| Case | Structure-evaluation work |
| --- | ---: |
| DAZ GALEX J1931 | -56% |
| DA 3000 | -46% |
| DA 20000 | -41% |
| DAZ G29-38 | -25% |
| DB 10000 | -23% |
| DA 4000 | -22% |
| DAZ G149-28 | -17% |
| DA 12000 | -13% |
| DA 5000, DZ J0738, DZ PG1225, DAB 20000 | -5% to -7% |
| DB 22000, DAH J1007+1237, DAH J1254+5612 | unchanged |

Work counts every structure evaluation, including thermal sweeps, priced at
the baseline's measured per-evaluation cost for that case and phase. All 15
models verify; none needs more work than its baseline. Spectra agree with
the baseline to better than 2e-5 of the peak flux. Two weakly constrained
outermost layers differ (Rosseland depth below 1e-7): the top node of DA 4000
and the top four nodes of DB 10000, where the baseline itself already shows a
pre-existing alternating temperature pattern. Neither affects the spectrum or
any check. The quick-quality D6 preset fails certification identically with
and without this change.

`tools/validate.py cold` passes all 16 requested cases (DB 10000 and 22000,
DB 22000 standard, DA 3000/4000/5000/20000, DA G76-48, DAB 20000, DAZ
G149-28/G29-38/GALEX J1931, DZ PG1225, DAH J1007+1237/J1254+5612 and DQ J1235),
and `tools/validate.py fast` passes. The full 48-depth/25,000-line J1637 cold
start was not rerun locally with this combination; it is part of the CI
full-validation cold tier.

## Reproduction and remaining qualification

Build the native extension and make the normal model data available first.
The benchmark does not download data. Run serially in an otherwise idle
checkout, with fresh output paths and an external wall cap:

```sh
export OPENWD_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=src
python benchmarks/benchmark_preconditioner_flux_handoff.py \
  --case daz --depths 8 --continuum 80 --metal-lines 16 \
  --preconditioner-steps 8 --steps 24 --samples 2 \
  --output /tmp/daz-handoff.json
```

Use `--case db` for the no-work-savings control. For the checkpoint-derived D6
comparison add `--case d6 --depths 16 --metal-lines 128 --checkpoint PATH`.
The retained pilot checkpoint SHA256 was
`79abb15433ba75f80ce357a013a3bed2a5231bafee28231e8e83c35a7fb7964d`.
Without a checkpoint, D6 uses its declared-physics grey initializer, which is
a different experiment and may not trigger the policy.

`run_bounded_solver_screen.py` supplies serial process timeouts. Inspect
`qualified_reduced_structure`, all physical gates, exact seed/R/J controls,
source/data/native identities and total work before interpreting timings.
JSON and sibling NPZ outputs retain the measurements and spectra. Command
completion alone is not qualification.

The following list predates default enablement; the section above records
which items were addressed. Before the earlier opt-in decision it read: obtain
qualified DZ endpoints and test wider ranges of
DA, DAH, DB and DAB temperatures, gravities and native grids, and improve or
bound the extra completion work exposed by the 5000 K DA case. Resolve the
capped DAB and failed diagnostic DAH controls without relaxing their gates.
Run a separately budgeted paired
J1637 cold test at unchanged 48-depth/25,000-line settings. If the capped
DAZ fixture is repeated, give each arm an independent equal cap. Count all
recovery and completion work, retain failed/capped attempts, require all
physical gates, and assess depth/wavelength refinement independently of the
A/B comparison. The standard DAZ cold pair above satisfies that case's
original production checks; it does not replace these wider qualifications.
DQ-specific work remains deferred.
