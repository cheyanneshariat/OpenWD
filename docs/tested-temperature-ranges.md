# Tested cold starts and limitations — September 2026

[Documentation home](README.md) · [Limitations overview](limitations.md)

These are **tested points**, not a guarantee throughout an interval.
Unless noted, log g = 8 and quality is production. Numerical convergence
certifies the declared equations on the structure grid, not complete physics
or depth-grid independence.
The separately synthesized spectrum also has known transfer-consistency
limitations. Historical linear-synthesis audits gave 0.98613 of stellar flux
for DA 3000 and 0.99314 for DB 10000. DA now defaults to monotone cubic synthesis;
the checked 12000-K standard cold model integrates to 1.00016 without scaling.
Other established synthesis defaults are unchanged. The new DQ module has its
own refractive transfer and mandatory independent spectrum check; it does not
enable the unfinished matched-transfer experiment for DA/DB. See the [DA guide](models/DA.md#spectrum-synthesis)
and the
[checkpoint's known spectrum limits](development/history/cold-start-numerics-2026-09-07.md#known-spectrum-consistency-limits-unfinished-changes-excluded).

Public generation starts from scratch. No saved atmosphere or neighboring
stellar model is required. Provisional thermal conditioning and lower-boundary
mesh adaptation occur inside the same calculation. Historical continuation
experiments remain in the research record but are excluded from this table.

| Composition / public workflow | Cold-start tested temperatures | Evidence |
| --- | --- | --- |
| DA, `compute_da` or `run_model` | 3000, 4000, 5000, 12000, 20000 K | All-depth flux, cell-local energy, measured unrestricted correction, scattering closure and bottom screening pass. The 12000-K default comparison also tests standard 40-layer resolution. |
| Dense pure-He DB, `run_model` | 5000, 8000 K | Fresh 80-layer initialization, static convergence and independent wavelength/angular/source audits. Experimental dense physics. |
| Established pure-He DB, `compute_db` or `run_model` | 10000, 22000 K | Strict structure-grid checks pass; 22000 K also tested at standard 40-layer resolution. The 10000 K calculation extends its own lower boundary from 80 to 84 nodes. |
| Molecular DAB/DBA, `run_model`, log10(N_H/N_He) = -2 | 7500, 8000, 9000, 10000 K | Fresh initialization with the final molecular/line physics. Strict static checks and independent fixed-state wavelength/angular audits. |
| Atomic DAB, `compute_dab` or automatically selected at 20000 K | 20000 K | Strict cold-start convergence and reviewed corrected-spectrum checks. The undoubled critical field intentionally changes high-series features relative to the old paper control. |
| DZ, PG 1225 composition, `compute_dz` | 10800 K | Strict production cold-start convergence. This is not exact reproduction of the 40-node paper spectrum. |
| DAZ, `compute_daz` or `run_model`, standard 40-layer resolution | G149-28: 8600 K, log g = 8.10; G29-38: 11820 K, log g = 8.40; GALEX J1931+0117: 20890 K, log g = 7.90 | Fresh public cold starts with each object's metal composition pass all five structure-grid certificate gates. These are individual points, not a temperature/abundance grid. |
| Refractive DQ, `compute_dq` or `run_model`, standard | J1225: 6294 K, log g = 7.924, log(C/He) = -5.33 | Stride-four structure default: true-cold public call passed all five atmosphere gates, independent 218520-point spectrum qualification and output reading. Prior dense-grid J1235 evidence is historical. See [qualification details](#dq-release-qualification). |
| Restricted NLTE DO, standard | 50000, 60000, 70000 K; log g=8 | Fresh public solves, 40 layers/He II 32/3 angles, independently certified. [Evidence and source revisions](development/history/do-dao-release-2026-09-20.md). |
| Restricted NLTE DAO, standard | 60000 K, log g=8, log H/He=2; GD153 40204 K, log g=7.82, log H/He=6 | Fresh public solves, 40 layers/He II 32/H8/3 angles. Same independent gates. |
| Restricted NLTE DAO, production | GD153 at the same fixed parameters | Fresh 80-layer/He II 8/4-angle solves with H8 and H20 separately. Observational profile discrepancies remain. |
| PG 1159, `compute_pg1159` or `run_model`, standard, `refine_upper_atmosphere=True` | PG 1707+427: 85000 K, log g = 7.5; PG 1424+535: 110000 K, log g = 7.0; PG 1159-035: 140000 K, log g = 7.0 with `include_radiative_acceleration=True` | Fresh cold starts with the published mass fractions, all `spectrum-qualified`: all-depth flux 0.0086, 0.0078 and 0.0027 (limit 0.01), local energy 5e-4, 3e-4 and 1e-4 (limit 3e-3), in 61, 71 and 80 minutes single-threaded (September 26). Known observational differences are listed in the [limitations](limitations.md#pg-1159). |

The revised solver repairs local energy errors that previously survived a
small interface-flux residual. It does not inherit historical success flags,
renormalize spectra, or impose convective flux as a remainder.
See [numerical repair](development/history/cold-start-numerics-2026-09-07.md) and
[certificate definitions](development/history/reliability-2026-09-07.md).

## Reproduction from scratch

For ordinary DA and warm DB/DAB presets:

```python
from wd_spectra import DAConfig, compute_da
result = compute_da(DAConfig(effective_temperature=3000, logg=8,
                            quality="production"))
assert result.atmosphere.metadata["equilibrium_certificate"]["verified"]
```

For automatic material selection, including dense helium and molecular mixtures:

```python
from wd_spectra import DBConfig, DABConfig, run_model

db = run_model(DBConfig(effective_temperature=5000, quality="production"),
               "results/db-5000-fresh", require_convergence=True)
dab = run_model(DABConfig(effective_temperature=7500, quality="production"),
                "results/dab-7500-fresh", require_convergence=True)
```

Use a new output directory for each calculation. Cool workers require a source
checkout with the normal installation and, for DAB, the
[declared molecular data](../research/cool_models/README.md).
Missing data or invalid material domains fail explicitly; there is no alternate
physics retry. `run_model` rejects checkpoint inputs.

`compute_*` functions remain explicit physics presets. In particular,
`compute_db` at 5000 K does not select dense helium, and merely enabling
`DABConfig(include_molecules=True)` does not select the qualified conservative
molecular transport workflow. Use `run_model` for that automatic selection.

### DQ release qualification

The September 18 **stride-four structure default** passed a fresh public
`compute_dq` qualification at J1225 (6294 K, log g 7.924, log C/He −5.33),
including automatic 40→41→42-node extensions. All five atmosphere gates,
the independent 218,520-point spectrum, and public output reading passed.
Fbol/(σTeff⁴) = **0.999176494939219**, without normalization. The final
atmosphere and full spectrum match the accepted pre-promotion experiment
bit-for-bit. Worker time was 22.91 minutes with tests/builds running alongside
part of the solve; this is not an isolated speed benchmark.

The accepted accuracy tradeoff is not a uniform 1% bound: J1225 differed
from the dense-grid reference by 1.49% native optical and 0.29% at 3 Å FWHM.
J1311's cold test was stopped unconverged; a separate warm diagnostic differed
by 8.13% native and 1.29% at 3 Å. The full final spectrum and all convergence
tolerances remain unchanged. See the
[stride-four release record](development/history/dq-stride4-default-2026-09-18.md).

The September 17 **dense-grid 2024 C–A/completed-Swan default** was independently
requalified at J1235 through the public `compute_dq` API, starting from gray
initialization without an input atmosphere, grid or spectrum. Its automatic
40-to-41-node lower-domain extension was followed by a successful re-solve.
All five measured gates passed: all-depth flux `3.90371258e-4`, local energy
`1.88632427e-3`, temperature stationarity `0`, source closure `2.37774197e-15`
and bottom-boundary response `9.81462143e-4`. The independent **218520-point**
spectrum is finite and positive, with
`F_bol/(sigma Teff^4) = 0.9999375372794359`, without normalization.
The worker took **58.96 minutes** within a 4200-second resource budget; the
public caller then read the result successfully and exited normally. No
physical tolerance or 4-GiB memory guard was relaxed, and production sources
and data stayed unchanged. See the
[preceding default record](development/history/dq-completed-default-2026-09-17.md)
for reproduction, memory fixes, tests and separate observational evidence.

The following earlier release measurements are retained as historical evidence,
not attributed to the new default. The earlier September 17 packaged J1235
worker started without an input atmosphere,
structure grid or spectrum, at the fixed parameters above. Its 41-depth
atmosphere passed all five measured gates: all-depth flux error `1.80490e-4`,
local energy error `1.58077e-3`, temperature stationarity `0`, source closure
`2.41258e-15` and bottom-boundary response `1.00809e-3`. The independent
154000-point spectrum had finite positive flux and
`F_bol/(sigma Teff^4) = 1.0002868900589381`, within the absolute 0.002 limit
without normalization. The worker reported about 100.5 minutes, excluding an
additional laptop-sleep interval; runtime is machine- and parameter-dependent.

The original public caller failed only while reconstructing the saved helium
state after the numerical worker finished. The corrected reader was verified
against that completed output, with exact recovery of all bulk state arrays
and flux. The calculation was not restarted, nor was that initial caller
failure reclassified as an uninterrupted API pass. The
[release record](development/history/dq-release-2026-09-17.md) preserves the
source/checksum audit and the distinction between worker and reader validation.

Use the [DQ quick start](getting-started.md#dq-heliumcarbon-atmospheres) with
`quality="standard"`. DQ's input tables are bundled and both public entry
points require a true cold start and mandatory atmosphere/spectrum qualification.
The saved-state spectral pytest is a separate regression, not cold evidence.
Earlier research results are not a qualification of every object in this
packaged release: historical J1803/J1311 cold starts remained unfinished and another J0916 cold check
was waived. No independent DQ depth-convergence or observational-accuracy
claim follows from this one release point.

## Numerical evidence

The table below retains the September 7–8 checkpoint measurements, not newly
measured residuals for every subsequent release. The
[microphysics audit](development/history/microphysics-audit-2026-09-10.md)
records the corrected cold starts and reviewed regression promotion.
Errors below are fractions, not percentages. Local energy uses each cell's
exchange scale, not total stellar luminosity. Fresh molecular models use
80 layers and the same 40-sweep provisional thermal budget; a subsequent
full static solve must independently pass every physical gate.

| Fresh case | All-depth flux error | Cell-local energy error | Independent audit |
| --- | ---: | ---: | --- |
| DA 3000 | 3.82e-4 | 3.24e-4 | Structure grid only |
| DA 4000 | 4.28e-6 | 4.18e-6 | Structure grid only |
| DA 5000 | 3.53e-6 | 1.17e-5 | Structure grid only |
| DA 20000 | 1.43e-7 | 4.29e-4 | Structure grid only |
| DB 10000 | 5.64e-4 | 4.31e-5 | Structure grid only; bottom escape bound 8.27e-26 |
| DB 22000, standard | 4.63e-10 | 1.69e-3 | Structure grid only |
| Atomic DAB 20000 | 8.53e-8 | 1.59e-3 | Structure grid only |
| Dense DB 5000 | 1.26e-4 | 1.24e-5 | Independent local error 0.00126 |
| Dense DB 8000 | 6.76e-6 | 1.04e-5 | Independent local error 0.000955 |
| Molecular DAB 7500 | 3.74e-10 | 2.90e-6 | Independent flux 8.01e-5, local 7.69e-5 |
| Molecular DAB 8000 | 1.91e-10 | 1.75e-6 | Independent flux 1.53e-4, local 6.39e-5 |
| Molecular DAB 9000 | 1.16e-10 | 2.53e-7 | Independent flux 3.16e-4, local 5.98e-5 |
| Molecular DAB 10000 | 2.23e-10 | 4.85e-8 | Independent flux 4.63e-4, local 8.66e-5 |
| DAZ G149-28, standard | 4.10e-6 | 2.54e-6 | Structure grid only; 32 iterations |
| DAZ G29-38, standard | 6.12e-5 | 4.05e-6 | Structure grid only; 58 iterations |
| DAZ GALEX J1931+0117, standard | 7.92e-6 | 8.88e-4 | Structure grid only; 97 iterations |

Independent quadrature audits hold the newly calculated atmosphere fixed.
They are checks of that result, not re-relaxations or inputs needed for a cold
start. They do not replace stationarity or depth-resolution tests.

The three DAZ cases were rerun from scratch on September 8 with the established
atmosphere and formal-integral synthesis methods. Temperatures, gas pressures,
column masses and saved spectra exactly reproduce their pre-experiment cold
results. The configurations and metal abundances are in
[`research/validate_daz_cold_start.py`](../research/validate_daz_cold_start.py).
The separate fixed-atmosphere DAZ paper comparisons also pass; these do not
imply that a new cold atmosphere is identical to a paper checkpoint or that
its final-spectrum integral is certified. See the [DAZ guide](models/DAZ.md).

The 8000 K DAB depth study compared 80, 159 and 317 layers: integrated absolute
spectral differences decreased from 0.688% to 0.172% of stellar flux, approximately
second order. The new 80-layer 7500 K cold spectrum differs by 0.985% in integrated
absolute flux from the historical 166-layer calculation. These are different
depth grids; neither comparison establishes universal depth independence.

## Limits and preservation of established results

- No DA below 3000 K or pure-He DB below 5000 K is qualified here.
- Molecular DAB 7250 K remains unqualified under the measured correction gate;
  5000 K experiments remain far from equilibrium. There is no claim of a
  reliable 5000–10000 K DAB grid.
- The old 7750 K continuation is not a cold-start qualification.
- No abundance or gravity grid has been demonstrated. The cool DAB points
  mean the homogeneous 1% hydrogen-by-number mixture above.
- Dense pure-He thermodynamics combine REOS with approximate HNC chemical
  potentials and trace-ion chemistry. Local table and trace-ion limits are
  enforced. Refraction and collective He-minus corrections remain absent in
  this pure-He DB workflow; the separate DQ adapter includes both.
  The dense-neutral closure is not inserted into warm ionized helium or
  hydrogen/helium mixtures.
- Molecular mixtures include H2, H2+, H-, H3+, H/He ionization, H2-He/H2-H2 CIA
  and neutral Ly-alpha wings. A consistent dense-mixture free energy,
  nonideal dissociation, mixed HeH+/He2+ chemistry and pressure-distorted CIA
  remain missing.
- The DA seed retains its previously tested below-5000 K initialization policy.
  The new local-energy completion and thermal step selection introduce no
  additional Teff/composition switch.
- DQ is restricted to nonmagnetic, hydrogen-free helium with trace carbon.
  Dense-mixture EOS, molecular collision profiles and the grey refractive ML2
  bridge remain approximate; DQp distortion and hot carbon-dominated DQs are
  not supplied. Numerical qualification is not a precision-fitting validation.

Separate reviewed corrected controls now protect UV, optical lines and IR for
DA/DB, the paper DAB 9000/20000 models, PG 1225-079 and SDSS J0738+1835.
Historical atmosphere/flux files remain unchanged, with hashes checked by tests.
Those checks alone are not cold-start or equilibrium evidence. The legacy
atomic 9000 K paper spectrum remains a spectral control; automatic generation
at that point selects molecular physics.
J0738's new production atmosphere passes its static checks, but its cold
spectrum fails the paper comparison; it is not in the qualified reproduction
table above. That discrepancy is not waived or explained away as a proven
pre-existing difference.

The repaired 10000 K DB changes optical and IR band flux by only 0.0176% and
0.0092%, but its 1150–3000 Angstrom flux rises by 0.271% and portions of the
faint far-UV tail change by about 10%. This is disclosed, not described as an
identical spectrum. The original boundary was not absorption-screened.

Historical investigations, including unsuccessful runs and continuation
experiments, remain in the [research archive](development/history/README.md).
They document chronology, not current public reproduction instructions.
