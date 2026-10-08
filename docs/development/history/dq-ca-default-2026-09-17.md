# DQ historical C–A default and transfer optimization

## Scope

The DQ default adds the missing C₂ C¹Πg–A¹Πu system without fitting stellar
parameters, abundances, molecular strengths, or an absolute flux scale. No
hydrogen or magnetism is added. The original Swan arrays and chemistry
partition convention are unchanged. `DQConfig(include_c2_ca=False)` is an
explicit diagnostic omission, not a fallback after a failed calculation.

The bundled 7.85 MB table is extracted from the historical Cooper/Nicholls
estimate and checksum-pinned in the constitutive-data manifest. Rebuilding
from the public, checksum-verified ExoMol states reproduces every packaged
cross section bitwise on the test platform. The builder, envelope integrator,
extraction tool, provenance, and attribution are included. This remains an
approximate rigid-rotor envelope, not a modern resolved C–A line list or a
validated C–A/helium collision profile. The measured electronic strength is
not adjusted using the stellar comparisons.

The ray and analytic-response kernels replace temporary-array expressions
with scalar compiled loops. Fourteen bitwise comparisons against the original
algebra and the existing transfer/finite-difference derivative tests protect
this change. The production full-response microbenchmark gives 3.09× speedup
(median 0.7234 s → 0.2343 s for 100 wavelengths and 45 depths, compilation
excluded); that is not an end-to-end timing claim. Reproduce with
`OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python tools/benchmark_dq_transfer.py`.

The accompanying carbon-line batching correction makes UV resonance support
a complete-atmosphere decision, invalidates local caches when that decision
changes, and batches the union of conservatively retained lines. Other
spectral classes keep their original default behavior.

## Independent infrared quadrature

The previous qualification mesh resolved the optical spectrum but sampled
the infrared only with its 4000-point broadband logarithmic mesh. This can
misestimate the total luminosity of a cool, carbon-rich DQ.

For the unchanged, reconverged J1311 candidate (5529 K, log g 8.178,
log C/He −5.27), both syntheses agree at shared wavelengths to 9e-14 relative.
The discrepancy is quadrature, not grid-dependent opacity or flux scaling:

| Sampling of the same atmosphere | Fbol / σTeff⁴ |
| --- | ---: |
| Original 17664-point structure grid | 1.0001990342 |
| Structure grid plus complete broadband mesh | 0.9994042950 |
| Old 154000-point independent grid | 0.9966943514 |
| Independent grid plus R≈24000 infrared | 0.9997737674 |

Infrared-only integrals from 6800–100000 Å, in units of σTeff⁴, are
0.51676144 at R≈6000, 0.51710601 at R≈12000, 0.51717297 at R≈24000, and
0.51715746 at R≈48000. The last refinement changes the integral by
0.00155% of the target luminosity, versus a 0.2% acceptance tolerance.
An initial suspicion about differing angular-node arguments was ruled out:
the refractive adapter consistently uses its own ray quadrature.

The released qualification mesh therefore retains the original optical
midpoints and adds 64521 logarithmic infrared nodes, giving 218520 unique
wavelengths. Structure sampling also retains all its line nodes and adds
the complete 4000-point continuum mesh. All five atmosphere gates and the
independent absolute bolometric tolerance remain unchanged. No spectrum is
renormalized. This sampling test does not establish independent depth-grid
convergence or precision spectroscopic accuracy.

## Qualification record

The final genuine J1235 cold start **passed** at the unchanged literature
parameters (9347 K, log g 8.041, log C/He −4.107), with C–A enabled. It took
2212.04 s (36.9 min), including independent synthesis, and required one
same-run domain extension from 40 to 41 layers. No external atmosphere,
saved spectrum, or fixed structure grid was supplied. The final five gates
are all measured and passed:

| Check | Value | Tolerance |
| --- | ---: | ---: |
| All-depth flux | 3.9937133e-4 | 0.002 |
| Local energy | 1.8935305e-3 | 0.002 |
| Temperature stationarity | 0 | 0.0002 |
| Source closure | 2.5641313e-15 | 1e-6 |
| Boundary screening | 1.0015842e-3 | 0.002 |

Its 218520-point independent spectrum is positive and finite throughout,
with `Fbol/(σTeff⁴) = 0.99993338154` (0.006662% error). The structure-grid
ratio is 0.99983167048. The public output reader successfully restored and
rechecked the saved result. All source and constitutive-data identities
match the qualified implementation. The logged transfer peak is 3.89 GiB,
below the unchanged 4 GiB guard. This verifies this cold canary, not universal
convergence, independent depth convergence, or full physical accuracy.

The worker and public-reader checks were run directly, not counted as a
passing invocation of the deselected slow pytest canary. Reproduce the cold
worker from the source checkout with a new output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python -m wd_spectra._dq.worker --teff 9347 --logg 8.041 \
  --log-carbon-to-helium -4.107 --seconds 5400 \
  --output results/dq-ca-release/j1235-cold-check
```

The first final-grid J1235 cold attempt reached thermal equilibrium on 40
layers, then correctly requested a deeper domain because its boundary escape
bound was 0.004956 > 0.002. Initializing the next layer exceeded the existing
4 GiB process-memory ceiling (4.59 GiB). That failed run is retained, not
promoted to a successful model. The EOS-corner observer was retaining the
previous domain's proposal system and its full radiation anchor. Domain
completion now clears that observer and collects unreachable solver cycles
before the next domain is initialized. A weak-reference regression verifies
that both the model system and its anchor are released. This changes object
lifetimes only, not physical equations, convergence tolerances, or the memory
ceiling. The successful fresh cold run above includes this repair. Its
40-layer temperature, pressure, mass, and optical-depth arrays are bitwise
identical to the failed run's equilibrium arrays.

A bounded warm diagnostic initialized from that failed run's retained
40-layer equilibrium crosses to 41 layers and passes all five atmosphere
gates in 412.69 s. Its peak memory is 3.08 GiB, below the unchanged 4 GiB
ceiling. This isolates the domain-transition repair; it is explicitly not
a cold-start certificate. A separate 218520-point synthesis of its 41-layer
state gives `Fbol/(σTeff⁴) = 0.99982648605`, within the unchanged 0.2%
bolometric tolerance. It does not replace the fresh cold run.

- Final regular suite after the memory-lifetime repair: **910 passed,
  5 optional-data skips, 8 slow tests deselected**, 162.12 s. No regression
  tolerance or fixture was loosened.
- The built wheel installed into an isolated temporary directory outside the
  checkout, found all five checksum-pinned data files without research paths,
  enabled C–A by default, and passed 46 selected DQ tests (one cold canary
  deselected). The old fixed-state spectrum is explicitly tested with C–A
  omitted, preserving its original strict tolerance as a transfer regression.
- J1311 warm reconvergence and synthesis (before the object-lifetime-only
  repair) passed in 441.39 s with
  unchanged source identities: all-depth flux 7.99e-6, local energy 2.51e-4,
  temperature stationarity zero, source closure 2.61e-15, and boundary
  screening 4.97e-7. Its 218520-point spectrum has
  `Fbol/(σTeff⁴) = 1.00037836472` (0.03784% error). This is explicitly warm
  numerical qualification, not cold-start evidence.
- J1311 optical shape RMS is 0.107272 versus the previous baseline 0.172776
  (37.9% lower); absolute optical RMS is 2.71965 versus 2.95936 (8.1% lower).
  The red model/observed ratio worsens to 1.2071. Shape diagnostics use the
  same red normalization as the earlier comparison; absolute diagnostics use
  the literature distance/radius without fitting. Extracted figure curves
  supply descriptive RMS, not a statistical chi-square.
- J1225 (6294 K, log g 7.924, log C/He −5.33) also passed warm
  reconvergence and the full independent grid in 573.20 s, with unchanged
  source identities. Its bolometric ratio is 1.00001743692, all-depth flux
  residual 5.62e-7, local energy 3.40e-5, source closure 2.39e-15, zero
  temperature correction, and boundary screening 6.24e-6. Optical shape RMS
  is 0.076378; absolute RMS is 0.71913 versus the original 0.71856, effectively
  unchanged. The red model/observed ratio is 1.00341. This remains warm
  qualification, not a second cold canary or a universal observational gain.
- A separate J1235 fixed-state UV screen compares the broadband mesh with
  R≈24000: the UV integral changes by 0.0114% of target luminosity. This is
  a sampling diagnostic, not an equilibrated spectrum or cold qualification.

The initial J1311 cold attempt using the old infrared-underresolved
qualification grid was stopped before completion once that sampling issue
was identified. Its retained report shows the interrupted compiled call as
a failure, not a certified result. A new warm solve with full-continuum
structure sampling converged; as expected, its old independent grid still
failed (ratio 0.99729856). Neither failure is hidden or treated as success.

Artifacts are retained under `results/dq-ca-release/`. These are development
outputs, not runtime inputs. The previous observational comparisons found a
substantial but partial improvement for J1311, little absolute improvement
for J1225, and no solution of the DQpec discrepancies; this is not a claim
that adding C–A fixes every observed band or continuum region.
