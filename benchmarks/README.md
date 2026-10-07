# Contained solver benchmarks

These benchmarks measure a single Jacobian at a fixed initial state. They do
not solve a cold atmosphere or certify model convergence. Both variants use
the current checkout: `baseline` disables the relevant reuse option and
`candidate` enables it. Build the native extension and install the bundled
model data normally before running. The benchmarks do not download data.

Pin numerical threads and run commands serially on an otherwise idle machine:

```sh
export OPENWD_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONPATH=src
python benchmarks/benchmark_lte_probe_reuse.py --case daz --samples 5 > /tmp/daz-reuse.json
python benchmarks/benchmark_hot_cache_reuse.py --variant baseline --depths 8 --output /tmp/do-baseline.json
python benchmarks/benchmark_hot_cache_reuse.py --variant candidate --depths 8 --output /tmp/do-candidate.json
```

The LTE script supports DA, DB, DAZ, DZ and D6. Its default fixture has eight
depths, 80 continuum points and at most 16 structure metal lines. It excludes
fixture construction and warmup, alternates the variant order, reports
material-call counts and checks residual/Jacobian equality.

The hot script supports eight or sixteen depths. Add `--mixed` for DAO;
otherwise it uses DO. Each fresh process performs an untimed warmup and one
measured evaluation on a fresh equations object. Output paths must be fresh.
Compare source, data, native-extension and thread identities, seed and output
arrays, and warmup equality before comparing timings. The JSON report and
sibling NPZ files retain this evidence. Single pairs demonstrate equivalence
and eliminated work; repeated alternating pairs are needed for timing claims.

For external wall limits, `run_bounded_solver_screen.py` accepts a JSON list:

```json
[
  {
    "name": "daz-reuse",
    "command": ["python", "benchmarks/benchmark_lte_probe_reuse.py", "--case", "daz", "--samples", "3"],
    "timeout_seconds": 120
  }
]
```

```sh
python benchmarks/run_bounded_solver_screen.py --plan /tmp/solver-plan.json --output /tmp/solver-screen
```

This POSIX runner pins threads, executes commands serially in the checkout,
captures stdout/stderr, terminates a timed-out process group and records source
hashes before and after execution. Use a fresh output directory to preserve
earlier evidence. A zero return code confirms command completion and unchanged
numerical source; inspect the benchmark's equivalence and provenance fields
separately.

## Separate public cold-start evidence

The fixed-state scripts above use same-checkout off/on controls. A separate
serial screen compared the pinned release with the frozen candidate through
public `run_model` calls, using standard settings, default wavelengths and
absent initial states. The observer saved callback diagnostics/checkpoints and
source/data/native identities, with numerical threads pinned to one.

DAZ G149-28 completed in 273.75 s release versus 247.86 s candidate, a 9.46%
reduction in one timing pair. All 33 recorded callbacks, final atmosphere,
physical diagnostics and spectrum match exactly; five required structure-grid
gates pass. Recorded radiative-equilibrium iterations are 34, so callback
count should not be interpreted as accepted Newton steps.

Both standard J1637 D6 attempts reached their original four-hour monotonic
caps during preconditioning. Their 11 common saved callback states and
diagnostics match exactly, but final convergence, spectrum and complete-model
performance remain unqualified. Raw timing includes recorded suspensions:
14.48 s candidate and 1778.73 s release for isolated diagnostic/native-test
windows. The reports retain raw clocks and separately adjust only pauses
preceding each matched milestone. Candidate laptop sleep is already excluded
from the model timer and is kept as separate UTC clock evidence.

These public-call observations supplement the contained component comparisons.
They do not provide repeated full-model timing statistics or an isolated
source-hunk attribution. See [qualification details](../docs/solver-cache-validation.md)
for the scope and frozen protocol identities.
