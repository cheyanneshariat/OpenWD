# Performance

[Documentation home](../README.md) · [Development](README.md)

The optional C extension accelerates formal transfer, metal lines, helium
profiles, and neutral-broadened hydrogen profiles without changing their
numerical settings. Compiled hydrogen profiles use up to eight worker threads
by default because atmospheric depths are independent.

Set `OPENWD_NUM_THREADS` to a positive integer to control that hydrogen-profile
work. When running independent models in parallel processes, use
`OPENWD_NUM_THREADS=1` and consider `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1` to avoid competing thread pools. Each model needs its
own output directory.

The D6 manifold path also compiles component deposition, impact-profile
evaluation, and interpolation of the fine and coarse profile grids. Atomic
manifolds, microfield sampling, dissolution, profile support, and FFT
convolution retain the Python algorithm and settings. A bounded cache reuses
microfield distributions only when all physical inputs match exactly. The
NumPy profile implementation remains the fallback when the extension is
unavailable; rebuild the extension after updating its C sources.

The shared Hooper microfield fit also reuses a single decaying exponential
for both logistic branches, preserving their bitwise results and avoiding
overflow in the unused branch.

From the repository root, the bundled data-independent benchmark exercises
the dominant Balmer-opacity path:

```bash
python benchmarks/benchmark_hot_paths.py
python benchmarks/benchmark_hot_paths.py --full
OPENWD_NUM_THREADS=1 python benchmarks/benchmark_hot_paths.py --full
```

With the bundled Mg atomic data and the compiled extension installed, compare
the native manifold kernels against the independent NumPy implementation:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python benchmarks/benchmark_stark_profiles.py --repeat 7
```

This benchmark warms both paths, alternates their execution order, and reports
the median speedup and the largest profile difference relative to its peak.
Its speedup is not the speedup of a full atmosphere calculation. Changes to
D6 performance also need paired broad-spectrum and full structure-opacity
timings, followed by cold-start convergence checks at unchanged tolerances.

Paired D6 measurements on macOS arm64 (Python 3.9, NumPy 1.26.4,
SciPy 1.11.1, one numerical-library thread), against commit `5dacf2b`, gave:

| Workload | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| J1109 optical spectrum, 100,001 wavelengths | 119.77 s | 60.28 s | 1.99x |
| J1637 structure opacity, 271,472 wavelengths | 295.88 s | 151.72 s | 1.95x |
| J1637 reduced-grid cold atmosphere | 263.03 s | 166.94 s | 1.58x |

The optical timings are medians of two paired trials; the other rows are
single paired trials. Both opacity cases use 48 depths and 25,000 lines.
The reduced-grid cold control uses 24 depths and 2,000 lines: both versions
converge in 20 iterations and pass all five equilibrium checks, with a maximum
fractional temperature difference of `6.08e-11`. The broad optical spectrum
and structure opacity differ by at most `3.3e-10` relatively. These comparisons
retain the same physics, line selection, grids, and convergence tolerances.
A production-grid cold start has not been retimed; the roughly twofold
opacity gain is not a measured twofold reduction of a full cold solve.

The Balmer benchmark reports a checksum with its timing. Timings are
diagnostic: runtime depends on machine and model. Lowering
numerical resolution or relaxing convergence tolerances is not an equivalent
speedup; numerical and spectral checks must still pass.
