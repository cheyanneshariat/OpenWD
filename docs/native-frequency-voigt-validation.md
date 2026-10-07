# Native frequency-Voigt manifold validation

This change accelerates the released D6 manifold profile without restoring
the older pseudo-Voigt prescription. It evaluates the frequency-space Voigt
with the released Humlicek W4 approximation, which has about `1e-4` relative
accuracy as a Faddeeva approximation. Native/reference equivalence is a
separate, much tighter check against that same W4 formula.

The new native kernel uses `nu = c/lambda`, widths converted at the line
centre, and the released constant per-Angstrom conversion at that centre.
This retains distant-wing wavelength asymmetry. There is no additional
wavelength-dependent conversion factor. A fused finish interpolates the
same fine/coarse grids, clips negative fine-bin masses before interpolation,
adds the same core and outer-wing mass weights, and divides by the same
bound probability. Atomic manifolds, microfield sampling, dissolution,
profile support, bin deposition and SciPy FFT convolution remain unchanged.

Only `_compiled_manifold_profile` selects the new methods. The original
Python W4, frequency-Voigt and impact-profile functions remain unchanged.
The existing static ordinary LTE W4 evaluator is also unchanged; a wrapper
shares its formula with the new manifold kernels. DQ changes are deferred.

Rebuild the optional extension to enable `frequency_voigt_profile` and
`stark_frequency_profile_finish`. Existing native APIs remain available.
An older extension can retain native deposition and Python frequency-Voigt
evaluation/assembly; without the extension, the independent NumPy manifold
implementation remains available. Partial availability of the two new
methods is supported independently.

The native methods are selected only when the centre and both widths are
builtin `float`/`int` or NumPy `float64` scalars. Other scalar types retain
the released Python arithmetic. Native finishing additionally requires a
one-dimensional native-float64 wavelength array; other shapes/dtypes keep
the original Python assembly, including float32 offset arithmetic.

## Validation

The final-source focused/native screen passed **71 tests**, with no skips.
The earlier, broader screen before the compatibility dispatch guards passed
137 tests, with one skip for a missing external cached Koester control.
These screens overlap and their counts are not added. Coverage includes the W4 regions,
frequency-wing asymmetry, fine/coarse interpolation boundaries and clipping,
post-FFT assembly, real Mg manifolds, wavelength-window invariance, old and
partial extensions, buffer errors, overlap rejection and buffer lifetime.
The final screen also checks unsupported scalar precision and wavelength
shape/dtype fallbacks, including their actual native-method call counts.

A negative control substituting the obsolete pseudo-Voigt kernel for the
new frequency-Voigt kernel produced
the expected wing-asymmetry test failure (ratio 1 instead of 9); the control
driver passed. A source distribution included `csrc/humlicek_w4.h`, its
extracted sources were byte-identical, and a fresh extension built from that
archive passed both new native-API preflight checks.

Fixed-atmosphere opacity comparisons used the archived J1637 v5 atmosphere
and the released compiled-manifold function from commit
`d9640656dce01bc6fc788c612d7a153c0d51c756`. Within each process, both modes
used the same state, selected lines, wavelength grid, atomic data and loaded
native binary. In the reduced repeated screen, both paths were warmed and
measured pairs alternated execution order. The full-size screen used one
released material warmup to populate the common atomic/field/FFT caches,
then primed the new native methods on small buffers outside timing. Its one
measured pair ran released then native; the native kernels have no persistent
state. Times include identical result copies and exclude setup, warmup, comparison
checks, radiation checks and separate profiling passes. Numerical libraries
were configured for one thread on macOS arm64.

| Runtime | Retained workload | Pairs | Released time | Native time | Time reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| Python 3.12.14 / NumPy 2.5.3 / SciPy 1.18.1, final guarded source | 12 depths, 3,000 lines, 33,008 wavelengths | 3, medians | 5.1865 s | 4.6162 s | 11.00% |
| Same runtime and final source, shared warmup | 48 depths, 25,000 lines, 271,472 wavelengths | 1 | 220.1513 s | 139.7359 s | 36.53% |

The maximum pointwise relative absorption differences were `2.12e-13`
(reduced screen) and `4.37e-13` (full screen), using a denominator floor of
`1e-20` of the reference opacity peak. The full-screen maximum difference
relative to that peak was `7.37e-17`. Scattering was bitwise equal. The
maximum changes in depth-integrated radiative flux were `1.75e-10` and
`5.30e-11` of `sigma Teff^4`, respectively. Every measured pair passed the
saved numerical gate, and source/data, native-binary and input-atmosphere
hashes remained unchanged.
Roundoff equivalence is required; bitwise absorption equality is not claimed.

Earlier source, before the compatibility-only dispatch guards, gave
`5.2723 -> 4.6864 s` over three pairs in that modern environment and
`7.3056 -> 6.7516 s` over two pairs in Python 3.9.16 / NumPy 1.26.4 /
SciPy 1.11.1. These are retained context with their original source hashes;
the native kernels are unchanged. The Python 3.9 absorption difference was
`1.65e-13` pointwise and its depth-flux difference was `5.86e-11` of
`sigma Teff^4`, with bitwise equal scattering.

The separate modern-runtime profile confirms 1,962 calls to each new native
method, with the same 1,962 native deposition calls in both modes. The
remaining FFT and opacity work limits the application gain. Profile times
are excluded from the timing medians.

These measurements qualify a contained retained-state opacity improvement.
They do not measure cold atmosphere construction, changed solver iteration
counts, full-model runtime, or convergence of a new atmosphere. The
full-size result is a single matched sample, without an uncertainty estimate.
Its controller completed in 618.074 seconds, including setup, warmup and
checks; the separately journaled cold-worker pause was 618.075609 seconds.
The first full-size attempt reached its 900-second cap before the measured
native call completed. Its retained partial report establishes no matched
gain and is not included in the table. The
September 30 native/pseudo-Voigt timings are a different historical
comparison and are not the baseline used here.

The evidence is retained in the workspace under
`results/d6-frequency-voigt-native-2026-10-06/`, including the build/pause
plans, `pilot-8depth-500lines/`, `screen-12depth-3000lines/` and
`screen-py39-12depth-3000lines/`, `final-screen-12depth-3000lines/`,
`full-shared-48depth-25000lines/` and `source-distribution/`, plus the
earlier capped `full-48depth-25000lines/` attempt.
Reports contain runtime, source/native
hashes, input identity, numerical gates and timing samples; NPZ files retain
the actual fixture, opacities and radiation comparisons.

## Reproduction

From the repository root, build the extension for the interpreter being
tested, then run the focused tests:

```bash
python setup.py build_ext --inplace --force
PYTHONPATH=src:tests python -m pytest -q \
  tests/test_stark_frequency_native.py tests/test_stark_native.py
```

The retained-state benchmark requires a separate immutable released
checkout and an archived atmosphere NPZ; it does not produce a cold model:

```bash
OPENWD_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
PYTHONPATH=src python benchmarks/benchmark_d6_voigt_opacity.py \
  --source-root . --reference-root /path/to/released-checkout \
  --atmosphere /path/to/archived/atmosphere.npz \
  --depths 12 --lines 3000 --continuum 250 --repeat 3 --radiation \
  --output /path/to/fresh-output
```

To reproduce the full retained-state comparison, use the same command with
`--depths 48 --lines 25000 --continuum 450 --repeat 1 --warmup shared` and
a fresh output directory. This still evaluates opacity and a fixed-state
radiation comparison; it does not run a cold model.
