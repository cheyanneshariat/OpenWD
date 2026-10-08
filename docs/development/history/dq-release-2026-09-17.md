# Refractive DQ release extraction — 2026-09-17

[History index](README.md) · [DQ usage and limitations](../../models/DQ.md)

## Scope

The release extracts the current-energy, corner-aware, UV-refined protocol
used for the successful J0752 and J1235 true-cold calculations. Numerical
kernels are relocated, not replaced by the older non-refractive public DQ
adapter. Research launchers, object-specific experiments, fitting scripts,
and old comparison/test programs are not runtime dependencies.

`wd_spectra.models.dq` owns the public configuration and isolated worker.
`wd_spectra._dq` owns its private refractive transfer, material responses,
convection coordinates and phase controller. Constitutive tables are bundled
under `wd_spectra/data/dq`, identical to the successful J1235 inputs.
No model atmosphere or spectrum is bundled as a runtime input.

Shared solver extensions remain opt-in. Default DA/DB/DAB/DZ numerical
policies are not switched to the DQ settings. Worker isolation also prevents
temporary DQ adapter hooks from leaking into other requests.

## Validation evidence

- Before extraction: 861 tests passed, five optional-data skips, seven slow
  canaries deselected. One pre-existing documentation-reachability failure
  concerned unindexed research notes.
- Packaged saved-state J1235 synthesis: 79 wavelengths agree with the original
  absolute fine-grid flux to maximum relative error `1.19349e-13`. This is a
  fixed-state regression, not a new cold atmosphere.
- Final normal pytest from the staged-tree export after the reader repair:
  869 passed, five optional-data skips, eight slow canaries deselected
  (207.60 seconds). The pre-repair export passed 868 tests.
  The export uses its own package files/data and freshly compiled C backend;
  read-only Git metadata is supplied for the validation-runner unit test.
  The release-contract file now includes 21 fast tests plus a separate cold canary.
- A wheel built successfully and was installed into an isolated target.
  From outside the repository, it verified every DQ data checksum and
  reproduced the same `1.19349e-13` fixed-state spectrum agreement without
  importing research modules. Wheel inspection found all package Python
  modules and all five DQ data files, with no research or results entries.
- The final wheel also includes the native C backend and third-party notices.
  The local Conda compiler targets the wrong architecture; using the native
  Apple compiler explicitly builds the backend without a global environment
  change. The installed native wheel passes the same spectrum/checksum test.
- The final repaired wheel was rebuilt and installed outside the checkout.
  Its 112 package Python files match the validated staged source hashes. The
  installed reader loads the actual completed cold output without changing
  any bulk state or flux; both the real save/reload regression and fixed-state
  absolute spectral regression pass. Wheel SHA-256:
  `2c92ac23073f15bed901292f27e896857f9e904a2d3155b0f2b411d5b3492a1e`.
- Existing cool-workflow component tests pass from an export containing only
  staged release files: 226 passed, eight optional external H2-data skips.
  The expanded public-contract/documentation/CI-policy checks pass 50 tests.
- A 60-second cold smoke test constructed the same 25471-point J1235
  structure grid (from the same 24311 original nodes), then stopped at its
  explicit budget. It did not produce a converged atmosphere or spectrum.
- The full packaged cold-start canary is separately marked `canary`; normal
  pytest never presents a saved-state fixture as cold-start evidence.
- All seven protected non-DQ cold canaries passed (pytest reported 4283.04
  seconds): DB10000 production, DB22000 production and standard, DA5000,
  DA20000, DA3000 and DA4000 production. These check certified cold atmospheres
  and absolute spectra against the existing approved controls.
- The packaged worker completed J1235 from a true cold start at fixed
  9347 K / logg 8.041 / log(C/He) -4.107. Its 41-depth atmosphere passed all
  five measured certificate gates. The independently synthesized spectrum
  contains the exact 154000-point grid, has finite positive flux everywhere,
  and has `Fbol/(sigma Teff^4) = 1.0002868900589381` (0.028689% error).
  Its six saved bulk atmosphere arrays and final-spectrum checksum are
  identical to the original successful research calculation. The worker
  reported 6032.73 monotonic seconds; wall time also included 26 minutes of
  laptop sleep. No input atmosphere, structure grid or prior spectrum was used.
- The original public call then failed while deserializing this completed
  result: the reader omitted the helium population object required for He/C
  EOS reclosure. The repair restores the helium host from saved temperature
  and pressure before reclosure; it changes no numerical solver, synthesis,
  physics, parameters or tolerance. The completed worker's source/data
  consistency checks passed before this repair. A subsequent source audit
  verified that only `_load_result` changed (all other package modules are
  byte-identical, and the remainder of the public module has identical AST).
- The repaired reader was tested against the actual completed output, with
  exact bulk-state and spectrum recovery and all original qualification gates
  rechecked. A new real EOS/save/reload unit regression fails on the old reader
  and passes with the repair. The original failed caller log is retained; this
  is a qualified true-cold worker followed by a verified reader repair, not a
  claim that the original end-to-end call was uninterrupted. No second cold
  calculation or atmosphere restart was needed.

### GitHub CI and documentation follow-up

The first [GitHub test run](https://github.com/kareemelbadry/OpenWD/actions/runs/35263175646)
passed four jobs, including all fixed-spectrum comparisons, but the Python
3.12 unit job failed seven DQ response tests under NumPy 2.5.3. The tests used
`np.trapz`, an API [removed in NumPy 2.4](https://numpy.org/doc/stable/release/2.4.0-notes.html#removal-of-deprecated-functions-and-arguments).
This was an API-compatibility exception, not a numerical-tolerance failure.
An audit also found the same legacy call in two DQ energy/boundary callbacks.

The follow-up replaces those four calls with the existing `_compat.trapezoid`
helper, which selects `np.trapz` on old NumPy and `np.trapezoid` on new NumPy.
On the original NumPy 1.26.4 environment the helper is the identical function
object; no equations, quadrature, physics, parameters, grids or tolerances
change. Two callback regressions also remove the old name explicitly so this
compatibility gap is caught even when testing with older NumPy.

A fresh NumPy 2.4.6 environment reproduced all seven original failures and
both newly covered callback failures before the patch. After the patch,
54 focused documentation/DQ tests passed, including the real saved-output
reader and fixed-state absolute spectrum regression. The full NumPy 1.26.4
suite passed 884 tests with five optional-data skips and eight cold canaries
deselected (203.57 seconds). The user documentation
and notebook now include DQ entry points and distinguish DQ's mandatory
refractive spectrum qualification from other workflows. No new cold result
is claimed for this API-name compatibility repair.

Earlier research qualification does not establish uniform-suite qualification
of this extracted release. In particular, J1803/J1311 qualification was
unfinished, and the user waived another cold confirmation for J0916. No such
run is relabeled as a completed cold qualification by packaging.

## Recoverable cleanup

482 untracked DQ research programs, superseded tests, historical notes and
generated Fortran module files were moved to the local sibling archive
`spectral_model/quarantine/dq-pre-release-2026-09-17`. Its `manifest.json`
records each original relative path, byte size and SHA-256. It is outside the
release repository and is not distributed. No old model results or checkpoints
were moved or deleted. Existing ignored `results/` remains local historical
evidence, never an installation dependency.

Useful equation-level tests were migrated to package imports. An explicitly
fixed-state J1235 spectrum fixture lives only under `tests/data/dq_regressions`,
not package data and not on the cold solver's input path.
