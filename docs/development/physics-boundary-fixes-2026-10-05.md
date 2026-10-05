# Four reproduced boundary and material-state defects

This note accompanies candidate changes to commit
`a1bf936422db8005b3c0e87dc38150d6a4f32549`. The candidate is not ready for
release: three fixed-atmosphere spectrum regressions fail, and full validation
from newly calculated atmospheres has not been run.

The patch repairs low-density Stark scaling, two native bounds errors, and
stored helium mean charge. Each figure below states its test conditions and
limits. These are deterministic calculations, with no uncertainty intervals.
The legends and before/after labels apply to their own panels.

## Figure 1: hydrogen density boundary

![Hydrogen density boundary](../assets/physics-boundary-fixes-2026-10-05/01_density_edge.png)

Local Hα profile at T = 5000 K and electron density = 10⁸ cm⁻³. The original code makes the core too narrow below the table density range. The repair keeps the full profile at the lower density limit. The thermal Gaussian is a reference. This test does not measure accuracy for an observed star. The fix is needed so the code does not make a thermally broadened line artificially narrow at low density.

## Figure 2: native Stark endpoint

![Native Stark endpoint](../assets/physics-boundary-fixes-2026-10-05/02_native_endpoint.png)

An eight-node, nonuniform table is queried at its last node. The original code reads index 8, outside the array. The repair uses index 7 and returns the expected value, 0.001. The shaded region contains valid indices. A separate AddressSanitizer test confirms the original invalid read. The bundled grids are uniform. The fix is needed to prevent a memory read outside the array, which can return a wrong value or crash the program.

## Figure 3: invalid native line center

![Invalid native line center](../assets/physics-boundary-fixes-2026-10-05/03_nan_line_center.png)

The line center is NaN (not a number). The original code reads index −1, before the buffer. The repair raises ValueError before interpolation. The shaded region shows the first valid indices of a 50-element buffer. A separate AddressSanitizer test confirms the original invalid read. The fix is needed so invalid input raises a clear error instead of causing an unsafe memory read.

## Figure 4: stored helium mean charge

![Stored helium mean charge](../assets/physics-boundary-fixes-2026-10-05/04_helium_charge.png)

Stored helium mean charge for a test mixture with He/C/O mass fractions of 0.33/0.50/0.17. The repair uses the new helium populations in the denominator. The maximum relative error falls from 56.4% to zero in this test. Electron and ion populations do not change. PG1159 later replaces this field during charge feedback. The fix is needed so the stored charge describes the helium actually present, rather than using the amount of helium from the previous state.

## Figure 5: spectrum changes from the density repair

![Fixed-atmosphere spectrum changes](../assets/physics-boundary-fixes-2026-10-05/07_fixed_spectrum_impact.png)

Flux changes from the density repair in the three failed frozen-spectrum checks. Atmospheric structures stay fixed. The panels use different vertical scales. Optical relative changes are below 5.2 × 10⁻⁹ in these controls; very faint far-UV pixels still fail the original thresholds. This test explains the failures but does not qualify the candidate for release. The density fix is needed to remove artificial line narrowing, even where its effect on the optical spectrum is very small.

## Validation status

Checks run on 2026-10-05 gave:

- Ordinary component suite: 1574 passed, 1 skipped, 44 deselected.
- Cool-component suite: 226 passed, 8 skipped.
- Final targeted recheck after documentation packaging: 97 passed, 1 skipped because a cached Koester validation spectrum is unavailable.
- New density regression: 19 of 24 cases failed before the repair; all 24 pass after it.
- Fixed-atmosphere spectra: 18 of 21 passed. The failed cases are `da-4000`, `da-5000`, and `dab-9000`.
- The reference spectra and numerical thresholds were not changed.

Independent fixed-structure calculations restored only the original density
scaling. All three then matched the immutable references within the existing
thresholds. This attributes the failures to the density correction; it does
not make the repaired default pass the regressions.

| Failed case | Pixels failing the threshold / total | Wavelength range (Å) | Maximum optical relative change |
|---|---:|---:|---:|
| DA 4000 K | 19 / 3000 | 1199–1215 | 2.02 × 10⁻⁹ |
| DA 5000 K | 4 / 3000 | 1215–1217 | 1.44 × 10⁻⁹ |
| DAB 9000 K | 7 / 3000 | 938.9–1216.4 | 5.19 × 10⁻⁹ |

Small changes to the optical spectra do not justify relaxing the failed
thresholds. The density policy and changed reference spectra need review.
If the policy is accepted, reviewed reference updates and full validation
remain required under the [development policy](README.md).

## Reproduce the regression checks

Use an existing environment with the checkout and test dependencies available:

```bash
python -m pytest tests/test_stark_density_floor.py tests/test_native_line_mean_validation.py tests/test_metal_state_charge_consistency.py tests/test_stark_and_opacity.py
python tools/validate.py fast
python tools/validate.py spectra --jobs 2
```

The numerical code has not changed since the validation results above.
Documentation and figure packaging do not establish full qualification.
The profile-area and Allard-combination concerns are outside this patch;
neither normalization nor Allard weighting is changed.
