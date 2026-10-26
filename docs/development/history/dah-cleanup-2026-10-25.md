# DAH clean-up and convergence repair

[Development guide](../README.md) · [DAH user guide](../../models/DAH.md)

An audit of the DAH module (12 observed objects at fixed published
parameters, both public prescriptions) found duplicated and dead code, one
structure/synthesis inconsistency, a validation metric that could not rank
models, and two cold starts that failed at `quality="standard"`. This change
fixes those without altering the default physics. Physics findings from the
audit (high-field continuum and field-spread quadrature) are not addressed here.

## Code

- Removed unused code: a duplicate RWA structure continuum, the unreachable
  n>8 RWA branching branch, an unused H2db threshold method and stored but
  never-read ground-state arrays.
- One implementation each of e–p free–free extinction, H I shell
  photoionization cross sections and the Humlicek W4 Faddeeva function
  (`_faddeeva.py`), shared by the ordinary and magnetic continua and the
  metal line profiles. The magnetic copy of W4 had truncated coefficients;
  the opt-in cyclotron spectrum changes by <6e-9 relative.
- The mean-field structure now builds its Balmer opacity with the same
  `balmer_profile` and `normalize_balmer_strength` as the synthesis. It
  previously always used unified, unnormalized profiles, so a mean-field run
  with the default Kurucz/Griem options mixed two prescriptions.
- Centered-motion substate energies and mass ratios depend only on the field
  and are now cached per field; a 50-depth magnetic EOS call with centered
  motion fell from ~2.9 s to 0.25 s, with identical results.
- The DAH request fingerprint ignores `polarized_transfer`, which never
  affects either structure, so a scalar synthesis on an IQUV run's structure
  keeps its certificate instead of warning about a request mismatch.

On fixed structures all default, mean-field, IQUV, magnetic-EOS and RWA
outputs are bit-identical to `ef04c54` (104 arrays compared), apart from the
cyclotron change above.

## Validation metric

The locally normalized `feature_rms` is dominated by photon noise: the
published Hardy et al. (2023) models score within 1e-3 of ours on every SDSS
target. The bundled spectra now include the SDSS/BOSS inverse variances
(existing arrays unchanged bit for bit), and `score_dah_spectrum` reports
noise-weighted χ² per pixel after one flux scale (`chi2_scale`) and after a
cubic multiplicative continuum (`chi2_cubic`), for the model and the Hardy
curves. GH Leo has no published errors; its noise is estimated and labelled.

## Convergence

DAH suppresses convection above 0.05 MG, so every cool DAH has a radiative
structure. Two cold starts failed at `standard` quality:

| Case | Before | After |
| --- | --- | --- |
| G 76−48, default (6680 K) | 40 layers, flux residual 0.92, trust region collapsed | 40 → 100 → 200 layers, certified (flux residual 1e-12, ~19 min) |
| J1018+0111, mean-field (10500 K) | 40 layers, flux residual 0.089, stalled | 40 → 100 layers, certified (22 min including IQUV synthesis) |

Production DA already re-solved a failed 100-layer radiative attempt on 200
layers. Standard DA now follows the same meshes (40 → 100 → 200), and the
mean-field DAH structure, which had no recovery path, uses the same ladder.
Doubling from 40 was tried first and does not work for G 76−48: the 80- and
160-layer solves stall at flux residuals 0.7 and 0.40, because 200 layers are
needed to resolve its ionization transition on the uniform DAH mesh, and a
100-layer attempt is the seed production relies on. Converged solves are never
refined; each refined solve must pass its own certificate.

The magnetic Saha/pressure iteration raises if it does not converge. It
converged at all 1200 (T, P) points between 3000 and 100000 K and
1e-3–1e10 dyn cm⁻² at 1–1000 MG, with and without centered motion, so it
remains a hard error rather than silently returning an unconverged EOS.

The converged 200-layer G 76−48 structure then exposed a latent crash in
full-IQUV synthesis with Kurucz/Griem profiles: the template mesh is the union
of per-depth meshes, and nodes from depths with nearly equal electron density
coincided to rounding on either side of another depth's β=20 discontinuity
(documented in `kurucz_griem.py`). The magneto-optical transform correctly
rejects two values at one frequency. The mesh now drops nodes within 64 ulp in
frequency of their neighbour, so the jump spans distinct nodes; default
spectra and the paper fixtures are unchanged.

New canaries: `da-radiative-g76-48-standard` and `dah-mean-field-j1018`.

## Legacy code

The pre-release development tree held a second, larger implementation
(moving-atom EOS, field-dependent atomic sizes, Zhao–Liu E+B Hα, Zhao–Stancil
photoionization, two-mode transfer, tuned Stark widths). It is archived outside
this repository with checksums; its README lists the ingredients not ported
and the evidence for each. None is needed by OpenWD.
