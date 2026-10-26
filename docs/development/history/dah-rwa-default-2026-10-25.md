# DAH default: RWA photoionization and magnetic EOS

[Development guide](../README.md) · [DAH user guide](../../models/DAH.md)

From 2026-10-25 the DAH public default evaluates, above 1 MG and in every
surface cell, the magnetic Saha/HM equilibrium with centered-motion
populations and the stationary-state rigid-wavefunction (RWA) H I
photoionization of Rohrmann (2026, A&A; arXiv:2603.07679), on the unchanged
nonmagnetic DA structure. The previous default, used for the paper figure, is
the named preset `paper_dah_config` (`DAH_PAPER_PRESCRIPTION`). The coupled
mean-field structure stays opt-in. The DAH physics revision is now
`dah-rwa-magnetic-eos-v5`.

## Implementation check

At 20000 K and ρ = 10⁻⁸ g cm⁻³ (no stimulated emission, as published) the
OpenWD RWA absorption reproduces Rohrmann's Zenodo table
(doi:10.5281/zenodo.19005375) to ±0.02 dex at 3700–7000 Å in all three
polarizations up to 47 MG, and to ~0.1 dex at 94–235 MG. Larger excursions
occur only at edges that move between the table's 0.1-dex field steps, or come
from its n = 9–17 shells, which OpenWD's explicit RWA omits. Photoionization
thresholds agree with Zhao & Stancil (2007) to 0.05%.

## Validity

The RWA shifts each zero-field cross section by its state's threshold
displacement, including the Landau zero-point energy (Rohrmann 2026, Eq. 56).
Compared with the resonance-averaged Zhao & Stancil (2007) quantum cross
sections, individual transitions agree to 0.06 dex at 23.5 MG and 0.1–0.3 dex
at 70–211 MG, but differ by 1–2 dex in the optical at 235 MG (e.g. 3d₋₁ π
+0.9 dex; 3p₋₁ and 4f σ⁺ −0.5 to −2 dex). Requests whose visible field exceeds
200 MG therefore record a domain note.

## Observational screen

Noise-weighted χ² per pixel after a cubic multiplicative continuum, at fixed
published parameters, standard quality, on certified nonmagnetic structures:

| Object | B_p (MG) | Paper prescription | New default | Hardy et al. |
| --- | ---: | ---: | ---: | ---: |
| J1034+0327 | 11 | 3.90 | 3.32 | 3.48 |
| J1154+0117 | 36 | 1.34 | 1.30 | 1.36 |
| J2149−0728 | 45 | 1.71 | 1.74 | 1.74 |
| J0908+0921 | 52 | 1.12 | 1.08 | 0.97 |
| J1254+5612 | 60 | 1.27 | 1.26 | 1.25 |
| J1018+0111 | 108 | 5.81 | 5.31 | – |
| J0732+3646 | 111 | 1.24 | 1.17 | 1.17 |
| GH Leo (J1033+2309) | 230 | 4.62 | 4.07 | 2.89 |
| J1351+5419 | 369 | 4.82 | 3.03 | 2.68 |
| J2247+1456 | 437 | 2.99 | 4.22 | – |

The default improves eight objects, ties J2149, and worsens J2247, whose
4100–4450 Å region it over-absorbs. With one flux scale only (no continuum
freedom) J1034, J1154, J0732 and GH Leo are slightly worse; J1034's
spectrophotometry is itself suspect. The coupled mean-field option scored
better on six objects but worse on J1154 and J2247 and costs ~7×; it was not
adopted.

## Cost

At standard quality the magnetic synthesis takes ~50 s instead of ~22 s per
dipole star (21 field bins); a full cold run rises from ~45 s to ~75 s. The
structure is unchanged.

## Controls

The eight frozen paper spectra are untouched and test `paper_dah_config` at
their original 2e-8 tolerance. Approved default spectra on the same atmospheres
are in `tests/data/approved_regressions/dah_rwa_default_2026_10_25`, captured
offline by `research/capture_dah_rwa_default_controls.py` after recovering the
paper spectra to ≤1.1e-13. Changes from the paper spectra: up to 5.8–12.3% near
3800 Å below 80 MG of visible field, 27–45% for J1254, J1018 and J1351, and
integrated flux from −5.1% (J2247) to +6.3% (J1351). The J1007 and J1254 cold
canaries now compare against the approved default spectra.

## Guide and paper figure

The DAH guide's eight-object figure (`docs/assets/dah-paper-*.{pdf,png}`) was
regenerated from the approved default spectra on the paper's certified
structures, using the archived 2026-09-29 comparison procedure and velocities
(replayed exactly from the old spectra before substitution). The model line now
uses the shared paper style (`#d95f02`, 0.6 pt; observations gray 0.30, 0.5 pt).
The smooth-rescaling factors span 0.78–1.16 (previously 0.81–1.19).
