# Host-ion microfield fixed-atmosphere controls (2026-10-12)

`metal_ionic_microfield_perturber_density` sets the charged-perturber density used by
metal-level dissolution (Q-MHD occupation probabilities) in the metal line opacity, the O I
dissolved-series pseudo-continuum and the linear-Stark manifold. It previously summed the
metal ions only. In a trace-metal atmosphere the host ions dominate the microfield: in GD 61
the metal-only moment was 1e-3 to 2e-5 of the electron density at every depth, while the
partition functions already used the electron density. The moment now also counts the host
He+/He++ ions and any remaining singly charged donors (n_e minus metal and host electrons, e.g.
H+). In a bulk-metal plasma (D6) it is unchanged.

Only the two DZ structures change; DAZ, D6 and the other frozen controls are unchanged. The
structures, configurations and wavelength grids are unchanged, and the previous controls in
`../si_kurucz_2026_10_12/fixed/` are kept. In the regression test, patching in the previous
metal-only moment must recover them; at capture it did so exactly.

| Fixed historical structure | Largest fractional pixel change | 1150--3000 Å band change |
| --- | ---: | ---: |
| PG 1225-079 | 4.0e-5 | +5e-7 |
| SDSS J0738+1835 | 2.9% | +0.013% |

`manifest.json` records the input, previous-control and output hashes, the request
configuration and the source hashes. Capture used
`research/capture_microfield_host_controls.py`, which refuses an existing output directory.
These structures were not re-relaxed; this is **not a cold-start qualification**.
