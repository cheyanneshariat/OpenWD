# Niobium and HS 0209+0832

[Development guide](README.md) · [DAB trace metals](../models/DAB.md#trace-metals)

Niobium (Z = 41) is available to every LTE metal preset: request `"Nb"` in
the abundances like any Stout element. It has a full Saha ladder (Nb I--VI),
partition functions from the NIST level lists, charge donation, and line
opacity with the ordinary thermal, radiative, neutral-perturber and optional
classical electron-Stark widths. The motivating star is HS 0209+0832, whose
Nb III and Nb IV lines were identified by Williams et al. (2026,
[arXiv:2610.07161](https://arxiv.org/abs/2610.07161)).

## Atomic data

`read_niobium_atomic_ion` builds the ions from checksum-pinned files in
`data/runtime/cache/metal-opacity/niobium/` (see its README and
`NIOBIUM_ATOMIC_DATA_FILES`):

| Ion | Levels | Lines | Transition probabilities |
| --- | --- | --- | --- |
| Nb I, II | 378, 353 | none | NIST has none |
| Nb III | 188 | 76 | HFR+CPOL gA, Nilsson et al. (2010) Table 7; only log gf > -0.5 |
| Nb IV | 182 | 819 | NIST ASD, mostly Tauheed & Reader (2005); grades B+ to E |
| Nb V, VI | 30, 104 | none | NIST has none |

Wavelengths are vacuum Ritz values from the NIST level energies, and f
follows from A. NIST and Iglesias (1955) give some strongly mixed Nb III
levels different LS names, so Table 7 rows are matched by both J values,
parity and wavelength (all 76 match uniquely within 0.08 A). All five Nb III
and 56 Nb IV lines that the paper lists have atomic matches within 0.015 A.
`IONIZATION_ENERGY_EV` now also carries the fourth and fifth NIST energies of
Al, Ti, Co, Ni, Cu and Zn, so hot models can extend those ladders to charge 4.

Limitations:

- No Nb photoionization data are bundled (Verner's fits stop at Zn). The
  thresholds of the populated Nb ions lie below 500 A, so the FUV continuum
  is unaffected, but the EUV omission is reported in the model metadata.
- Nb III lacks its weaker lines (log gf < -0.5), and Nb II and Nb V have no
  lines. Cool polluted stars, where Nb II dominates, therefore show no Nb
  lines yet.
- Lines use LTE populations; there is no Nb NLTE atom.

## HS 0209+0832 at the published parameters

The host is the [DAB preset with trace metals](../models/DAB.md#trace-metals):
35800 K, log g = 7.90, log H/He = 1.90 and the paper's nine abundances
(C, Al, Si, Ca, Ti, Ni, Cu, Zn and log Nb/H = -6.33), with ladders to charge
4 (Nb to 5) and classical electron-Stark metal widths. Nothing is fitted.

```bash
python research/hs0209_niobium.py model --output results/hs0209-niobium/standard
python research/hs0209_niobium.py windows --output results/hs0209-niobium/standard
python research/hs0209_niobium.py flux --output results/hs0209-niobium/standard
python research/hs0209_niobium.py model --control metal-free --output results/hs0209-niobium/control-metal-free
python research/hs0209_niobium.py model --control negligible-metals --output results/hs0209-niobium/control-negligible-metals
python research/plot_hs0209_niobium.py --model results/hs0209-niobium/standard \
    --raw RAW --paper-model CURVES.npz --output results/hs0209-niobium/figures/hs0209_figure1_openwd
```

`RAW` holds the MAST HST/STIS E140M HASP coadd and x1d (program 7473), the
CalFUSE `all` file of C0260201 and the STScI E140M line-spread functions.
`CURVES.npz` is optional: `research/extract_williams2026_figure1.py`
(PyMuPDF) recovers the published model curves from the vector graphics of
the paper's Figure 1, calibrating each panel on its own tick marks.

Results (standard quality, 2026-10-07, one Apple-silicon process):

- The cold start took 279 s with 1.9 GB peak memory and passed all five
  certificate gates. The emergent flux integrates to 1.00018 sigma T^4
  over 100 A--100 microns; 12.7% of it lies below 900 A.
- Nb is mostly Nb IV (54--84%) with Nb V (12--57%) and Nb III (0.4--4%)
  through the photosphere; Nb VI stays below 1%.
- With every metal at -20 the cold start reproduces the metal-free preset
  (above). The paper's metals warm the photosphere by 0.1--0.2% and change
  the 1000--1700 A continuum by at most 1%.

The model is scaled by (R/d)^2 with the paper's R = 0.0145 Rsun and
d = 82.6 pc, without reddening or continuum renormalization. It is shifted by
76.8 km/s for STIS (the paper's mean for Ca, Ti, Ni, Cu and Zn) and by
79.5 km/s for FUSE, the one fitted nuisance. The STIS projection uses the
STScI LSF on native pixels; FUSE uses a Gaussian with R = 20000 on the
coadd of the channels covering each window, each aligned to LiF1A.

| Window | chi2/pixel with Nb | without Nb | median data/model |
| --- | --- | --- | --- |
| FUSE 1002.5--1008 A | 2.22 | 7.14 | 1.075 |
| FUSE 1049--1057 A | 1.61 | 2.68 | 1.031 |
| STIS 1365--1370.5 A | 3.30 | 3.34 | 0.995 |
| STIS 1433--1436 A | 1.58 | 2.80 | 0.992 |
| STIS 1451.5--1454 A | 2.57 | 2.61 | 0.991 |
| STIS 1638.5--1643.5 A | 1.57 | 1.60 | 0.965 |

These chi-square values use the pipeline errors only and ignore pixel
covariance and the missing lines below, so they compare models rather than
measure goodness of fit. Nb reproduces the Nb IV 1434.14/1434.22 doublet
and the FUSE Nb IV lines. Measured on LiF1A alone, five of seven FUSE Nb IV
equivalent widths agree with the Nb-only model within 2 sigma; the model is
too strong for 1002.76 A (40 versus 20 +/- 7 mA, although LiF2B gives
38 +/- 9 mA) and 1005.70 A (83 versus 57 +/- 6 mA). The model cores are deeper than the FUSE coadd, which is
smoothed by residual channel misalignment and an uncertain resolution. The
OpenWD FUSE continuum lies 3--8% below the data and the paper's fit.

Remaining differences come from missing atomic data, not Nb. Stout has only
10 Zn IV and 23 Cu IV transitions and lacks Ni IV 1452.22 A, so those
features in the 1366 A and 1452 A windows are absent. The paper used the
Rauch et al. (2014, 2020) Zn IV and Cu IV data; adding them is the next step.
