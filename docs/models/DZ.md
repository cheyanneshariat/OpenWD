# DZ/DBZ module

[Model guide](README.md) · [Getting started](../getting-started.md)

`compute_dz` solves a helium-dominated polluted LTE atmosphere at fixed input
abundances. Metals contribute electrons, bound-free opacity, sampled line
blanketing, and therefore feed back on the relaxed structure. The default GD
40 mixture uses the Stout v3.00b4 line data used for every object in the
published DZ/DAZ comparison, with levels through charge 3, Verner
photoionization, dense-helium ionization shifts, and available unified Mg
I--He and Ca I--He profiles. Observable helium and trace-hydrogen lines use
the same policies as DB and DAB. Evaluated NIST replacements for matched
strong transitions remain available as the explicit
`strong_line_atomic_data="nist-asd"` alternative. Trace-hydrogen Lyman lines
likewise retain the figure's charged-particle Stark treatment by default;
`lyman_profile_source="allard"` explicitly selects the later unified-profile
option.

```bash
python examples/one_shot_dz.py --teff 15300 --logg 8.0 \
  --abundance O=-5.61 --abundance Mg=-6.24 --abundance Si=-6.76 \
  --abundance Ca=-6.88 --abundance Fe=-6.48 --log-h-he -6.16 \
  --quality standard --output results/gd40
```

Abundances are `log10[N(element)/N(He)]`; supplying any `--abundance` entries
replaces the entire default abundance dictionary. The model does not refit
`Teff`, `log g`, or composition. Standard line-rich atmospheres can take from
about 30 minutes to several hours. `dense_helium_eos="reos3"` is available in
the Python configuration as an explicitly experimental bulk-EOS option; the
validated production default remains the chemical-picture EOS.

Paper-spectrum regression and cold-start convergence are separate checks.
See [tested points](../tested-temperature-ranges.md) and
[reference-comparison limitations](../limitations.md#spectrum-accuracy-and-reference-comparisons)
for the status of PG 1225 and SDSS J0738+1835.

## Paper comparison

The paper compares seven polluted white dwarfs at fixed literature parameters
and abundances. Five have helium-dominated atmospheres and use DZ/DBZ; the
first and last panels are hydrogen-host [DAZ models](DAZ.md#paper-comparison).
The abundance labels in each panel specify the complete adopted mixture and
its H or He denominator.

| Object | Host | Teff (K) | log g | Observation | Parameter source |
| --- | --- | ---: | ---: | --- | --- |
| G149-28 | H | 8600 | 8.10 | DESI DR1 | [Zuckerman et al. (2011)](https://doi.org/10.1088/0004-637X/739/2/101) |
| PG 1225-079 | He | 10800 | 8.00 | VLT/UVES | [Klein et al. (2011)](https://doi.org/10.1088/0004-637X/741/1/64) |
| WD J1013+0259 | He | 12255 | 7.90 | VLT/X-shooter | [Izquierdo et al. (2023)](https://doi.org/10.1093/mnras/stad282) |
| SDSS J0738+1835 | He | 13950 | 8.40 | SDSS | [Dufour et al. (2012)](https://doi.org/10.1088/0004-637X/749/1/6) |
| WD J0259-0721 | He | 16390 | 8.26 | VLT/X-shooter | [Izquierdo et al. (2023)](https://doi.org/10.1093/mnras/stad282) |
| Ton 345 | He | 19780 | 8.18 | SDSS | [Wilson et al. (2015)](https://doi.org/10.1093/mnras/stv1201) |
| GALEX J1931+0117 | H | 20890 | 7.90 | VLT/UVES | [Vennes et al. (2011)](https://doi.org/10.1111/j.1365-2966.2011.18323.x) |

[![Observed spectra and fixed-composition predictions for seven polluted white dwarfs](../assets/dz-daz-paper.png)](../assets/dz-daz-paper.pdf)

[Download the comparison (PDF)](../assets/dz-daz-paper.pdf).
Every plotted atmosphere was relaxed at its displayed composition. Metal
electrons, continuum opacity and line blanketing participate in the structure
calculation; the final synthesis adds the detailed Stout line list. Helium
models use the nonideal He EOS, trace hydrogen where listed, dense-He metal
ionization shifts and ML2/alpha=1.25 convection. The reduced Ca II scattering
source affects the final line cores while retaining LTE populations.

Predictions are convolved to the corresponding instrumental resolution:
`R = 2000` (DESI), 19540 (PG 1225 UVES), 5453 (X-shooter), and 40970
(GALEX J1931 UVES), or Gaussian FWHM 2.23 and 2.57 Å for J0738 and Ton 345.
Data and models then receive the same independent pseudo-continuum procedure
and are plotted in rest-frame vacuum wavelengths. These display operations
leave the atmospheric parameters and abundances fixed.

The figure tests the metal-line pattern and normalized profiles across very
different mixtures. It does not establish an absolute continuum match or a
uniformly validated abundance grid. In particular, the older plotted J0738
atmosphere and a fresh calculation with the present certificate are distinct
checks; consult the linked tested-point record before treating it as a
qualified cold start.
