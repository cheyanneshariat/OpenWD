# Observed DZ spectrum

`pg1225-uves.npz` contains the PG 1225-079 UVES coadd used by
[the fitting notebook](../fit_dz_spectrum.ipynb). No download is needed.

| Array | Meaning |
| --- | --- |
| `wave` | Vacuum, barycentric wavelength in Angstrom |
| `flux` | Flux density in 10^-16 erg cm^-2 s^-1 Angstrom^-1 |
| `error` | Approximate pipeline error in the same units |

`pg1225-uves.json` records the checksum, four source products, processing,
and all original FITS headers. Neighboring pixels have correlated errors.

The data are copyright ESO and distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
They come from programmes 165.H-0588(A) and 167.D-0407(A),
[UVES archive DOI 10.18727/archive/50](https://doi.org/10.18727/archive/50).

See [fitting and data-preparation notes](../../docs/examples/fitting-dz.md).
