# Terminal results, October 17

The bounded latest-source tier has 22 numerically accepted results and one unfinished DAO timeout from 23 requests. The selected grid is unchanged. Numerical acceptance does not establish physical accuracy or independent depth convergence.

- [Verification and all outcomes](TERMINAL_VERIFICATION.json), [accounting](ACCOUNTING.json).
- [DAB interpolation](DAB_INTERPOLATION.json): 0/4 independent targets pass both accuracy limits.
- [Cold native pairs](NATIVE_PAIRS.json), [completed hot spectra](HOT_SPECTRA.json).
- [Figure captions](CAPTIONS.md) and [inspection proof](FIGURE_VERIFICATION.json).
- PDFs: [DAB interpolation](dab_interpolation.pdf), [cold metals](cold_metal_changes.pdf), [hot spectra](hot_spectra.pdf).

The NPZ files retain native samples and derived predictions used by the figures. The full 7,239,886-byte raw delivery archive is preserved locally and in private scratch; its hash is in the verification record. Permanent archival remains pending. Scratch is not permanent storage.

Recreate the three plots from saved arrays, without atmosphere calculations:

```sh
python plot_results.py --inputs . --output reproduced-figures --style plot-style.mplstyle
```

The source-matched DAB inputs reproduce the earlier native spectra, but the four independent targets still fail the pointwise accuracy target. The DAO timeout remains unresolved. No automatic retry or grid expansion follows from this tier.
