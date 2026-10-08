# DZ fitting example

The [notebook](../../examples/fit_dz_spectrum.ipynb) fits Ca, Mg and Fe in
PG 1225-079. It loads an observed spectrum from the repository, solves one
atmosphere, and scans abundances on that fixed thermal structure.
The notebook needs only the normal OpenWD dependencies and Jupyter.

## Observed spectrum

The [included coadd](../../examples/data/README.md) combines four 300-second
UVES blue-arm exposures from programmes 165.H-0588(A) and 167.D-0407(A).
The companion JSON file contains the source URLs, SHA-256 checksums and
unaltered primary and spectrum-table FITS headers. It also identifies the
coadd's units, wavelength frame and processing. This is a selected example,
not a complete inventory of the archive observations.

The original products use air wavelengths in the topocentric frame. The
preparation code uses the same iterated air-to-vacuum formula as OpenWD
(Morton 1991), then applies each header's barycentric correction. It linearly
interpolates flux and error amplitudes onto a 0.03-Angstrom grid. It scales
the exposures to the first spectrum in 4150–4400 Angstrom, then forms an
inverse-variance mean. These diagonal weights do not propagate resampling
covariance. The nominal resolving power is 19540; the stellar line-spread
function is not measured. The [ESO release description](https://www.eso.org/sci/observing/phase3/data_releases/uves_echelle_v1.2.pdf)
describes the pipeline and its flux-calibration limits.
Model spectra use 2 km/s steps, Gaussian broadening and point interpolation.
Pixel integration and finer-grid sensitivity are not quantified here.

To rebuild the coadd, install the optional `validation` extra for `astropy`
and run from the repository root:

```bash
python docs/examples/prepare_pg1225.py --raw-dir results/pg1225-raw --output results/pg1225-rebuilt.npz
```

The [preparation script](prepare_pg1225.py) checks cached FITS files before
use and refuses to overwrite an output. New downloads keep certificate and
hostname verification enabled. It uses the configured trust store, an
installed `certifi` bundle, or the macOS system bundle. If no trusted bundle
is available, it reports how to set `SSL_CERT_FILE`. No packages or security
settings are changed by the script.

## What the fit estimates

Teff = 10800 K, log g = 8.0, H/He and the starting metal abundances come
from Table 5 of [Klein et al. (2011)](https://doi.org/10.1088/0004-637X/741/1/64).
They are log number ratios relative to helium. Teff and log g are not fitted.
The paper's 10500/7.7 and 11100/8.3 solutions give similar He I 5876
strengths and shift the absolute abundances by roughly 0.2–0.4 dex.

Each trial recomputes populations, electron contributions and opacity on
the fixed thermal/depth structure. It does not restore equilibrium for that
mixture. The five numerical certificate gates apply to the starting
atmosphere's declared equations and grid. They do not establish complete
physics, grid independence or observational accuracy.

The fit uses a separate velocity nuisance shift for each element and a
linear continuum in each window. The velocity spread is not a systemic-RV
measurement. Ca II H/K cores are excluded because their emission is not
reproduced. Klein et al. discuss NLTE effects and chromospheric activity as
possible causes; neither is established. No separate interstellar or
circumstellar absorber is fitted. The missing Ti II region near 3685 Angstrom
is outside all fit windows; Ti stays fixed.

The printed widths are local chi-square curvature scales. They omit
resampling covariance, nuisance-parameter uncertainty and model errors.
Reduced chi-square below one does not calibrate those widths. The Fe windows
contain Fe I: the paper gives Fe I/He = −7.50 separately from its combined
Fe/He = −7.42. The paper used Keck/HIRES data and Koester/VALD models.
This example uses UVES data and OpenWD/Stout, and omits undetected elements.
The literature comparison is therefore not a calibrated bias test.

## Saved example and structural feedback

At source commit `39664f1`, the original 12-cell run completed in 44.4 minutes.
The production cold start took 27.4 minutes and passed all five required
certificate gates. The fixed-structure fit gave:

| Element | Initial structure | Nearby re-solved structure | Change (dex) |
| --- | ---: | ---: | ---: |
| Ca | −7.791 | −7.810 | −0.019 |
| Mg | −7.297 | −7.301 | −0.004 |
| Fe | −7.481 | −7.500 | −0.019 |

The separate 25.7-minute cold solve used an earlier provisional mixture:
Ca = −7.79, Mg = −7.30 and Fe = −7.52. Its certificate and live code/data
fingerprint were verified. Its Fe input differs from the completed fit by
0.039 dex. A 14-trial refit used the original local quadratic sample points,
the same sequential fitting conditions and the same velocity shifts. It
took 11.8 minutes on one CPU. The Ca shift exceeds its formal curvature
width, even though the shift is small compared with the Teff/log g sensitivity.

This is one nearby-mixture feedback check. A cold solve and spectral refit
at the resulting composition are still needed for an iterated final solution.
No atmosphere grid, MCMC run or new physics qualification is implied.
