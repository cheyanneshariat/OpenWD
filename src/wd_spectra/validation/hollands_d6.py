"""Comparison with the digitized Hollands et al. (2025) J1637 Figure 1.

Hollands et al. (2025, MNRAS 541, 2231) show their GTC/OSIRIS spectrum of
SDSS J1637+3631 and their best Koester-code model in one broad panel (F_nu,
after multiplying the model by a fifth-order polynomial fit to data/model)
and four locally normalized panels.  All wavelengths are vacuum and in the
stellar rest frame.  The digitization records the plotted vector paths; it
is a regression target, not the authors' numerical arrays.

Two panel metrics are reported for a model convolved to R=2500:

* the development-history metric normalizes each panel with a robust
  log-linear continuum through its outer 12 per cent and reports the RMS
  difference from the digitized curve.  It is kept unchanged so that scores
  remain comparable across revisions, but its floor is not zero: the authors
  normalized their panels differently, and the digitized Koester model scores
  0.030, 0.043, 0.004 and 0.010 against itself in panels 1-4;
* the matched metric divides out a smooth quadratic continuum ratio between
  the model and the digitized curve (median-binned, as in the paper's own
  re-fluxing) before taking the RMS.  A perfect model scores zero, so it
  measures line and local-shape differences rather than normalization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .observed import local_normalize

FloatArray = NDArray[np.float64]

HOLLANDS2025_PANEL_WINDOWS = (
    (3580.0, 4620.0),
    (4580.0, 5620.0),
    (5580.0, 6620.0),
    (6580.0, 7620.0),
)
HOLLANDS2025_RESOLVING_POWER = 2500.0
DEFAULT_DIGITIZATION = (
    Path(__file__).resolve().parent.parent
    / "data" / "validation" / "hollands2025_j1637_figure1.npz"
)


def convolve_to_resolving_power(
    wavelength: ArrayLike, flux: ArrayLike, resolving_power: float
) -> FloatArray:
    """Gaussian convolution at constant resolving power (development metric)."""

    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    log_grid = np.linspace(np.log(wavelength[0]), np.log(wavelength[-1]), wavelength.size)
    sampled = np.interp(log_grid, np.log(wavelength), flux)
    sigma_pixel = 1.0 / (
        2.0 * np.sqrt(2.0 * np.log(2.0))
        * resolving_power * (log_grid[1] - log_grid[0])
    )
    radius = max(1, int(np.ceil(4.0 * sigma_pixel)))
    offset = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offset / sigma_pixel) ** 2)
    kernel /= np.sum(kernel)
    convolved = np.convolve(np.pad(sampled, radius, mode="reflect"), kernel, mode="same")
    return np.interp(np.log(wavelength), log_grid, convolved[radius:-radius])


def load_hollands2025_digitization(path: str | Path | None = None):
    source = DEFAULT_DIGITIZATION if path is None else Path(path)
    with np.load(source) as saved:
        return {name: saved[name] for name in saved.files}


def hollands2025_panel_rms(
    wavelength: ArrayLike,
    convolved_flux: ArrayLike,
    *,
    target: str = "koester",
    digitization: dict | None = None,
) -> list[float]:
    """Normalized RMS residuals of an R=2500 model in the four panels."""

    if target not in ("koester", "observed"):
        raise ValueError("target must be 'koester' or 'observed'")
    published = load_hollands2025_digitization() if digitization is None else digitization
    residuals = []
    for panel, (lower, upper) in enumerate(HOLLANDS2025_PANEL_WINDOWS, start=1):
        normalized_wave, normalized_model = local_normalize(
            wavelength, convolved_flux, lower, upper, continuum_fraction=0.12
        )
        paper_wave = published[f"panel{panel}_{target}_wavelength"]
        paper_flux = published[f"panel{panel}_{target}_normalized_flux"]
        sampled = np.interp(paper_wave, normalized_wave, normalized_model)
        valid = np.isfinite(paper_flux)
        if target == "observed":
            # Exclude figure-frame clipping and the lowest-S/N blue spikes.
            valid &= (paper_flux > 0.05) & (paper_flux < 1.35)
        residuals.append(float(np.sqrt(np.mean((sampled[valid] - paper_flux[valid]) ** 2))))
    return residuals


def hollands2025_matched_panel_rms(
    wavelength: ArrayLike,
    convolved_flux: ArrayLike,
    *,
    target: str = "koester",
    digitization: dict | None = None,
    degree: int = 2,
    n_bins: int = 40,
) -> list[float]:
    """RMS after removing a smooth continuum ratio to the digitized panel."""

    if target not in ("koester", "observed"):
        raise ValueError("target must be 'koester' or 'observed'")
    published = load_hollands2025_digitization() if digitization is None else digitization
    wavelength = np.asarray(wavelength, dtype=np.float64)
    flux = np.asarray(convolved_flux, dtype=np.float64)
    residuals = []
    for panel, (lower, upper) in enumerate(HOLLANDS2025_PANEL_WINDOWS, start=1):
        paper_wave = published[f"panel{panel}_{target}_wavelength"]
        paper_flux = published[f"panel{panel}_{target}_normalized_flux"]
        order = np.argsort(paper_wave)
        paper_wave, paper_flux = paper_wave[order], paper_flux[order]
        valid = (
            np.isfinite(paper_flux) & (paper_flux > 0.05)
            & (paper_wave >= wavelength[0]) & (paper_wave <= wavelength[-1])
        )
        if target == "observed":
            valid &= paper_flux < 1.35
        paper_wave, paper_flux = paper_wave[valid], paper_flux[valid]
        model = np.interp(paper_wave, wavelength, flux)
        x = 2.0 * (paper_wave - lower) / (upper - lower) - 1.0
        log_ratio = np.log(paper_flux / model)
        edges = np.linspace(x.min(), x.max(), n_bins + 1)
        bx, by = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            inside = (x >= low) & (x <= high)
            if np.count_nonzero(inside) >= 3:
                bx.append(np.median(x[inside]))
                by.append(np.median(log_ratio[inside]))
        coefficient = np.polynomial.polynomial.polyfit(bx, by, degree)
        matched = model * np.exp(np.polynomial.polynomial.polyval(x, coefficient))
        residuals.append(float(np.sqrt(np.mean((matched - paper_flux) ** 2))))
    return residuals


def hollands2025_refluxed_broad_model(
    wavelength: ArrayLike,
    convolved_flux_lambda: ArrayLike,
    *,
    digitization: dict | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Apply the paper's fifth-order data/model re-fluxing in F_nu.

    Returns the model F_nu shape multiplied by the clipped fifth-order
    polynomial fit to observed/model, in the digitized mJy units.
    """

    published = load_hollands2025_digitization() if digitization is None else digitization
    wavelength = np.asarray(wavelength, dtype=np.float64)
    shape = np.asarray(convolved_flux_lambda, dtype=np.float64) * wavelength**2
    observed_wave = published["broad_observed_wavelength"]
    observed_flux = published["broad_observed_flux_mjy"]
    sampled = np.interp(observed_wave, wavelength, shape)
    valid = (
        (observed_wave >= wavelength[0]) & (observed_wave <= wavelength[-1])
        & np.isfinite(observed_flux) & (observed_flux > 0.003)
    )
    fit_wave = observed_wave[valid]
    ratio = observed_flux[valid] / np.maximum(sampled[valid], np.finfo(float).tiny)
    x = 2.0 * (fit_wave - fit_wave.min()) / np.ptp(fit_wave) - 1.0
    retained = np.ones(fit_wave.size, dtype=bool)
    coefficient = np.polynomial.polynomial.polyfit(x, ratio, 5)
    for _ in range(4):
        residual = ratio - np.polynomial.polynomial.polyval(x, coefficient)
        scale = 1.4826 * np.median(np.abs(residual[retained] - np.median(residual[retained])))
        if not np.isfinite(scale) or scale <= 0.0:
            break
        retained = np.abs(residual) < 4.0 * scale
        coefficient = np.polynomial.polynomial.polyfit(x[retained], ratio[retained], 5)
    model_x = 2.0 * (wavelength - fit_wave.min()) / np.ptp(fit_wave) - 1.0
    return wavelength, shape * np.polynomial.polynomial.polyval(model_x, coefficient)


def hollands2025_scores(
    wavelength: ArrayLike,
    flux_lambda: ArrayLike,
    *,
    digitization: dict | None = None,
    already_convolved: bool = False,
) -> dict[str, object]:
    """Return the standard J1637 regression scores for an intrinsic spectrum."""

    published = load_hollands2025_digitization() if digitization is None else digitization
    wavelength = np.asarray(wavelength, dtype=np.float64)
    convolved = (
        np.asarray(flux_lambda, dtype=np.float64)
        if already_convolved
        else convolve_to_resolving_power(wavelength, flux_lambda, HOLLANDS2025_RESOLVING_POWER)
    )
    koester = hollands2025_panel_rms(wavelength, convolved, digitization=published)
    observed = hollands2025_panel_rms(
        wavelength, convolved, target="observed", digitization=published
    )
    matched = hollands2025_matched_panel_rms(wavelength, convolved, digitization=published)
    matched_observed = hollands2025_matched_panel_rms(
        wavelength, convolved, target="observed", digitization=published
    )
    broad_wave, broad_model = hollands2025_refluxed_broad_model(
        wavelength, convolved, digitization=published
    )
    paper_wave = published["broad_koester_wavelength"]
    paper_flux = published["broad_koester_flux_mjy"]
    inside = (paper_wave >= broad_wave[0]) & (paper_wave <= broad_wave[-1])
    broad_ratio = np.interp(paper_wave[inside], broad_wave, broad_model) / paper_flux[inside]
    return {
        "koester_panel_rms": koester,
        "koester_mean_panel_rms": float(np.mean(koester)),
        "observed_panel_rms": observed,
        "observed_mean_panel_rms": float(np.mean(observed)),
        "koester_matched_panel_rms": matched,
        "koester_mean_matched_panel_rms": float(np.mean(matched)),
        "observed_matched_panel_rms": matched_observed,
        "observed_mean_matched_panel_rms": float(np.mean(matched_observed)),
        "broad_refluxed_koester_rms_fractional": float(
            np.sqrt(np.mean((broad_ratio - 1.0) ** 2))
        ),
    }
