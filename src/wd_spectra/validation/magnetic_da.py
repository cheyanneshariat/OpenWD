"""Observed magnetic-DA (DAH) validation targets and fixed-parameter scores.

Eight DAHs from the official CDS release of Hardy, Dufour & Jordan (2023,
MNRAS 520, 6111): six well-fit offset dipoles (their Table 5) and two harder
Table 6 stress tests, plus the two independent Vera-Rueda & Rohrmann (2024,
A&A 687, A141) reproductions J1018+0111 and J2247+1456.  Published Teff,
log g, polar field, inclination and offset are held fixed; only a radial
velocity and one flux scale (5200--6100 A) are nuisance parameters.

``score_dah_spectrum`` returns

* ``broad_rms``: RMS of the ratio of 12-pixel-smoothed model and observation
  over 3820--6950 A, after the single flux scale (the metric of the
  development-tree validation suite, so values are directly comparable);
* ``blue_ratio``: median smoothed model/observation over 3820--4400 A;
* ``feature_rms``: RMS difference of the locally normalized spectra
  (each divided by its own 150-pixel running mean), which isolates the
  magnetic line components from the continuum slope.

The same scores of the digitized Hardy model are reported as a reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .._compat import trapezoid
from ..constants import LIGHT_SPEED
from .observed import convolve_variable_log_gaussian, gaussian_smooth


@dataclass(frozen=True)
class DAHValidationTarget:
    key: str
    display: str
    effective_temperature: float
    logg: float
    polar_field_megagauss: float
    inclination_deg: float
    offset_radius: tuple[float, float, float]
    ebv: float
    reference: str
    source_file: str
    source_format: str = "sdss-fits"
    hardy_curve_file: str | None = None
    resolution_fwhm_angstrom: float | None = None

    def config_kwargs(self) -> dict[str, object]:
        return dict(
            effective_temperature=self.effective_temperature,
            logg=self.logg,
            magnetic_field_megagauss=self.polar_field_megagauss,
            field_geometry="dipole",
            field_strength_definition="dipole-polar",
            dipole_inclination_deg=self.inclination_deg,
            dipole_offset_radius=self.offset_radius,
        )


_CACHE = ".cache/magnetic-validation/"
_HARDY = _CACHE + "hardy-cds/model-curves/"

DAH_VALIDATION_TARGETS = (
    DAHValidationTarget(
        "j1034+0327", "J1034+0327 / HS 1031+0343", 15_756.0, 8.80, 11.17, 77.0,
        (0.0, 0.0, 0.09), 0.003, "Hardy et al. 2023, Table 5",
        "results/magnetic-da-continuum-audit-v1/j1034-photometric-recalibration/"
        "J1034+0327-photometry-recalibrated.fits",
        hardy_curve_file=_HARDY + "table5_J1034+0327-model.npz",
    ),
    DAHValidationTarget(
        "j2149-0728", "J2149-0728 / WD 2146-077", 22_642.0, 8.37, 45.09, 66.0,
        (0.0, 0.0, 0.17), 0.021, "Hardy et al. 2023, Table 5",
        _CACHE + "sdss-strong/J2149-0728.fits",
        hardy_curve_file=_HARDY + "table5_J2149-0728-model.npz",
    ),
    DAHValidationTarget(
        "j1254+5612", "J1254+5612", 12_870.0, 8.58, 60.43, 62.0,
        (0.0, 0.0, 0.18), 0.014, "Hardy et al. 2023, Table 5",
        _CACHE + "sdss-strong/J1254+5612.fits",
        hardy_curve_file=_HARDY + "table5_J1254+5612-model.npz",
    ),
    DAHValidationTarget(
        "j0908+0921", "J0908+0921", 29_293.0, 8.93, 51.55, 54.0,
        (0.0, 0.0, 0.24), 0.012, "Hardy et al. 2023, Table 5",
        _CACHE + "sdss-extended/J0908+0921.fits",
        hardy_curve_file=_HARDY + "table5_J0908+0921-model.npz",
    ),
    DAHValidationTarget(
        "j1154+0117", "J1154+0117 / WD 1151+015", 29_316.0, 8.90, 35.53, 87.0,
        (0.0, 0.0, -0.23), 0.016, "Hardy et al. 2023, Table 5",
        _CACHE + "sdss-extended/J1154+0117.fits",
        hardy_curve_file=_HARDY + "table5_J1154+0117-model.npz",
    ),
    DAHValidationTarget(
        "j0732+3646", "J0732+3646", 25_904.0, 9.31, 111.49, 63.0,
        (0.0, 0.0, 0.04), 0.013, "Hardy et al. 2023, Table 5",
        _CACHE + "sdss-strong/J0732+3646.fits",
        hardy_curve_file=_HARDY + "table5_J0732+3646-model.npz",
    ),
    DAHValidationTarget(
        "j1018+0111", "J1018+0111", 10_500.0, 8.0, 108.12, 60.0,
        (0.0, 0.07, 0.10), 0.0, "Vera-Rueda & Rohrmann 2024",
        _CACHE + "vera-rueda-2024/spec-0503-51999-0244.fits",
    ),
    DAHValidationTarget(
        "j1033+2309", "J1033+2309 / GH Leo", 26_025.0, 8.77, 230.49, 40.0,
        (0.0, 0.0, 0.26), 0.002, "Hardy et al. 2023, Table 6",
        _CACHE + "sdss-extended/J1033+2309-Gianninas.txt",
        source_format="mwdd-fnu-csv",
        hardy_curve_file=_HARDY + "table6_J1033+2309-model.npz",
        resolution_fwhm_angstrom=6.0,
    ),
    DAHValidationTarget(
        "j1351+5419", "J1351+5419 / WD 1349+545", 13_937.0, 8.43, 368.52, 34.0,
        (0.0, 0.0, 0.07), 0.007, "Hardy et al. 2023, Table 6",
        _CACHE + "sdss-extended/J1351+5419.fits",
        hardy_curve_file=_HARDY + "table6_J1351+5419-model.npz",
    ),
    DAHValidationTarget(
        "j2247+1456", "J2247+1456", 19_000.0, 8.0, 437.1, 10.0,
        (0.0, 0.0, -0.15), 0.0, "Vera-Rueda & Rohrmann 2024",
        _CACHE + "sdss-high-field/spec-5040-56243-0482.fits",
    ),
)
DAH_VALIDATION_BY_KEY = {target.key: target for target in DAH_VALIDATION_TARGETS}


@dataclass(frozen=True)
class DAHObservation:
    wavelength_angstrom: np.ndarray
    flux_nu: np.ndarray
    sigma_log_wavelength: np.ndarray
    hardy_wavelength_angstrom: np.ndarray | None
    hardy_flux: np.ndarray | None


def load_dah_observation(key: str, root: str | Path | None = None) -> DAHObservation:
    """Load one bundled observation (and Hardy curve when available)."""

    if key not in DAH_VALIDATION_BY_KEY:
        raise KeyError(f"unknown DAH validation target {key!r}")
    directory = (
        Path(str(files("wd_spectra").joinpath("data/validation/dah")))
        if root is None
        else Path(root)
    )
    with np.load(directory / f"{key}.npz") as saved:
        return DAHObservation(
            np.asarray(saved["wavelength_angstrom"]),
            np.asarray(saved["flux_nu"]),
            np.asarray(saved["sigma_log_wavelength"]),
            np.asarray(saved["hardy_wavelength_angstrom"]) if "hardy_flux" in saved else None,
            np.asarray(saved["hardy_flux"]) if "hardy_flux" in saved else None,
        )


def _fit_velocity(observed_wave, observed_fnu, model_wave, model_fnu) -> tuple[float, float]:
    """Cross-correlate continuum-insensitive log-flux derivatives."""

    def derivative(wave, flux):
        log_flux = gaussian_smooth(np.log(np.maximum(flux, np.finfo(float).tiny)), 4.0)
        slope = np.gradient(log_flux, np.log(wave))
        return slope - gaussian_smooth(slope, 60.0)

    observed_derivative = derivative(observed_wave, observed_fnu)
    model_derivative = derivative(model_wave, model_fnu)
    velocities = np.arange(-200.0, 200.01, 2.0)
    scores = []
    for velocity in velocities:
        rest = observed_wave / (1.0 + velocity * 1.0e5 / LIGHT_SPEED)
        selected = (rest >= 3_850.0) & (rest <= 6_900.0)
        a = observed_derivative[selected]
        b = np.interp(rest[selected], model_wave, model_derivative)
        a = (a - a.mean()) / a.std()
        b = (b - b.mean()) / b.std()
        scores.append(float(np.mean(a * b)))
    index = int(np.argmax(scores))
    correlation = float(scores[index])
    # A boundary optimum or weak correlation does not constrain the velocity.
    if correlation < 0.15 or index in (0, velocities.size - 1):
        return 0.0, 0.0
    return float(velocities[index]), correlation


def _scores(rest_wave: np.ndarray, observed: np.ndarray, model: np.ndarray) -> dict[str, float]:
    control = (rest_wave >= 5_200.0) & (rest_wave <= 6_100.0)
    model = model * float(np.median(observed[control] / model[control]))
    ratio = gaussian_smooth(model, 12.0) / gaussian_smooth(observed, 12.0)
    full = (rest_wave >= 3_820.0) & (rest_wave <= 6_950.0)
    blue = (rest_wave >= 3_820.0) & (rest_wave <= 4_400.0)
    local_observed = observed / gaussian_smooth(observed, 150.0)
    local_model = model / gaussian_smooth(model, 150.0)
    inner = full & (rest_wave > rest_wave[0] + 60.0) & (rest_wave < rest_wave[-1] - 60.0)
    return {
        "broad_rms": float(np.sqrt(np.mean((ratio[full] - 1.0) ** 2))),
        "blue_ratio": float(np.median(ratio[blue])),
        "feature_rms": float(
            np.sqrt(np.mean((local_model[inner] - local_observed[inner]) ** 2))
        ),
    }


def score_dah_spectrum(
    key: str,
    model_wavelength_angstrom: ArrayLike,
    model_flux_lambda: ArrayLike,
    *,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Score a rest-frame vacuum surface-flux spectrum against one target."""

    observation = load_dah_observation(key, root)
    model_wave = np.asarray(model_wavelength_angstrom, dtype=np.float64)
    model_fnu = (
        np.asarray(model_flux_lambda, dtype=np.float64) * model_wave**2 * 1.0e-8 / LIGHT_SPEED
    )
    inside = (
        (observation.wavelength_angstrom >= model_wave[0] + 10.0)
        & (observation.wavelength_angstrom <= model_wave[-1] - 10.0)
    )
    wave = observation.wavelength_angstrom[inside]
    flux = observation.flux_nu[inside]
    sigma = observation.sigma_log_wavelength[inside]
    reference_wave, reference_flux = (
        (observation.hardy_wavelength_angstrom, observation.hardy_flux)
        if observation.hardy_flux is not None
        else (model_wave, model_fnu)
    )
    velocity, correlation = _fit_velocity(wave, flux, reference_wave, reference_flux)
    rest = wave / (1.0 + velocity * 1.0e5 / LIGHT_SPEED)
    sampled = convolve_variable_log_gaussian(model_wave, model_fnu, rest, sigma)
    result: dict[str, object] = {
        "target": key,
        "velocity_kms": velocity,
        "velocity_correlation": correlation,
        "velocity_reference": "Hardy curve" if observation.hardy_flux is not None else "model",
        **_scores(rest, flux, sampled),
    }
    if observation.hardy_flux is not None:
        hardy = np.interp(rest, observation.hardy_wavelength_angstrom, observation.hardy_flux)
        result["hardy"] = _scores(rest, flux, hardy)
    return result


def comparison_arrays(
    key: str,
    model_wavelength_angstrom: ArrayLike,
    model_flux_lambda: ArrayLike,
    *,
    velocity_kms: float,
    root: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Rest-frame observation, model and Hardy curve on one scale, for plots."""

    observation = load_dah_observation(key, root)
    model_wave = np.asarray(model_wavelength_angstrom, dtype=np.float64)
    model_fnu = (
        np.asarray(model_flux_lambda, dtype=np.float64) * model_wave**2 * 1.0e-8 / LIGHT_SPEED
    )
    inside = (
        (observation.wavelength_angstrom >= model_wave[0] + 10.0)
        & (observation.wavelength_angstrom <= model_wave[-1] - 10.0)
    )
    wave = observation.wavelength_angstrom[inside]
    flux = observation.flux_nu[inside]
    rest = wave / (1.0 + velocity_kms * 1.0e5 / LIGHT_SPEED)
    model = convolve_variable_log_gaussian(
        model_wave, model_fnu, rest, observation.sigma_log_wavelength[inside]
    )
    control = (rest >= 5_200.0) & (rest <= 6_100.0)
    norm = float(np.median(flux[control]))
    output = {
        "wavelength_angstrom": rest,
        "observed": flux / norm,
        "model": model * float(np.median(flux[control] / model[control])) / norm,
    }
    if observation.hardy_flux is not None:
        hardy = np.interp(rest, observation.hardy_wavelength_angstrom, observation.hardy_flux)
        output["hardy"] = hardy * float(np.median(flux[control] / hardy[control])) / norm
    return output


# ---------------------------------------------------------------------------
# Weak-field SPY/UVES targets (locally normalized Balmer profiles)

DAH_WEAK_LINE_WINDOWS = (
    ("Halpha", 6564.636, 75.0, 18_770.0),
    ("Hbeta", 4862.694, 65.0, 18_770.0),
    ("Hgamma", 4341.691, 55.0, 19_540.0),
    ("Hdelta", 4102.898, 50.0, 19_540.0),
)


@dataclass(frozen=True)
class DAHWeakFieldTarget:
    key: str
    display: str
    effective_temperature: float
    logg: float
    field_megagauss: float
    reference: str
    product_pairs: tuple[tuple[str, str], ...]

    def config_kwargs(self) -> dict[str, object]:
        return dict(
            effective_temperature=self.effective_temperature,
            logg=self.logg,
            magnetic_field_megagauss=self.field_megagauss,
        )


DAH_WEAK_FIELD_TARGETS = (
    DAHWeakFieldTarget(
        "gd9", "WD 0058-044 = GD 9", 16_700.0, 8.07, 0.325,
        "SPY/UVES; Koester et al. (2009) parameters",
        (
            ("ADP.2021-08-19T13:09:35.581", "ADP.2021-08-19T13:09:35.685"),
            ("ADP.2021-09-30T14:35:40.955", "ADP.2020-09-11T05:55:45.875"),
        ),
    ),
    DAHWeakFieldTarget(
        "g76-48", "WD 0257+080 = G 76-48", 6_680.0, 7.96, 0.090,
        "SPY/UVES; Bergeron et al. solution quoted by Koester et al. (2009)",
        (
            ("ADP.2021-08-30T07:37:00.938", "ADP.2021-08-30T07:37:01.090"),
            ("ADP.2021-08-29T15:00:47.253", "ADP.2021-08-29T15:00:47.329"),
        ),
    ),
)
DAH_WEAK_FIELD_BY_KEY = {target.key: target for target in DAH_WEAK_FIELD_TARGETS}


def _weak_epochs(key: str, root: str | Path | None):
    directory = (
        Path(str(files("wd_spectra").joinpath("data/validation/dah")))
        if root is None
        else Path(root)
    )
    with np.load(directory / f"{key}.npz") as saved:
        count = int(saved["n_epochs"])
        return [
            (
                np.asarray(saved[f"epoch{index}_wavelength_angstrom"]),
                np.asarray(saved[f"epoch{index}_flux"]),
                np.asarray(saved[f"epoch{index}_inverse_variance"]),
            )
            for index in range(count)
        ]


def _weak_velocity_score(epoch, velocity, model_wave, convolved) -> float:
    from .observed import local_normalize

    wave_obs, flux, _ = epoch
    rest = wave_obs / (1.0 + velocity * 1.0e5 / LIGHT_SPEED)
    scores = []
    kernel = np.full(11, 1.0 / 11.0)
    for _, center, half, power in DAH_WEAK_LINE_WINDOWS:
        selected = np.abs(rest - center) <= half
        if np.count_nonzero(selected) < 40:
            continue
        wave, observed = local_normalize(rest[selected], flux[selected], center - half, center + half)
        _, model = local_normalize(
            wave, np.interp(wave, model_wave, convolved[power]), center - half, center + half
        )
        observed = np.convolve(observed, kernel, mode="same")
        model = np.convolve(model, kernel, mode="same")
        inner = np.abs(wave - center) <= min(24.0, 0.7 * half)
        inner[: kernel.size] = False
        inner[-kernel.size :] = False
        design = np.column_stack((np.ones(np.count_nonzero(inner)), model[inner] - 1.0))
        coefficients = np.linalg.lstsq(design, observed[inner] - 1.0, rcond=None)[0]
        residual = observed[inner] - 1.0 - design @ coefficients
        scores.append(float(np.mean(residual**2)))
    return float(np.mean(scores))


def score_dah_weak_field_profiles(
    key: str,
    model_wavelength_angstrom: ArrayLike,
    model_flux_lambda: ArrayLike,
    *,
    root: str | Path | None = None,
) -> dict[str, object]:
    """Score locally normalized Halpha--Hdelta against coadded UVES epochs.

    One topocentric velocity per epoch is fitted from all four profiles
    (a depth scale and offset are eliminated for the velocity only); the
    epochs are then coadded with inverse-variance weights and compared with
    the model convolved to the UVES resolving power.
    """

    from .observed import convolve_constant_resolving_power, local_normalize

    model_wave = np.asarray(model_wavelength_angstrom, dtype=np.float64)
    model_flux = np.asarray(model_flux_lambda, dtype=np.float64)
    convolved = {
        power: convolve_constant_resolving_power(
            model_wave, model_flux, power, output_wavelength_angstrom=model_wave
        )
        for power in {item[3] for item in DAH_WEAK_LINE_WINDOWS}
    }
    epochs = _weak_epochs(key, root)
    velocities = []
    for epoch in epochs:
        coarse = np.arange(-200.0, 200.01, 1.0)
        best = coarse[int(np.argmin([_weak_velocity_score(epoch, v, model_wave, convolved) for v in coarse]))]
        fine = np.arange(best - 2.0, best + 2.001, 0.1)
        velocities.append(
            float(fine[int(np.argmin([_weak_velocity_score(epoch, v, model_wave, convolved) for v in fine]))])
        )
    lines: dict[str, dict[str, float]] = {}
    for name, center, half, power in DAH_WEAK_LINE_WINDOWS:
        pieces = []
        for (wave_obs, flux, inverse_variance), velocity in zip(epochs, velocities):
            rest = wave_obs / (1.0 + velocity * 1.0e5 / LIGHT_SPEED)
            selected = (np.abs(rest - center) <= half) & (inverse_variance > 0.0)
            wave, normalized = local_normalize(rest[selected], flux[selected], center - half, center + half)
            continuum = flux[selected] / normalized
            pieces.append((wave, normalized, inverse_variance[selected] * continuum**2))
        lower = max(piece[0][0] for piece in pieces)
        upper = min(piece[0][-1] for piece in pieces)
        step = float(np.median(np.concatenate([np.diff(piece[0]) for piece in pieces])))
        grid = np.arange(lower, upper + 0.25 * step, step)
        numerator = sum(np.interp(grid, w, f) * np.interp(grid, w, q) for w, f, q in pieces)
        weight = sum(np.interp(grid, w, q) for w, _, q in pieces)
        observed = numerator / weight
        _, model = local_normalize(
            grid, np.interp(grid, model_wave, convolved[power]), center - half, center + half
        )
        lines[name] = {
            "rms": float(np.sqrt(np.mean((model - observed) ** 2))),
            "observed_equivalent_width_angstrom": float(-trapezoid(observed - 1.0, grid)),
            "model_equivalent_width_angstrom": float(-trapezoid(model - 1.0, grid)),
        }
    return {
        "target": key,
        "epoch_velocities_kms": velocities,
        "lines": lines,
        "mean_rms": float(np.mean([item["rms"] for item in lines.values()])),
    }


def weak_field_profile_arrays(
    key: str,
    model_wavelength_angstrom: ArrayLike,
    model_flux_lambda: ArrayLike,
    velocities_kms,
    *,
    root: str | Path | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Coadded observed and model normalized profiles per line, for plots."""

    from .observed import convolve_constant_resolving_power, local_normalize

    model_wave = np.asarray(model_wavelength_angstrom, dtype=np.float64)
    model_flux = np.asarray(model_flux_lambda, dtype=np.float64)
    epochs = _weak_epochs(key, root)
    output = {}
    for name, center, half, power in DAH_WEAK_LINE_WINDOWS:
        pieces = []
        for (wave_obs, flux, inverse_variance), velocity in zip(epochs, velocities_kms):
            rest = wave_obs / (1.0 + velocity * 1.0e5 / LIGHT_SPEED)
            selected = (np.abs(rest - center) <= half) & (inverse_variance > 0.0)
            wave, normalized = local_normalize(rest[selected], flux[selected], center - half, center + half)
            continuum = flux[selected] / normalized
            pieces.append((wave, normalized, inverse_variance[selected] * continuum**2))
        lower = max(piece[0][0] for piece in pieces)
        upper = min(piece[0][-1] for piece in pieces)
        step = float(np.median(np.concatenate([np.diff(piece[0]) for piece in pieces])))
        grid = np.arange(lower, upper + 0.25 * step, step)
        observed = sum(np.interp(grid, w, f) * np.interp(grid, w, q) for w, f, q in pieces) / sum(
            np.interp(grid, w, q) for w, _, q in pieces
        )
        model = convolve_constant_resolving_power(model_wave, model_flux, power, output_wavelength_angstrom=grid)
        _, model = local_normalize(grid, model, center - half, center + half)
        output[name] = (grid, observed, model)
    return output


__all__ = [
    "DAHObservation",
    "DAHWeakFieldTarget",
    "DAH_WEAK_FIELD_BY_KEY",
    "DAH_WEAK_FIELD_TARGETS",
    "DAH_WEAK_LINE_WINDOWS",
    "score_dah_weak_field_profiles",
    "weak_field_profile_arrays",
    "DAHValidationTarget",
    "DAH_VALIDATION_BY_KEY",
    "DAH_VALIDATION_TARGETS",
    "comparison_arrays",
    "load_dah_observation",
    "score_dah_spectrum",
]
