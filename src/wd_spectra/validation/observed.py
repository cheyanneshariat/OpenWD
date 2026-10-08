"""Observed-spectrum readers and comparison utilities.

The routines in this module keep survey conventions out of the atmosphere
solver.  In particular, SPY/UVES spectra are compared as locally normalized
line profiles: the SPY echelle data were not spectrophotometrically calibrated
and their broad order shapes are not an independent SED measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from .._compat import trapezoid
from numpy.typing import ArrayLike, NDArray

from ..atmosphere import Atmosphere
from ..constants import LIGHT_SPEED
from ..eos import hummer_mihalas_helium_lte, hummer_mihalas_hydrogen_lte
from .koester import physical_air_to_vacuum


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ObservedSpectrum:
    """One observed spectrum with an explicit wavelength/flux convention."""

    wavelength_angstrom: FloatArray
    flux_lambda: FloatArray
    inverse_variance: FloatArray | None
    source_path: Path
    survey: str
    wavelength_medium: Literal["vacuum"] = "vacuum"
    resolution_sigma_log_wavelength: FloatArray | None = None
    source_paths: tuple[Path, ...] = ()
    barycentric_correction_kms: tuple[float, ...] = ()


def _validate_observed_arrays(
    wavelength: ArrayLike,
    flux: ArrayLike,
    inverse_variance: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray | None]:
    wave = np.asarray(wavelength, dtype=np.float64)
    values = np.asarray(flux, dtype=np.float64)
    if wave.ndim != 1 or values.shape != wave.shape:
        raise ValueError("observed wavelength and flux must be matching 1D arrays")
    weight = (
        None
        if inverse_variance is None
        else np.asarray(inverse_variance, dtype=np.float64)
    )
    if weight is not None and weight.shape != wave.shape:
        raise ValueError("inverse variance must match the wavelength grid")
    valid = np.isfinite(wave) & np.isfinite(values) & (wave > 0.0)
    if weight is not None:
        valid &= np.isfinite(weight) & (weight >= 0.0)
    wave = wave[valid]
    values = values[valid]
    weight = None if weight is None else weight[valid]
    order = np.argsort(wave)
    wave = wave[order]
    values = values[order]
    weight = None if weight is None else weight[order]
    unique = np.concatenate(([True], np.diff(wave) > 0.0))
    if np.count_nonzero(unique) < 2:
        raise ValueError("observed spectrum has fewer than two unique wavelengths")
    return wave[unique], values[unique], None if weight is None else weight[unique]


def read_mwdd_ascii_spectrum(
    path: str | Path,
    *,
    survey: str,
    wavelength_medium: Literal["air", "vacuum"],
    flux_convention: Literal["flambda", "fnu"] = "flambda",
) -> ObservedSpectrum:
    """Read a two-column public MWDD mirror spectrum.

    MWDD is used only as a file mirror by the validation scripts.  ``survey``
    records the actual provenance from the file header, such as SPY/UVES.
    """

    source = Path(path)
    data = np.genfromtxt(source, delimiter=",", skip_header=2)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"could not read a two-column spectrum from {source}")
    wavelength, flux, _ = _validate_observed_arrays(data[:, 0], data[:, 1])
    if wavelength_medium == "air":
        wavelength = physical_air_to_vacuum(wavelength)
    elif wavelength_medium != "vacuum":
        raise ValueError("wavelength_medium must be 'air' or 'vacuum'")
    if flux_convention == "fnu":
        flux = flux * LIGHT_SPEED / (wavelength * 1.0e-8) ** 2 * 1.0e-8
    elif flux_convention != "flambda":
        raise ValueError("flux_convention must be 'flambda' or 'fnu'")
    return ObservedSpectrum(
        wavelength, flux, None, source, survey, source_paths=(source,)
    )


def read_sdss_spectrum(path: str | Path) -> ObservedSpectrum:
    """Read an SDSS ``spec-PLATE-MJD-FIBER.fits`` spectrum.

    Astropy is an optional validation dependency and is imported lazily so
    that the core synthesis package remains NumPy-only.
    """

    try:
        from astropy.io import fits
    except ImportError as error:  # pragma: no cover - depends on optional env
        raise ImportError("reading SDSS FITS spectra requires astropy") from error

    source = Path(path)
    with fits.open(source, memmap=False) as hdus:
        table = hdus[1].data
        wavelength = 10.0 ** np.asarray(table["loglam"], dtype=np.float64)
        flux = np.asarray(table["flux"], dtype=np.float64)
        inverse_variance = np.asarray(table["ivar"], dtype=np.float64)
        # SDSS ``and_mask`` marks pixels carrying a reduction defect in every
        # contributing exposure.  Their ivar can nevertheless remain positive
        # (the 5577-A BRIGHTSKY residual is a common example), so relying on
        # ivar alone admits large artificial spikes into validation metrics.
        if "and_mask" in (table.names or ()):
            inverse_variance = np.where(
                np.asarray(table["and_mask"]) == 0,
                inverse_variance,
                0.0,
            )
        # SDSS wdisp is the instrumental Gaussian sigma in native log-lambda
        # pixels.  BOSS spectra have Delta log10(lambda)=1e-4 per pixel.
        resolution_sigma_log = (
            np.asarray(table["wdisp"], dtype=np.float64)
            * np.log(10.0)
            * 1.0e-4
        )
    valid = (
        np.isfinite(wavelength)
        & np.isfinite(flux)
        & np.isfinite(inverse_variance)
        & np.isfinite(resolution_sigma_log)
        & (wavelength > 0.0)
        & (inverse_variance >= 0.0)
        & (resolution_sigma_log > 0.0)
    )
    order = np.argsort(wavelength[valid])
    wavelength = wavelength[valid][order]
    flux = flux[valid][order]
    inverse_variance = inverse_variance[valid][order]
    resolution_sigma_log = resolution_sigma_log[valid][order]
    return ObservedSpectrum(
        wavelength,
        flux,
        inverse_variance,
        source,
        "SDSS DR17/BOSS",
        resolution_sigma_log_wavelength=resolution_sigma_log,
        source_paths=(source,),
    )


def read_hst_hasp_spectrum(
    path: str | Path,
    *,
    survey: str = "HST/STIS HASP",
) -> ObservedSpectrum:
    """Read a coadded Hubble Advanced Spectral Product.

    HASP ``*_cspec.fits`` products store one vacuum-wavelength spectrum in
    the ``SCI`` vector table.  Keeping this adapter here makes the archive
    convention explicit and avoids target-specific FITS parsing.
    """

    try:
        from astropy.io import fits
    except ImportError as error:  # pragma: no cover - depends on optional env
        raise ImportError("reading HST HASP spectra requires astropy") from error

    source = Path(path)
    with fits.open(source, memmap=False) as hdus:
        if "SCI" not in hdus:
            raise ValueError(f"{source} has no SCI extension")
        table = hdus["SCI"].data
        names = set(table.names or ())
        required = {"WAVELENGTH", "FLUX", "ERROR"}
        if not required.issubset(names):
            raise ValueError(
                f"{source} SCI extension is missing {sorted(required - names)}"
            )
        wavelength = np.asarray(table["WAVELENGTH"], dtype=np.float64).reshape(-1)
        flux = np.asarray(table["FLUX"], dtype=np.float64).reshape(-1)
        error = np.asarray(table["ERROR"], dtype=np.float64).reshape(-1)
    valid_error = np.isfinite(error) & (error > 0.0)
    inverse_variance = np.zeros_like(error)
    inverse_variance[valid_error] = error[valid_error] ** -2
    wavelength, flux, inverse_variance = _validate_observed_arrays(
        wavelength, flux, inverse_variance
    )
    return ObservedSpectrum(
        wavelength,
        flux,
        inverse_variance,
        source,
        survey,
        source_paths=(source,),
    )


def read_eso_phase3_spectrum(
    paths: Iterable[str | Path],
    *,
    survey: str = "SPY/UVES (ESO Phase 3)",
    coadd_overlapping_products: bool = False,
    apply_barycentric_correction: bool = False,
) -> ObservedSpectrum:
    """Read and merge ESO Phase-3 one-dimensional spectral products.

    The in-house reprocessed UVES and X-shooter products store each
    spectrograph arm as a one-row vector table.  ``WAVE`` is explicitly
    documented as air wavelength in the column metadata; this reader verifies
    that convention, converts its declared FITS unit to Angstrom before the
    air-to-vacuum conversion, discards zero-filled edge pixels, and propagates
    ``ERR`` to inverse variance.  Reading the unit is essential because the
    public UVES products use Angstrom whereas X-shooter products use nm.
    Multiple arms may be supplied in any order.  If
    ``apply_barycentric_correction`` is true, the ESO
    ``QC VRAD BARYCOR`` header velocity is applied to every exposure before
    merging.  This is necessary when the declared ``SPECSYS`` is
    ``TOPOCENT`` and prevents repeat exposures obtained on different dates
    from smearing narrow lines.  When
    ``coadd_overlapping_products`` is true, products with essentially the same
    wavelength coverage are treated as repeat exposures and combined on the
    densest exposure grid with inverse-variance weights after a robust scalar
    flux rescaling.  This reproduces the X-shooter reduction strategy used by
    Izquierdo et al. (2023) without confusing adjacent arms with repeats.
    """

    try:
        from astropy.io import fits
        from astropy import units as u
    except ImportError as error:  # pragma: no cover - depends on optional env
        raise ImportError("reading ESO Phase-3 FITS spectra requires astropy") from error

    sources = tuple(Path(path) for path in paths)
    if not sources:
        raise ValueError("at least one ESO Phase-3 spectrum is required")
    wavelength_parts: list[FloatArray] = []
    flux_parts: list[FloatArray] = []
    inverse_variance_parts: list[FloatArray] = []
    coverage_parts: list[tuple[float, float]] = []
    barycentric_corrections: list[float] = []
    for source in sources:
        with fits.open(source, memmap=False) as hdus:
            if "SPECTRUM" not in hdus:
                raise ValueError(f"{source} has no SPECTRUM extension")
            spectrum_hdu = hdus["SPECTRUM"]
            table = spectrum_hdu.data
            names = set(table.names or ())
            if {"FLUX", "ERR"}.issubset(names):
                flux_name, error_name = "FLUX", "ERR"
            elif {"FLUX_REDUCED", "ERR_REDUCED"}.issubset(names):
                # Some otherwise science-grade UVES products could not be
                # flux calibrated and retain count-density columns only.
                # Their local normalized profiles remain valid.
                flux_name, error_name = "FLUX_REDUCED", "ERR_REDUCED"
            else:
                raise ValueError(
                    f"{source} has neither calibrated nor reduced flux/error columns"
                )
            if "WAVE" not in names:
                raise ValueError(f"{source} is missing the ESO Phase-3 WAVE column")
            wave_index = list(table.names).index("WAVE") + 1
            comment = str(spectrum_hdu.header.comments[f"TUCD{wave_index}"])
            if "air wavelength" not in comment.lower():
                raise ValueError(
                    f"{source} does not explicitly identify WAVE as air wavelength"
                )
            wavelength_unit_string = spectrum_hdu.columns["WAVE"].unit or "angstrom"
            try:
                wavelength_scale = u.Unit(wavelength_unit_string).to(u.AA)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{source} has unsupported WAVE unit {wavelength_unit_string!r}"
                ) from error
            wavelength_air = (
                np.asarray(table["WAVE"], dtype=np.float64).reshape(-1)
                * wavelength_scale
            )
            correction = spectrum_hdu.header.get("ESO QC VRAD BARYCOR")
            if correction is None:
                correction = hdus[0].header.get("ESO QC VRAD BARYCOR")
            if apply_barycentric_correction:
                if correction is None or not np.isfinite(float(correction)):
                    raise ValueError(
                        f"{source} lacks a finite ESO QC VRAD BARYCOR header"
                    )
                wavelength_air *= 1.0 + float(correction) * 1.0e5 / LIGHT_SPEED
            barycentric_corrections.append(
                float(correction) if correction is not None else float("nan")
            )
            coverage = (
                float(np.nanmin(wavelength_air)),
                float(np.nanmax(wavelength_air)),
            )
            flux = np.asarray(table[flux_name], dtype=np.float64).reshape(-1)
            error = np.asarray(table[error_name], dtype=np.float64).reshape(-1)
            if flux_name == "FLUX":
                target_flux_unit = u.erg / (u.cm**2 * u.s * u.AA)
                flux_unit_string = spectrum_hdu.columns[flux_name].unit
                error_unit_string = spectrum_hdu.columns[error_name].unit
                if flux_unit_string:
                    try:
                        flux *= u.Unit(flux_unit_string).to(target_flux_unit)
                    except (TypeError, ValueError) as unit_error:
                        raise ValueError(
                            f"{source} has unsupported FLUX unit {flux_unit_string!r}"
                        ) from unit_error
                if error_unit_string:
                    try:
                        error *= u.Unit(error_unit_string).to(target_flux_unit)
                    except (TypeError, ValueError) as unit_error:
                        raise ValueError(
                            f"{source} has unsupported ERR unit {error_unit_string!r}"
                        ) from unit_error
            valid = (
                np.isfinite(wavelength_air)
                & np.isfinite(flux)
                & np.isfinite(error)
                & (wavelength_air > 0.0)
                & (flux > 0.0)
                & (error > 0.0)
            )
            if "QUAL" in names:
                valid &= np.asarray(table["QUAL"]).reshape(-1) == 0
            wavelength_parts.append(physical_air_to_vacuum(wavelength_air[valid]))
            flux_parts.append(flux[valid])
            inverse_variance_parts.append(error[valid] ** -2)
            coverage_parts.append(coverage)

    if coadd_overlapping_products:
        grouped: list[list[int]] = []
        for index, coverage in enumerate(coverage_parts):
            for group in grouped:
                reference = coverage_parts[group[0]]
                scale = max(reference[1] - reference[0], 1.0)
                same_coverage = (
                    abs(coverage[0] - reference[0]) < 0.01 * scale
                    and abs(coverage[1] - reference[1]) < 0.01 * scale
                )
                if same_coverage:
                    group.append(index)
                    break
            else:
                grouped.append([index])

        coadded_wavelength: list[FloatArray] = []
        coadded_flux: list[FloatArray] = []
        coadded_inverse_variance: list[FloatArray] = []
        for group in grouped:
            if len(group) == 1:
                index = group[0]
                coadded_wavelength.append(wavelength_parts[index])
                coadded_flux.append(flux_parts[index])
                coadded_inverse_variance.append(inverse_variance_parts[index])
                continue
            reference_index = max(
                group, key=lambda candidate: wavelength_parts[candidate].size
            )
            reference_wavelength = wavelength_parts[reference_index]
            reference_flux = flux_parts[reference_index]
            sampled_flux = np.full(
                (len(group), reference_wavelength.size), np.nan, dtype=np.float64
            )
            sampled_weight = np.zeros_like(sampled_flux)
            for row, index in enumerate(group):
                wavelength = wavelength_parts[index]
                flux = flux_parts[index]
                weight = inverse_variance_parts[index]
                inside = (
                    (reference_wavelength >= wavelength[0])
                    & (reference_wavelength <= wavelength[-1])
                )
                if wavelength.size > 1:
                    nearest_upper = np.clip(
                        np.searchsorted(wavelength, reference_wavelength),
                        1,
                        wavelength.size - 1,
                    )
                    nearest_distance = np.minimum(
                        abs(reference_wavelength - wavelength[nearest_upper - 1]),
                        abs(reference_wavelength - wavelength[nearest_upper]),
                    )
                    inside &= nearest_distance <= 0.75 * np.percentile(
                        np.diff(wavelength), 25.0
                    )
                sampled_flux[row, inside] = np.interp(
                    reference_wavelength[inside], wavelength, flux
                )
                sampled_weight[row, inside] = np.interp(
                    reference_wavelength[inside], wavelength, weight
                )
                common = (
                    inside
                    & np.isfinite(sampled_flux[row])
                    & (sampled_flux[row] > 0.0)
                    & np.isfinite(reference_flux)
                    & (reference_flux > 0.0)
                )
                if np.count_nonzero(common) > 20:
                    ratio = reference_flux[common] / sampled_flux[row, common]
                    lower, upper = np.percentile(ratio, (20.0, 80.0))
                    central = ratio[(ratio >= lower) & (ratio <= upper)]
                    rescale = float(np.median(central))
                    sampled_flux[row] *= rescale
                    sampled_weight[row] /= rescale**2

            # Reject isolated cosmic rays before the inverse-variance mean.
            if len(group) >= 3:
                median = np.nanmedian(sampled_flux, axis=0)
                absolute_deviation = np.abs(sampled_flux - median[np.newaxis, :])
                mad = 1.4826 * np.nanmedian(absolute_deviation, axis=0)
                reported_sigma = np.full_like(sampled_weight, np.inf)
                np.sqrt(
                    np.divide(
                        1.0,
                        sampled_weight,
                        out=reported_sigma,
                        where=sampled_weight > 0.0,
                    ),
                    out=reported_sigma,
                    where=sampled_weight > 0.0,
                )
                threshold = 7.0 * np.maximum(
                    mad[np.newaxis, :], reported_sigma
                )
                sampled_weight[absolute_deviation > threshold] = 0.0
            finite = np.isfinite(sampled_flux) & (sampled_weight > 0.0)
            sampled_weight = np.where(finite, sampled_weight, 0.0)
            sampled_flux = np.where(finite, sampled_flux, 0.0)
            total_weight = np.sum(sampled_weight, axis=0)
            valid = total_weight > 0.0
            coadded_wavelength.append(reference_wavelength[valid])
            coadded_flux.append(
                np.sum(sampled_weight * sampled_flux, axis=0)[valid]
                / total_weight[valid]
            )
            coadded_inverse_variance.append(total_weight[valid])
        wavelength_parts = coadded_wavelength
        flux_parts = coadded_flux
        inverse_variance_parts = coadded_inverse_variance

    wavelength, flux, inverse_variance = _validate_observed_arrays(
        np.concatenate(wavelength_parts),
        np.concatenate(flux_parts),
        np.concatenate(inverse_variance_parts),
    )
    return ObservedSpectrum(
        wavelength,
        flux,
        inverse_variance,
        sources[0],
        survey,
        source_paths=sources,
        barycentric_correction_kms=tuple(barycentric_corrections),
    )


def gaussian_smooth(values: ArrayLike, sigma_pixels: float) -> FloatArray:
    """Convolve a sampled series with a normalized Gaussian kernel."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.isfinite(sigma_pixels) or sigma_pixels < 0.0:
        raise ValueError("sigma_pixels must be finite and non-negative")
    if sigma_pixels <= 0.05:
        return array.copy()
    radius = max(1, int(np.ceil(4.0 * sigma_pixels)))
    offset = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offset / sigma_pixels) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(array, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="same")[radius:-radius]


def convolve_constant_resolving_power(
    wavelength_angstrom: ArrayLike,
    flux: ArrayLike,
    resolving_power: float,
    *,
    output_wavelength_angstrom: ArrayLike | None = None,
) -> FloatArray:
    """Convolve a spectrum with a Gaussian constant-``lambda/dlambda`` LSF."""

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    values = np.asarray(flux, dtype=np.float64)
    if wavelength.ndim != 1 or values.shape != wavelength.shape:
        raise ValueError("wavelength and flux must be matching 1D arrays")
    if np.any(np.diff(wavelength) <= 0.0) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength must be positive and strictly increasing")
    if not np.isfinite(resolving_power) or resolving_power <= 0.0:
        raise ValueError("resolving_power must be finite and positive")
    log_wavelength = np.log(wavelength)
    log_step = float(np.min(np.diff(log_wavelength)))
    uniform_log = np.arange(
        log_wavelength[0], log_wavelength[-1] + 0.5 * log_step, log_step
    )
    uniform_flux = np.interp(uniform_log, log_wavelength, values)
    sigma_pixels = 1.0 / (2.354_820_045 * resolving_power * log_step)
    smoothed = gaussian_smooth(uniform_flux, sigma_pixels)
    output = (
        wavelength
        if output_wavelength_angstrom is None
        else np.asarray(output_wavelength_angstrom, dtype=np.float64)
    )
    return np.interp(np.log(output), uniform_log, smoothed)


def convolve_variable_log_gaussian(
    wavelength_angstrom: ArrayLike,
    flux: ArrayLike,
    output_wavelength_angstrom: ArrayLike,
    sigma_log_wavelength: ArrayLike,
) -> FloatArray:
    """Apply a wavelength-dependent Gaussian LSF in log wavelength.

    ``sigma_log_wavelength`` is the Gaussian sigma in natural-log wavelength
    at every output pixel.  This is the native convention obtained from the
    SDSS ``wdisp`` vector after multiplication by ``ln(10) * 1e-4``.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    values = np.asarray(flux, dtype=np.float64)
    output = np.asarray(output_wavelength_angstrom, dtype=np.float64)
    sigma = np.asarray(sigma_log_wavelength, dtype=np.float64)
    if wavelength.ndim != 1 or values.shape != wavelength.shape:
        raise ValueError("wavelength and flux must be matching 1D arrays")
    if output.ndim != 1 or sigma.shape != output.shape:
        raise ValueError("output wavelength and sigma must be matching 1D arrays")
    if np.any(np.diff(wavelength) <= 0.0) or np.any(wavelength <= 0.0):
        raise ValueError("wavelength must be positive and strictly increasing")
    if np.any(output <= 0.0) or np.any(~np.isfinite(output)):
        raise ValueError("output wavelength must be finite and positive")
    if np.any(sigma <= 0.0) or np.any(~np.isfinite(sigma)):
        raise ValueError("sigma_log_wavelength must be finite and positive")

    source_log = np.log(wavelength)
    # Composite diagnostic grids can contain distinct floating-point
    # wavelengths that collapse to the same logarithm (for example where a
    # 0.05-A line-core grid overlaps a 0.5-A broad grid).  Taking the literal
    # minimum spacing would then request an infinite or petabyte-sized
    # uniform grid.  Merge only numerically coincident samples; 1e-10 in
    # ln(lambda) is orders of magnitude finer than any supported LSF.
    separated = np.concatenate((
        np.asarray((True,)),
        np.diff(source_log) > 1.0e-10,
    ))
    if not np.all(separated):
        source_log = source_log[separated]
        values = values[separated]
    step = float(np.min(np.diff(source_log)))
    uniform_log = np.arange(source_log[0], source_log[-1] + 0.5 * step, step)
    uniform_flux = np.interp(uniform_log, source_log, values)
    output_log = np.log(output)
    convolved = np.empty_like(output)
    for index, (center, local_sigma) in enumerate(zip(output_log, sigma)):
        lower = int(np.searchsorted(uniform_log, center - 4.0 * local_sigma))
        upper = int(np.searchsorted(uniform_log, center + 4.0 * local_sigma))
        lower = max(0, lower)
        upper = min(uniform_log.size, max(lower + 1, upper))
        distance = (uniform_log[lower:upper] - center) / local_sigma
        weight = np.exp(-0.5 * distance**2)
        convolved[index] = np.sum(weight * uniform_flux[lower:upper]) / np.sum(weight)
    return convolved


def local_normalize(
    wavelength_angstrom: ArrayLike,
    flux: ArrayLike,
    lower: float,
    upper: float,
    *,
    continuum_fraction: float = 0.16,
) -> tuple[FloatArray, FloatArray]:
    """Remove a robust local log-linear continuum around a line window."""

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    values = np.asarray(flux, dtype=np.float64)
    selected = (
        (wavelength >= lower)
        & (wavelength <= upper)
        & np.isfinite(values)
        & (values > 0.0)
    )
    wave = wavelength[selected]
    local = values[selected]
    if wave.size < 5:
        return wave, local / np.nanmedian(local)
    coordinate = 2.0 * (wave - lower) / (upper - lower) - 1.0
    side = (coordinate <= -1.0 + 2.0 * continuum_fraction) | (
        coordinate >= 1.0 - 2.0 * continuum_fraction
    )
    logarithm = np.log(local)
    fit = np.polyfit(coordinate[side], logarithm[side], 1)
    for _ in range(3):
        residual = logarithm - np.polyval(fit, coordinate)
        center = np.median(residual[side])
        scatter = 1.4826 * np.median(np.abs(residual[side] - center))
        if not np.isfinite(scatter) or scatter <= 0.0:
            break
        # Reject absorption and rare positive artifacts without repeatedly
        # selecting the top of a noisy continuum.  The older percentile loop
        # biased low-S/N SPY continua upward by as much as ten percent.
        retained = side & (residual >= center - 2.5 * scatter) & (
            residual <= center + 4.0 * scatter
        )
        if np.count_nonzero(retained) >= 3:
            side = retained
            fit = np.polyfit(coordinate[side], logarithm[side], 1)
    return wave, local / np.exp(np.polyval(fit, coordinate))


def normalized_profile_metrics(
    observed_wavelength: ArrayLike,
    observed_flux: ArrayLike,
    model_wavelength: ArrayLike,
    model_flux: ArrayLike,
    lower: float,
    upper: float,
    observed_inverse_variance: ArrayLike | None = None,
) -> dict[str, float]:
    """Measure normalized RMS, equivalent widths, and core depths.

    When delivered inverse variance is available, the result also includes a
    continuum-normalized weighted RMS and a reduced-chi-square-like mean.  The
    latter is intentionally labeled as a diagnostic because continuum fitting
    and oversampling introduce covariance between neighboring pixels.
    """

    observed_wave, observed = local_normalize(
        observed_wavelength, observed_flux, lower, upper
    )
    model_wave, model = local_normalize(model_wavelength, model_flux, lower, upper)
    if observed_wave.size < 3 or model_wave.size < 3:
        raise ValueError("line window contains too few valid samples")
    sampled_model = np.interp(observed_wave, model_wave, model)
    residual = sampled_model - observed
    adjacent_difference = np.diff(observed)
    if adjacent_difference.size:
        difference_center = np.median(adjacent_difference)
        observed_noise = (
            1.4826
            * np.median(np.abs(adjacent_difference - difference_center))
            / np.sqrt(2.0)
        )
    else:
        observed_noise = 0.0
    rms = float(np.sqrt(np.mean(residual**2)))
    result = {
        "normalized_rms": rms,
        "observed_noise_estimate": float(observed_noise),
        "noise_corrected_rms": float(
            np.sqrt(max(rms**2 - observed_noise**2, 0.0))
        ),
        "mean_residual": float(np.mean(residual)),
        "observed_minimum": float(np.min(observed)),
        "model_minimum": float(np.min(model)),
        "observed_equivalent_width": float(
            trapezoid(1.0 - observed, observed_wave)
        ),
        "model_equivalent_width": float(trapezoid(1.0 - model, model_wave)),
    }
    if observed_inverse_variance is not None:
        original_wave = np.asarray(observed_wavelength, dtype=np.float64)
        original_flux = np.asarray(observed_flux, dtype=np.float64)
        inverse_variance = np.asarray(
            observed_inverse_variance, dtype=np.float64
        )
        if inverse_variance.shape != original_wave.shape:
            raise ValueError("observed inverse variance must match wavelength")
        selected = (
            (original_wave >= lower)
            & (original_wave <= upper)
            & np.isfinite(original_flux)
            & (original_flux > 0.0)
        )
        selected_inverse_variance = inverse_variance[selected]
        continuum = original_flux[selected] / observed
        normalized_inverse_variance = selected_inverse_variance * continuum**2
        usable = (
            np.isfinite(normalized_inverse_variance)
            & (normalized_inverse_variance > 0.0)
            & np.isfinite(residual)
        )
        if np.any(usable):
            weights = normalized_inverse_variance[usable]
            result.update(
                {
                    "reported_noise_rms": float(
                        np.sqrt(np.mean(1.0 / weights))
                    ),
                    "inverse_variance_weighted_rms": float(
                        np.sqrt(np.sum(weights * residual[usable] ** 2) / np.sum(weights))
                    ),
                    "mean_chi_square": float(
                        np.mean(weights * residual[usable] ** 2)
                    ),
                }
            )
    return result
