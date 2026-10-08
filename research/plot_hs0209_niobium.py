#!/usr/bin/env python3
"""Figure 1 of Williams et al. (2026) with the OpenWD HS 0209+0832 model.

    python research/plot_hs0209_niobium.py --model DIR --raw RAW --output STEM \
        [--paper-model CURVES.npz] [--compare-model DIR2 --compare-label TEXT]

``DIR`` holds ``windows.npz`` from ``research/hs0209_niobium.py``; ``RAW`` holds
the archival HST/STIS E140M HASP coadd and x1d, the CalFUSE ``all`` file and
the STScI E140M line-spread functions.  The model is the paper's fixed
parameters; nothing is fitted.  Its surface flux is scaled by (R/d)^2 with the
paper's R = 0.0145 Rsun and d = 82.6 pc, without reddening or any local
continuum adjustment, Doppler shifted by one velocity per instrument, convolved
with the instrumental profile and averaged over each observed pixel.
``--paper-model`` overplots the published fit as recovered from the vector
graphics of the paper's Figure 1 (research/extract_williams2026_figure1.py).
``--compare-model`` overplots a second ``windows.npz`` (for example the model
without the opt-in line supplements) projected in exactly the same way, with
the velocities fitted to ``DIR``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from astropy.io import fits

C_KMS = 299_792.458
RSUN_CM, PC_CM = 6.957e10, 3.085_677_581_491_367e18
FLUX_SCALE = (0.0145 * RSUN_CM / (82.6 * PC_CM)) ** 2
# Mean photospheric velocity of the elements dominating these windows
# (Ca, Ti, Ni, Cu, Zn: 78.0, 75.8, 76.7, 77.1, 76.4 km/s; Extended Data Table 3).
STIS_VELOCITY_KMS = 76.8
FUSE_RESOLVING_POWER = 20_000.0

# Panel limits (observed frame) and paper Figure 1 identifications at
# vacuum rest wavelengths (Extended Data Table 2; He II multiplet centroid).
PANELS = (
    ("fuse_1003", "FUSE", (1002.5, 1008.0), (
        ("Nb IV", (1002.756, 1005.700, 1007.015), 1),
        ("Ti III", (1004.669,), 2),
    )),
    ("fuse_1050", "FUSE", (1049.0, 1057.0), (
        ("Nb IV", (1049.610, 1050.976, 1054.384, 1055.874), 1),
    )),
    ("stis_1366", "HST", (1365.0, 1370.5), (
        ("Zn IV", (1365.253,), 1), ("Zn III", (1365.706, 1366.968), 1),
        ("Cu IV", (1367.519,), 1, "right"), ("Cu III", (1367.628,), 1, "left"),
        ("Zn IV", (1369.510,), 1),
    )),
    ("stis_1434", "HST", (1433.0, 1436.0), (
        ("Nb IV", (1434.140, 1434.223), 1),
    )),
    ("stis_1452", "HST", (1451.5, 1454.0), (
        ("Nb III", (1451.628,), 1), ("Ni IV", (1452.220,), 1),
        ("Ca III", (1453.161,), 1),
    )),
    ("stis_1640", "HST", (1638.5, 1643.5), (
        ("Zn III", (1639.320,), 1), ("He II", (1640.474,), 2),
        ("Cu III", (1642.202,), 1),
    )),
)


def doppler(velocity_kms: float) -> float:
    beta = velocity_kms / C_KMS
    return float(np.sqrt((1.0 + beta) / (1.0 - beta)))


def pixel_edges(wavelength: np.ndarray) -> np.ndarray:
    middle = 0.5 * (wavelength[1:] + wavelength[:-1])
    return np.concatenate(([2 * wavelength[0] - middle[0]], middle,
                           [2 * wavelength[-1] - middle[-1]]))


def bin_average(fine_wave, fine_flux, edges):
    """Exact average of a linearly interpolated curve over each pixel."""

    grid = np.unique(np.concatenate((fine_wave, edges)))
    grid = grid[(grid >= edges[0]) & (grid <= edges[-1])]
    values = np.interp(grid, fine_wave, fine_flux)
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(grid))))
    at_edges = np.interp(edges, grid, cumulative)
    return np.diff(at_edges) / np.diff(edges)


def convolve_kernel(wave, flux, offsets_angstrom, kernel):
    """Convolve a uniformly sampled model with a tabulated wavelength kernel."""

    step = wave[1] - wave[0]
    half = int(np.ceil(np.max(np.abs(offsets_angstrom)) / step))
    grid = np.arange(-half, half + 1) * step
    weights = np.interp(grid, offsets_angstrom, kernel, left=0.0, right=0.0)
    weights /= weights.sum()
    padded = np.pad(flux, half, mode="edge")
    return np.convolve(padded, weights[::-1], mode="valid")


def stis_data(raw: Path):
    with fits.open(raw / "hst_7473_stis_hs0209p0832_e140m_o55f_cspec.fits") as hdul:
        wave, flux, error = (np.asarray(hdul[1].data[name][0], float)
                             for name in ("WAVELENGTH", "FLUX", "ERROR"))
    good = np.isfinite(flux) & np.isfinite(error) & (error > 0)
    return wave[good], flux[good], error[good]


def stis_native_dispersion(raw: Path, wavelength: float) -> float:
    """Median native E140M pixel width (A) of the order best covering lambda."""

    best = None
    with fits.open(raw / "o55f01010_x1d.fits") as hdul:
        for row in hdul[1].data:
            wave = np.asarray(row["WAVELENGTH"], float)
            if wave[0] < wavelength < wave[-1]:
                margin = min(wavelength - wave[0], wave[-1] - wavelength)
                local = np.median(np.diff(wave)[np.abs(wave[:-1] - wavelength) < 2.0])
                if best is None or margin > best[0]:
                    best = (margin, local)
    return best[1]


def stis_lsf(raw: Path, wavelength: float):
    """STScI 0.2x0.2 LSF, linear in wavelength between 1200 and 1500 A."""

    tables = {key: np.loadtxt(raw / f"LSF_E140M_{key}.txt", skiprows=2) for key in (1200, 1500)}
    weight = float(np.clip((wavelength - 1200.0) / 300.0, 0.0, 1.0))
    pixels = tables[1200][:, 0]
    profile = ((1 - weight) * tables[1200][:, 3]
               + weight * np.interp(pixels, tables[1500][:, 0], tables[1500][:, 3]))
    return pixels, profile


def fuse_coadd(raw: Path, low: float, high: float, alignment_margin: float = 5.0,
               reference_channel: str = "1ALIF"):
    """Align FUSE channels to LiF1A (or ``reference_channel``), scale and coadd.

    Channel zero points differ by up to ~0.1 A.  Each shift maximizes the
    correlation with LiF1A over the window +/- ``alignment_margin``, which
    contains more absorption lines than the plotted window alone.
    """

    with fits.open(raw / "c026020100000all4ttagfcal.fit.gz") as hdul:
        channels = {hdu.name: tuple(np.asarray(hdu.data[c], float) for c in ("WAVE", "FLUX", "ERROR"))
                    for hdu in hdul[1:]}
    reference_wave, reference_flux, reference_error = channels[reference_channel]
    keep = (reference_wave > low) & (reference_wave < high)
    grid = reference_wave[keep]
    used, numerator, denominator = {}, np.zeros_like(grid), np.zeros_like(grid)
    for name, (wave, flux, error) in channels.items():
        good = (error > 0) & np.isfinite(flux) & (flux != 0.0)
        good &= (wave > low - 6.0) & (wave < high + 6.0)
        if np.count_nonzero(good & (wave > low) & (wave < high)) < 0.8 * grid.size:
            continue
        wave, flux, error = wave[good], flux[good], error[good]
        shift, correlation = 0.0, 1.0
        if name != reference_channel:
            wide = ((reference_wave > low - alignment_margin)
                    & (reference_wave < high + alignment_margin) & (reference_error > 0))
            wide &= (reference_wave > wave[0] + 0.2) & (reference_wave < wave[-1] - 0.2)
            trial = np.arange(-0.15, 0.1501, 0.001)
            reference = reference_flux[wide]
            scores = []
            for value in trial:
                sample = np.interp(reference_wave[wide], wave + value, flux)
                scores.append(np.corrcoef(sample - np.median(sample), reference - np.median(reference))[0, 1])
            shift = float(trial[int(np.argmax(scores))])
            correlation = float(np.max(scores))
        sample = np.interp(grid, wave + shift, flux)
        sample_error = np.interp(grid, wave + shift, error)
        scale = float(np.median(reference_flux[keep]) / np.median(sample))
        sample, sample_error = sample * scale, sample_error * scale
        numerator += sample / sample_error**2
        denominator += 1.0 / sample_error**2
        used[name] = {"shift_angstrom": shift, "peak_correlation": correlation,
                      "flux_scale": scale}
    return grid, numerator / denominator, 1.0 / np.sqrt(denominator), used


def model_on_pixels(model, name, observed_wave, velocity_kms, kernel):
    factor = doppler(velocity_kms)
    wave = model[f"{name}_wavelength"] * factor
    curves = {}
    for label in ("full", "no_nb"):
        flux = model[f"{name}_{label}"] * FLUX_SCALE / factor
        smooth = convolve_kernel(wave, flux, *kernel)
        curves[label] = bin_average(wave, smooth, pixel_edges(observed_wave))
    return curves


def fuse_velocity(model, raw):
    """Common FUSE wavelength zero point from the full model, both windows."""

    trial = np.arange(40.0, 110.01, 0.5)
    scores = np.zeros_like(trial)
    for name, _, (low, high), _ in PANELS[:2]:
        wave, flux, error, _ = fuse_coadd(raw, low, high)
        sigma = np.median(wave) / FUSE_RESOLVING_POWER / 2.354_820_045
        offsets = np.linspace(-6 * sigma, 6 * sigma, 241)
        kernel = (offsets, np.exp(-0.5 * (offsets / sigma) ** 2))
        for index, velocity in enumerate(trial):
            curve = model_on_pixels(model, name, wave, velocity, kernel)["full"]
            scale = np.sum(flux * curve / error**2) / np.sum(curve**2 / error**2)
            scores[index] += np.sum(((flux - scale * curve) / error) ** 2)
    return float(trial[np.argmin(scores)]), trial, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-model", type=Path)
    parser.add_argument("--compare-model", type=Path)
    parser.add_argument("--compare-label", default="OpenWD, Stout lines only")
    parser.add_argument("--model-label", default="OpenWD model")
    args = parser.parse_args()
    paper = None if args.paper_model is None else np.load(args.paper_model)
    model = np.load(args.model / "windows.npz")
    compare = None if args.compare_model is None else np.load(args.compare_model / "windows.npz")
    stis = stis_data(args.raw)
    fuse_kms, _, _ = fuse_velocity(model, args.raw)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 13, "axes.labelsize": 15,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 7.8))
    record = {"flux_scale_R_over_d_squared": FLUX_SCALE, "stis_velocity_kms": STIS_VELOCITY_KMS,
              "fuse_velocity_kms": fuse_kms, "model": str(args.model),
              "compare_model": None if compare is None else str(args.compare_model),
              "panels": {}}
    for axis, (name, instrument, (low, high), labels) in zip(axes.flat, PANELS):
        if instrument == "HST":
            keep = (stis[0] > low - 0.3) & (stis[0] < high + 0.3)
            wave, flux, error = (array[keep] for array in stis)
            center = 0.5 * (low + high)
            pixel = stis_native_dispersion(args.raw, center)
            offsets, profile = stis_lsf(args.raw, center)
            kernel = (offsets * pixel, profile)
            velocity = STIS_VELOCITY_KMS
            info = {"native_pixel_angstrom": pixel, "lsf": "STScI E140M 0.2x0.2"}
        else:
            wave, flux, error, channels = fuse_coadd(args.raw, low - 0.3, high + 0.3)
            sigma = np.median(wave) / FUSE_RESOLVING_POWER / 2.354_820_045
            offsets = np.linspace(-6 * sigma, 6 * sigma, 241)
            kernel = (offsets, np.exp(-0.5 * (offsets / sigma) ** 2))
            velocity = fuse_kms
            info = {"channels": channels, "lsf": f"Gaussian R={FUSE_RESOLVING_POWER:.0f}"}
        curves = model_on_pixels(model, name, wave, velocity, kernel)
        shown = (wave >= low) & (wave <= high)
        axis.plot(wave, flux / 1e-12, color="black", lw=0.9, label=f"{instrument} spectrum")
        values = [flux[shown] / 1e-12, curves["full"][shown] / 1e-12]
        if paper is not None:
            paper_wave = paper[f"{name}_wavelength"]
            paper_flux = paper[f"{name}_flux_1e-12"]
            axis.plot(paper_wave, paper_flux, color="#e41a1c", lw=1.2,
                      label="Williams et al. (2026) model")
            values.append(paper_flux[(paper_wave >= low) & (paper_wave <= high)])
        if compare is not None:
            compared = model_on_pixels(compare, name, wave, velocity, kernel)["full"]
            axis.plot(wave, compared / 1e-12, color="#7f7f7f", lw=1.0, ls="--",
                      label=args.compare_label)
            values.append(compared[shown] / 1e-12)
        axis.plot(wave, curves["full"] / 1e-12, color="#2166ac", lw=1.3, label=args.model_label)
        values = np.concatenate(values)
        bottom, top = np.min(values), np.percentile(values, 99.5)
        span = top - bottom
        axis.set_ylim(bottom - 0.08 * span, top + 0.42 * span)
        axis.set_xlim(low, high)
        axis.text(0.025, 0.955, instrument, transform=axis.transAxes, va="top", fontsize=13)
        factor = doppler(velocity)
        tick_top = top + 0.15 * span
        for ion, rests, level, *alignment in labels:
            observed = [rest * factor for rest in rests]
            y = tick_top + (level - 1) * 0.12 * span
            for x in observed:
                axis.plot([x, x], [y - 0.07 * span, y], color="black", lw=1.0)
            if len(observed) > 1:
                axis.plot([min(observed), max(observed)], [y, y], color="black", lw=1.0)
            align = alignment[0] if alignment else "center"
            nudge = {"center": 0.0, "right": 0.03, "left": -0.03}[align]
            axis.text(np.mean(observed) + nudge, y + 0.02 * span, ion, ha=align,
                      va="bottom", fontsize=12)
        axis.ticklabel_format(useOffset=False, axis="x")
        axis.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10], prune="lower"))
        residual = (flux[shown] - curves["full"][shown]) / error[shown]
        info.update({"velocity_kms": velocity, "n_pixels": int(shown.sum()),
                     "chi2_per_pixel_full": float(np.mean(residual**2)),
                     "chi2_per_pixel_no_nb": float(np.mean(
                         ((flux[shown] - curves["no_nb"][shown]) / error[shown]) ** 2)),
                     "median_data_over_model": float(np.median(flux[shown] / curves["full"][shown]))})
        if compare is not None:
            info["chi2_per_pixel_compare"] = float(np.mean(((flux[shown] - compared[shown])
                                                           / error[shown]) ** 2))
        record["panels"][name] = info
    handles, names = axes[0, 2].get_legend_handles_labels()
    handles[0].set_label("Observed (FUSE or HST/STIS)")
    figure.legend(handles, [h.get_label() for h in handles], loc="upper center",
                  ncol=len(handles),
                  frameon=False, fontsize=13, bbox_to_anchor=(0.53, 1.0), handlelength=2.5)
    figure.supxlabel(r"Wavelength (\AA)".replace(r"\AA", "Å"), fontsize=15, y=0.015)
    figure.supylabel(r"Flux ($10^{-12}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=15, x=0.012)
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.085, top=0.935, wspace=0.17, hspace=0.17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".pdf"))
    figure.savefig(args.output.with_suffix(".png"), dpi=250)
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
