#!/usr/bin/env python3
"""Plot the G191-B2B FUSE + STIS comparison for the hot DA/DAO trace-metal model.

    python research/plot_g191b2b_validation.py --model DIR --output PATH.pdf [--extended]

DIR/spectrum.npz is a hot DA/DAO trace-metal model (surface F_lambda, vacuum
wavelengths), e.g. from research/hot_daz_g191b2b.py. The observations are the
bundled MAST HLSP coadds and line lists (wd_spectra/data/hot_daz/observations).
Six 20 A panels show photospheric lines of C, N, O, Al, Si, P and Fe (S IV
and Ni lines appear only in blends). The deep 1334.5 A line is interstellar
C II; the 1335.7 A line, listed as photospheric N III/Ni IV, is shaded as
probable interstellar C II* (see EXTRA_INTERSTELLAR). The model is shifted to the photospheric velocity
(23.8 km/s), convolved with a Gaussian line-spread function (FUSE R = 20000,
STIS E140H/E230H R = 144000) and bin-averaged on the observed pixels exactly as
in research/plot_hot_daz_uv_atlas.py; both are then plotted in the stellar rest
frame. Data and model are normalized independently and identically in each
panel by a quadratic fit to the 88th-percentile envelope in nine bins (the
PG 1424 FUSE procedure). Interstellar lines identified in the HLSP line lists
(origin ISM1/ISM2, Preval et al. 2013) are shaded over +-20 km/s; labelled
photospheric identifications are from the same lists. No parameter is fitted.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from hot_daz_benchmark import read_observation  # noqa: E402
from hot_daz_data import G191B2B_OBSERVATIONS  # noqa: E402
from plot_hot_daz_uv_atlas import sample_segments, VELOCITY  # noqa: E402

plt.rcParams["mathtext.fontset"] = "dejavusans"
C = 299792.458
OBSERVATION = "0.30"
OPENWD = "#d95f02"
ISM_SHADE = "#9ecae1"
ISM_HALF_WIDTH_KMS = 20.0
DATA = {
    "FUSE": ("hlsp_wd-linelist_fuse_spec_g191-b2b_fuv_v1_coadd-spec.fits",
             "hlsp_wd-linelist_fuse_spec_g191-b2b_fuv_v1_linelist.txt", 20000.0),
    "STIS E140H": ("hlsp_wd-linelist_hst_stis_g191-b2b_e140h_v1_coadd-spec.fits",
                   "hlsp_wd-linelist_hst_stis_g191-b2b_e140h_v1_linelist.txt", 144000.0),
    "STIS E230H": ("hlsp_wd-linelist_hst_stis_g191-b2b_e230h_v1_coadd-spec.fits",
                   "hlsp_wd-linelist_hst_stis_g191-b2b_e230h_v1_linelist.txt", 144000.0),
}
# (instrument, lower, upper, photospheric labels: (text, label position, horizontal
# alignment, rest wavelengths of the marked lines)).
PANELS = (
    ("FUSE", 1115.0, 1135.0, (("P V", 1117.98, "center", (1117.98,)), ("N IV", 1122.06, "right", (1122.06,)),
                              ("Si IV", 1122.48, "left", (1122.48,)), ("P V", 1128.01, "right", (1128.01,)),
                              ("Si IV", 1128.34, "left", (1128.34,)))),
    ("STIS E140H", 1165.0, 1185.0, (("C III", 1175.65, "center",
                                     (1174.93, 1175.26, 1175.59, 1175.71, 1175.99, 1176.37)),)),
    ("STIS E140H", 1235.0, 1255.0, (("N V", 1238.82, "center", (1238.82,)), ("N V", 1242.80, "center", (1242.80,)),
                                    ("Fe V", 1244.18, "center", (1244.18,)))),
    ("STIS E140H", 1325.0, 1345.0, (("Fe V", 1330.40, "center", (1330.40,)),
                                    ("O IV", 1338.62, "center", (1338.62,)), ("O IV", 1343.51, "center", (1343.51,)))),
    ("STIS E140H", 1390.0, 1410.0, (("Si IV", 1393.76, "center", (1393.76,)), ("Fe V", 1400.24, "center", (1400.24,)),
                                    ("Fe V", 1402.38, "right", (1402.38,)), ("Si IV", 1402.77, "left", (1402.77,)),
                                    ("Fe V", 1406.96, "center", (1406.67, 1407.25)), ("Fe V", 1409.45, "center", (1409.45,)))),
    ("STIS E230H", 1850.0, 1870.0, (("Al III", 1854.72, "center", (1854.72,)), ("Al III", 1862.79, "center", (1862.79,)))),
)

# Optional extra window (--extended): S IV.
EXTRA_PANELS = (
    ("FUSE", 1060.0, 1080.0, (("S IV", 1062.66, "center", (1062.66,)), ("Si IV", 1066.63, "center", (1066.63,)),
                              ("S IV", 1072.97, "center", (1072.97,)))),
)
# Observed wavelengths shaded as interstellar beyond the HLSP ISM1/ISM2 lines.
# 1335.741: listed as photospheric N III 1335.64 (lab wavelength +-0.11 A) /
# Ni IV 1335.62, but it lies within ~1 km/s of interstellar C II* 1335.708 at
# the Hyades-cloud velocity (+8.0 km/s from C II* 1037.02 in the same lists);
# the atomic data contain no N III line near 1335.6 A.
EXTRA_INTERSTELLAR = {"STIS E140H": (1335.741,)}

def to_rest(wavelength):
    return wavelength / (1.0 + VELOCITY / C)


def normalize(wavelength, flux, lower, upper):
    """Quadratic fit to the 88th-percentile envelope in nine bins (PG 1424 FUSE panels)."""
    edges = np.linspace(lower, upper, 10)
    centers, envelope = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        keep = (wavelength >= left) & (wavelength < right) & np.isfinite(flux) & (flux > 0)
        if np.count_nonzero(keep) >= 5:
            centers.append(0.5 * (left + right))
            envelope.append(np.nanpercentile(flux[keep], 88.0))
    fit = np.polynomial.Chebyshev.fit(centers, envelope, min(2, len(centers) - 1), domain=(lower, upper))
    reference = float(np.nanmedian(envelope))
    return flux / np.clip(fit(wavelength), 0.55 * reference, 1.65 * reference)


def interstellar(linelist):
    """Observed wavelengths of lines the HLSP list attributes to ISM1/ISM2."""
    lines = []
    for text in Path(linelist).read_text().splitlines():
        if text.startswith("#") or not text.strip():
            continue
        fields = re.findall(r'"[^"]*"|\S+', text)
        if fields[11].strip('"') != "PHOT":
            lines.append(float(fields[0]))
    return np.asarray(lines)


def make_figure(model_directory, output_path, panels=PANELS):
    with np.load(Path(model_directory) / "spectrum.npz") as saved:
        model_wave, model_flux = saved["wavelength"].copy(), saved["flux"].copy()
    spectra = {}
    for instrument, (spectrum, linelist, resolving_power) in DATA.items():
        wave, flux, _ = read_observation(G191B2B_OBSERVATIONS / spectrum)
        # Only the plotted windows (plus a margin for the line-spread function).
        rest = to_rest(wave)
        good = np.isfinite(flux) & np.any([(rest > lower - 2.0) & (rest < upper + 2.0)
                                           for name, lower, upper, _ in panels if name == instrument]
                                          + [np.zeros(wave.size, bool)], axis=0)
        wave, flux = wave[good], flux[good]
        if wave.size == 0:
            continue
        spectra[instrument] = (wave, flux, sample_segments(model_wave, model_flux, wave, resolving_power),
                               np.concatenate((interstellar(G191B2B_OBSERVATIONS / linelist),
                                               EXTRA_INTERSTELLAR.get(instrument, ()))))
    figure, axes = plt.subplots(len(panels), 1, figsize=(8.0, 1.25 + 1.27 * len(panels)), gridspec_kw={"hspace": 0.36})
    for axis, (instrument, lower, upper, labels) in zip(axes, panels):
        wave, data, model, ism = spectra[instrument]
        rest = to_rest(wave)
        keep = (rest >= lower) & (rest <= upper) & np.isfinite(model)
        w = rest[keep]
        observed = normalize(w, data[keep], lower, upper)
        predicted = normalize(w, model[keep], lower, upper)
        for line in to_rest(ism):
            if lower < line < upper:
                half = line * ISM_HALF_WIDTH_KMS / C
                axis.axvspan(line - half, line + half, color=ISM_SHADE, alpha=0.45, lw=0, zorder=0)
        axis.plot(w, observed, color=OBSERVATION, lw=0.45, rasterized=True)
        axis.plot(w, predicted, color=OPENWD, lw=0.65)
        for text, position, alignment, centers in labels:
            for center in centers:
                axis.plot((center, center), (1.08, 1.15), color="0.25", lw=0.6)
            offset = {"left": -0.04, "right": 0.04, "center": 0.0}[alignment]
            axis.text(position + offset, 1.17, text, ha=alignment, va="bottom", fontsize=7.5, color="0.15")
        axis.set_xlim(lower, upper)
        axis.set_ylim(0.15, 1.32)
        axis.set_yticks((0.4, 0.7, 1.0))
        axis.tick_params(axis="both", which="both", direction="in", top=True, right=True,
                         labelsize=9, length=4)
        axis.text(0.012, 0.06, instrument, transform=axis.transAxes, ha="left", va="bottom",
                  fontsize=8.5, color="0.25", zorder=5)
    axes[0].set_title(r"G191$-$B2B   $T_{\rm eff} = 52500\,$K, $\log g = 7.53$, He/H $= 10^{-5}$;"
                      " C, N, O, Al, Si, P, S, Fe, Ni (Preval et al. 2013)", fontsize=9.5, loc="left")
    axes[-1].legend(
        handles=(Line2D([], [], color=OBSERVATION, lw=0.8, label="Observation"),
                 Line2D([], [], color=OPENWD, lw=0.8, label="OpenWD"),
                 Patch(facecolor=ISM_SHADE, alpha=0.45, lw=0, label="ISM absorption")),
        loc="lower right", frameon=False, fontsize=8.5, ncol=3, handlelength=2.0, columnspacing=1.2,
    )
    axes[-1].set_xlabel(r"vacuum rest wavelength [$\mathrm{\AA}$]", fontsize=11)
    figure.supylabel("normalized flux", fontsize=11, x=0.015)
    figure.subplots_adjust(left=0.105, right=0.965, bottom=0.07, top=0.955)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    figure.savefig(output_path.with_suffix(".png"), dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extended", action="store_true",
                        help="also show the S IV 1063/1073 and N V 1239/1243 windows")
    args = parser.parse_args()
    panels = (tuple(sorted(PANELS + EXTRA_PANELS, key=lambda panel: panel[1])) if args.extended
              else PANELS)
    make_figure(args.model, args.output, panels)
    print(args.output)


if __name__ == "__main__":
    main()
