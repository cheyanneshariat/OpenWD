#!/usr/bin/env python3
"""HS 0209+0832 line windows in the style of Williams et al. (2026) Figure 1.

    python research/plot_hs0209_regions.py --model DIR --raw RAW --output STEM \
        [--window-set figure1|carbon|metals] [--stis-velocity KMS] \
        [--paper-model CURVES.npz] [--compare-model DIR2]

``figure1`` uses ``windows.npz`` from ``research/hs0209_niobium.py``; ``carbon``
(the windows of the paper's Extended Data Figure 2) and ``metals`` (STIS regions
dense in the Zn IV, Cu IV and Ni IV lines of Extended Data Table 2) use
``windows_extended.npz``.  Each OpenWD model is projected as in
``plot_hs0209_niobium.py`` (paper R and d, one velocity per instrument, STScI
LSF or a Gaussian FUSE profile) and then multiplied by one constant per panel,
fitted to the observed continuum: the median data/model ratio over pixels
where the model lies within 1% of its 90th percentile, iterated with 3-sigma
clipping of the data (an error-weighted fit would favour the low-flux pixels
of unmodelled interstellar troughs).  The constants are recorded in the JSON
output.  The published model curves (``--paper-model``, Figure 1 windows only)
are shown unscaled.  Labels follow the paper: Figure 1 identifications for
``figure1``, otherwise every Extended Data Table 2 line in the panel, joined
per ion, and Extended Data Table 4 interstellar lines at 5 km/s.
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

import plot_hs0209_niobium as base

# Extended Data Table 2 of Williams et al. (2026), vacuum wavelengths (A).
PAPER_LINES = {
    "C II": (1009.858, 1010.083, 1010.371, 1036.337, 1037.018, 1323.951, 1334.532, 1335.708),
    "C III": (977.020, 1174.933, 1175.263, 1175.590, 1175.711, 1175.987, 1176.370, 1247.370),
    "C IV": (1548.187, 1550.772),
    "N III": (991.579,), "O III": (1153.775,), "Al III": (1605.766, 1611.873),
    "Si III": (1109.970, 1206.500), "Si IV": (1122.485, 1128.340, 1393.755, 1402.770),
    "P V": (1117.977,), "S IV": (1062.678,),
    "Ca III": (1278.393, 1281.554, 1286.525, 1298.034, 1335.128, 1385.430, 1453.161,
               1459.793, 1461.890, 1463.341, 1484.868, 1545.300, 1555.526, 1562.464),
    "Ti III": (1004.669, 1286.369, 1289.299, 1291.625, 1293.225, 1294.717, 1295.884,
               1298.697, 1298.996, 1327.609, 1420.439, 1421.641, 1421.755, 1424.136,
               1455.195, 1498.695, 1499.177),
    "Ti IV": (1183.635, 1195.208, 1451.736, 1467.338, 1469.188),
    "V III": (1125.699,), "Cr III": (1068.391,), "Mn IV": (1257.278,), "Fe III": (1124.874,),
    "Ni III": (1393.042, 1396.118, 1451.499, 1687.908, 1692.508),
    "Ni IV": (1357.064, 1398.193, 1399.947, 1411.451, 1414.597, 1421.216, 1430.190,
              1449.021, 1452.220, 1509.101, 1534.710),
    "Co IV": (1521.644,),
    "Cu III": (1355.442, 1367.628, 1376.790, 1382.538, 1385.892, 1430.358, 1543.450,
               1593.748, 1600.185, 1603.138, 1605.965, 1606.727, 1609.752, 1616.163,
               1626.133, 1626.405, 1628.089, 1628.291, 1642.202, 1654.561, 1681.468,
               1687.119, 1692.694),
    "Cu IV": (1309.406, 1312.715, 1318.117, 1340.062, 1347.785, 1362.019, 1367.519,
              1372.113, 1375.758, 1380.794, 1383.120, 1384.992, 1388.785, 1390.416,
              1392.097, 1410.016, 1410.531, 1411.570, 1412.206, 1413.016, 1426.615,
              1429.515, 1431.928, 1442.394, 1443.120, 1446.299, 1449.660),
    "Zn III": (1253.299, 1262.508, 1274.384, 1292.228, 1303.535, 1307.350, 1319.092,
               1323.501, 1328.367, 1343.346, 1359.601, 1359.799, 1362.520, 1364.323,
               1365.706, 1366.968, 1373.691, 1394.910, 1432.148, 1456.709, 1464.180,
               1473.399, 1490.951, 1498.778, 1499.408, 1500.411, 1505.913, 1515.834,
               1552.284, 1552.947, 1560.771, 1581.510, 1582.036, 1598.504, 1600.863,
               1619.601, 1622.505, 1629.163, 1639.320, 1644.807, 1651.738, 1673.077),
    "Zn IV": (1265.707, 1272.202, 1272.990, 1277.080, 1280.500, 1283.478, 1284.711,
              1291.826, 1292.476, 1296.652, 1296.734, 1301.189, 1306.657, 1318.001,
              1320.704, 1321.215, 1322.316, 1322.428, 1326.774, 1329.110, 1329.959,
              1333.326, 1340.156, 1342.716, 1343.750, 1344.122, 1347.954, 1349.876,
              1352.883, 1357.801, 1359.477, 1363.432, 1363.912, 1365.253, 1369.510,
              1375.325, 1377.615, 1387.193, 1387.694, 1459.944),
    "Sr IV": (1361.154,), "Y II": (1339.548,), "Zr III": (1356.947,),
    "Nb III": (1451.628, 1456.692, 1513.831, 1524.927, 1639.512),
    "Nb IV": (981.270, 992.567, 993.538, 1002.756, 1005.700, 1007.015, 1010.178,
              1013.808, 1030.271, 1035.216, 1044.904, 1049.610, 1050.976, 1054.384,
              1055.874, 1063.050, 1065.569, 1086.748, 1094.622, 1103.044, 1107.844,
              1116.081, 1119.835, 1120.243, 1125.274, 1127.498, 1128.611, 1136.689,
              1316.906, 1330.601, 1349.634, 1363.762, 1369.762, 1379.492, 1386.238,
              1418.884, 1420.662, 1424.368, 1434.140, 1434.223, 1444.491, 1447.491,
              1466.015, 1472.246, 1472.664, 1476.974, 1487.218, 1487.260, 1500.579,
              1508.721, 1510.832, 1517.461, 1524.383, 1532.606, 1532.981, 1534.059),
    "Ba II": (1554.295,), "Pb II": (1433.960,),
    "He II": (1640.474,),
}
# Extended Data Table 4 interstellar lines (shown at 5 km/s).
ISM_LINES = (936.629, 977.020, 988.773, 989.799, 989.873, 996.980, 1036.337, 1039.231,
             1083.990, 1134.165, 1134.415, 1134.980, 1190.416, 1193.290, 1199.550,
             1200.223, 1200.710, 1259.519, 1260.422, 1295.653, 1302.168, 1304.370,
             1334.532, 1526.707)
ISM_VELOCITY_KMS = 5.0

FIGURE1_PANELS = {name: (limits, labels) for name, _, limits, labels in base.PANELS}
CARBON_PANELS = {
    "fuse_977": (975.0, 979.0),
    "fuse_1037": (1035.7, 1038.3),
    "stis_1176": (1174.0, 1178.0),
    "stis_1247": (1246.5, 1249.0),
    "stis_1335": (1334.3, 1336.7),
    "stis_1550": (1547.5, 1552.5),
}
METAL_PANELS = {
    "stis_1284": (1282.5, 1286.5),
    "stis_1321": (1319.5, 1323.5),
    "stis_1398": (1395.5, 1400.5),
    "stis_1412": (1409.5, 1415.0),
}


def continuum_scale(flux, error, model, iterations=10):
    """Clipped median data/model ratio on near-continuum model pixels."""

    near = model >= 0.99 * np.percentile(model, 90)
    keep = near.copy()
    scale = float(np.median(flux[near] / model[near]))
    for _ in range(iterations):
        updated = near & (np.abs(flux - scale * model) < 3.0 * error)
        scale = float(np.median(flux[updated] / model[updated]))
        if np.array_equal(updated, keep):
            break
        keep = updated
    return scale, int(keep.sum())


def table_labels(low, high, stellar_factor):
    """Observed positions of every Extended Data Table 2/4 line, stacked in rows.

    Each ion is one group, labelled at the middle of its bar.  A group takes
    the lowest row where neither its bar nor its label overlaps another group,
    and where its ticks, which reach down into the row below, miss that row's
    labels (and its label misses the ticks of the row above).
    """

    ism_factor = base.doppler(ISM_VELOCITY_KMS)
    groups = []
    for ion, rests in PAPER_LINES.items():
        observed = [rest * stellar_factor for rest in rests
                    if low <= rest * stellar_factor <= high]
        if observed:
            groups.append((ion, observed))
    ism = [rest * ism_factor for rest in ISM_LINES if low <= rest * ism_factor <= high]
    if ism:
        groups.append(("ISM", ism))
    character = 0.019 * (high - low)
    pad = 0.012 * (high - low)
    rows: dict[int, list[tuple[tuple[float, float], tuple[float, float]]]] = {}

    def overlaps(first, second):
        return first[0] < second[1] and second[0] < first[1]

    placed = []
    for ion, observed in sorted(groups, key=lambda item: (-len(item[1]), min(item[1]))):
        middle = 0.5 * (min(observed) + max(observed))
        text = (middle - 0.5 * character * len(ion) - pad, middle + 0.5 * character * len(ion) + pad)
        ticks = (min(observed) - pad, max(observed) + pad)
        extent = (min(ticks[0], text[0]), max(ticks[1], text[1]))
        level = 1
        while (
            any(overlaps(extent, (min(t[0], x[0]), max(t[1], x[1]))) for t, x in rows.get(level, ()))
            or any(overlaps(ticks, x) for _, x in rows.get(level - 1, ()))
            or any(overlaps(text, t) for t, _ in rows.get(level + 1, ()))
        ):
            level += 1
        rows.setdefault(level, []).append((ticks, text))
        placed.append((ion, observed, level))
    return placed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-set", default="figure1", choices=("figure1", "carbon", "metals"))
    parser.add_argument("--stis-velocity", type=float, default=base.STIS_VELOCITY_KMS,
                        help="STIS photospheric velocity (default: the paper's mean)")
    parser.add_argument("--paper-model", type=Path)
    parser.add_argument("--compare-model", type=Path)
    parser.add_argument("--model-label", default="OpenWD model")
    parser.add_argument("--compare-label", default="OpenWD, Stout lines only")
    args = parser.parse_args()

    panels = {"figure1": FIGURE1_PANELS, "carbon": CARBON_PANELS,
              "metals": METAL_PANELS}[args.window_set]
    stem = "windows" if args.window_set == "figure1" else "windows_extended"
    model = np.load(args.model / f"{stem}.npz")
    compare = None if args.compare_model is None else np.load(args.compare_model / f"{stem}.npz")
    paper = None if args.paper_model is None else np.load(args.paper_model)
    # The FUSE zero point is fitted on the Figure 1 Nb IV windows, as in Figure 1.
    fuse_kms, _, _ = base.fuse_velocity(np.load(args.model / "windows.npz"), args.raw)
    stis = base.stis_data(args.raw)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 13, "axes.labelsize": 15,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
    })
    columns = 3 if len(panels) > 4 else 2
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.67 * columns, 3.9 * rows),
                                squeeze=False)
    record = {"window_set": args.window_set, "stis_velocity_kms": args.stis_velocity,
              "fuse_velocity_kms": fuse_kms, "ism_velocity_kms": ISM_VELOCITY_KMS,
              "model": str(args.model),
              "compare_model": None if compare is None else str(args.compare_model),
              "scaling": "one constant per panel and model: clipped median data/model "
                         "ratio on near-continuum model pixels",
              "panels": {}}
    for axis, (name, value) in zip(axes.flat, panels.items()):
        (low, high), labels = (value if args.window_set == "figure1" else (value, None))
        if name.startswith("stis"):
            keep = (stis[0] > low - 0.3) & (stis[0] < high + 0.3)
            wave, flux, error = (array[keep] for array in stis)
            center = 0.5 * (low + high)
            offsets, profile = base.stis_lsf(args.raw, center)
            kernel = (offsets * base.stis_native_dispersion(args.raw, center), profile)
            velocity, instrument = args.stis_velocity, "HST"
        else:
            # LiF1A starts at 987 A; below it the SiC 2A channel is the reference.
            reference = "1ALIF" if low > 988.0 else "2ASIC"
            wave, flux, error, _ = base.fuse_coadd(args.raw, low - 0.3, high + 0.3,
                                                   reference_channel=reference)
            sigma = np.median(wave) / base.FUSE_RESOLVING_POWER / 2.354_820_045
            offsets = np.linspace(-6 * sigma, 6 * sigma, 241)
            kernel = (offsets, np.exp(-0.5 * (offsets / sigma) ** 2))
            velocity, instrument = fuse_kms, "FUSE"
        shown = (wave >= low) & (wave <= high)
        info = {"instrument": instrument, "velocity_kms": velocity, "n_pixels": int(shown.sum())}
        curves = {}
        for label, source in (("model", model), ("compare", compare)):
            if source is None:
                continue
            curve = base.model_on_pixels(source, name, wave, velocity, kernel)["full"]
            scale, used = continuum_scale(flux[shown], error[shown], curve[shown])
            curves[label] = scale * curve
            residual = (flux[shown] - curves[label][shown]) / error[shown]
            info[label] = {"continuum_scale": scale, "continuum_pixels": used,
                           "chi2_per_pixel": float(np.mean(residual ** 2))}
            if label == "model":
                # Same structure and scale with Nb removed from the synthesis.
                no_nb = scale * base.model_on_pixels(source, name, wave, velocity,
                                                     kernel)["no_nb"]
                info[label]["chi2_per_pixel_no_nb"] = float(np.mean(
                    ((flux[shown] - no_nb[shown]) / error[shown]) ** 2))

        axis.plot(wave, flux / 1e-12, color="black", lw=0.9, label=f"{instrument} spectrum")
        values = [flux[shown] / 1e-12, curves["model"][shown] / 1e-12]
        if paper is not None and f"{name}_wavelength" in paper.files:
            paper_wave = paper[f"{name}_wavelength"]
            paper_flux = paper[f"{name}_flux_1e-12"]
            axis.plot(paper_wave, paper_flux, color="#e41a1c", lw=1.2,
                      label="Williams et al. (2026) model")
            values.append(paper_flux[(paper_wave >= low) & (paper_wave <= high)])
        if "compare" in curves:
            axis.plot(wave, curves["compare"] / 1e-12, color="#7f7f7f", lw=1.0, ls="--",
                      label=args.compare_label)
        axis.plot(wave, curves["model"] / 1e-12, color="#2166ac", lw=1.3, label=args.model_label)
        values = np.concatenate(values)
        bottom, top = np.min(values), np.percentile(values, 99.5)
        span = top - bottom
        factor = base.doppler(velocity)
        if labels is not None:
            placed = [(ion, [rest * factor for rest in rests], level, *alignment)
                      for ion, rests, level, *alignment in labels]
            if name == "fuse_1003":
                # Lift Ti III clear of the Nb IV bar and its label.
                placed = [(ion, xs, 3 if ion == "Ti III" else level, *rest)
                          for ion, xs, level, *rest in placed]
        else:
            placed = table_labels(low, high, factor)
        levels = max(item[2] for item in placed) if placed else 1
        axis.set_ylim(bottom - 0.08 * span, top + (0.30 + 0.12 * levels) * span)
        axis.set_xlim(low, high)
        axis.text(0.025, 0.955, instrument, transform=axis.transAxes, va="top", fontsize=13)
        tick_top = top + 0.15 * span
        for ion, observed, level, *alignment in placed:
            y = tick_top + (level - 1) * 0.12 * span
            for x in observed:
                axis.plot([x, x], [y - 0.07 * span, y], color="black", lw=1.0)
            if len(observed) > 1:
                axis.plot([min(observed), max(observed)], [y, y], color="black", lw=1.0)
            align = alignment[0] if alignment else "center"
            nudge = {"center": 0.0, "right": 0.03, "left": -0.03}[align]
            axis.text(np.mean([min(observed), max(observed)]) + nudge, y + 0.02 * span, ion,
                      ha=align, va="bottom", fontsize=12)
        axis.ticklabel_format(useOffset=False, axis="x")
        axis.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 5, 10], prune="lower"))
        axis.xaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        record["panels"][name] = info
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    handles, labels = [], []
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label.endswith("spectrum"):
                label = "Observed (FUSE or HST/STIS)"
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    figure.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False,
                  fontsize=13, bbox_to_anchor=(0.53, 1.0), handlelength=2.5)
    figure.supxlabel("Wavelength (Å)", fontsize=15, y=0.015)
    figure.supylabel(r"Flux ($10^{-12}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)", fontsize=15, x=0.012)
    figure.subplots_adjust(left=0.065 if columns == 3 else 0.095, right=0.99,
                           bottom=0.085, top=0.935, wspace=0.17, hspace=0.17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".pdf"))
    figure.savefig(args.output.with_suffix(".png"), dpi=250)
    args.output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in info.items() if k in ("model", "compare")}
                      for name, info in record["panels"].items()}, indent=1))


if __name__ == "__main__":
    main()
