#!/usr/bin/env python3
"""Plot the cool DZ/DAZ paper comparison (five helium-dominated DZs and one DAZ).

Inputs are the models computed by ``run_model.py`` (MODELS/<key>/) and the
public spectra downloaded by ``fetch_spectra.py`` (DATA/<key>/).  Parameters
and abundances are the published values in ``objects.json``; each model's
stored configuration is checked against them.

Display operations only:
  * one radial velocity per star, the median of chi^2 minima over strong-line
    windows (each fitted with a local linear continuum);
  * for X-shooter, the UVB arm is put on the VIS flux scale from the
    observation alone, by matching straight-line fits on either side of the
    5560-A dichroic junction;
  * where objects.json names a "continuum_reference" (WD J1336-0337, whose
    X-shooter continuum carries a chromatic slit-loss slope), the X-shooter
    flux is divided by a quadratic fitted to the ratio of the two observed
    spectra in 30-A bins over 3800-8800 A; the model is not involved;
  * one multiplicative constant for the whole model (no slope or continuum
    normalization), so the panels test the SED shape and the line profiles.
The model is convolved with a Gaussian line-spread function at the
resolution of the spectrograph segment covering each wavelength (the nearest
segment inside data gaps) and drawn continuously; data and model are
averaged in 2-A bins of vacuum rest wavelength.

usage: python research/cool_dz_paper/plot_figure.py MODELS DATA OUTPUT.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

from wd_spectra.validation.observed import convolve_variable_log_gaussian, read_sdss_spectrum  # noqa: E402

HERE = Path(__file__).resolve().parent
ORDER = ("wd2251", "j0916", "vma2", "j1227", "j0255", "j1336")
C_KMS = 299792.458
WAVELENGTH_LIMITS = (3500.0, 7000.0)
BIN = 2.0
XSHOOTER_JUNCTION = (5540.0, 5590.0)          # UVB/VIS dichroic edge pixels, not shown
# telluric and sky-residual masks (vacuum A)
TELLURIC = ((5570, 5585), (6865, 6935), (7165, 7340), (7595, 7700), (8130, 8350), (8950, 9850), (6280, 6320))
# strong-line windows used to measure the radial velocity (vacuum A)
VELOCITY_WINDOWS = ((3926, 3942), (3962, 3978), (4222, 4234), (3826, 3842), (5162, 5190),
                    (5884, 5902), (6556, 6572), (8494, 8506), (8536, 8550), (8656, 8670))
SDSS_RESOLVING_POWER = 1900.0
DESI_RESOLVING_POWER = {"b": 2600.0, "r": 3650.0, "z": 4600.0}   # arm midpoints (DESI Collaboration 2022)
INSTRUMENT_LABELS = {"xshooter": "VLT/X-shooter", "desi": "DESI DR1", "sdss": "SDSS"}
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})


def air_to_vacuum(w):
    s2 = (1e4 / w) ** 2
    return w * (1 + 6.4328e-5 + 2.94981e-2 / (146 - s2) + 2.5540e-4 / (41 - s2))


def unmasked(w):
    keep = np.ones(w.size, bool)
    for a, b in TELLURIC:
        keep &= ~((w > a) & (w < b))
    return keep


def read_segments(entry, directory):
    """Observed spectrograph segments: vacuum barycentric wavelength, flux, inverse variance, R."""
    if entry["dataset"] == "xshooter":
        from astropy.io import fits
        segments = []
        for dataset_id in entry["eso"]:
            with fits.open(directory / f"{dataset_id.replace(':', '_')}.fits", memmap=False) as hdus:
                header, table = hdus[0].header, hdus[1]
                column = list(table.columns.names).index("WAVE") + 1
                if "obs.atmos" not in str(table.header.get(f"TUCD{column}", "")):
                    raise ValueError(f"{dataset_id}: X-shooter wavelengths are expected in air")
                unit = {"nm": 10.0, "angstrom": 1.0, "Angstrom": 1.0}[table.header[f"TUNIT{column}"]]
                w = air_to_vacuum(np.asarray(table.data["WAVE"], float).ravel() * unit)
                f = np.asarray(table.data["FLUX"], float).ravel()
                error = np.asarray(table.data["ERR"], float).ravel()
                quality = np.asarray(table.data["QUAL"]).ravel()
                w = w * (1 + float(header.get("ESO QC VRAD BARYCOR", 0.0)) / C_KMS)
                arm = str(header.get("ESO SEQ ARM", dataset_id))
                resolving_power = float(header.get("SPEC_RES", 5000.0))
            if arm == "NIR":
                continue
            with np.errstate(over="ignore", divide="ignore"):
                iv = np.where((error > 0) & (quality == 0) & (f != 0), 1 / np.maximum(error, 1e-300) ** 2, 0.0)
            iv = np.where(w > 5560, 0.0, iv) if arm == "UVB" else np.where(w < 5560, 0.0, iv)
            segments.append(dict(w=w, f=f, iv=iv, R=resolving_power, arm=arm))
        segments.sort(key=lambda s: s["w"][0])
        # cross-arm calibration from the observation alone: match the two arms at the junction
        uvb, vis = segments[0], segments[1]
        k0 = (uvb["w"] > 5300) & (uvb["w"] < 5560) & (uvb["iv"] > 0)
        k1 = (vis["w"] > 5560) & (vis["w"] < 5800) & (vis["iv"] > 0)
        scale = (np.polyval(np.polyfit(vis["w"][k1] - 5560, vis["f"][k1], 1), 0.0)
                 / np.polyval(np.polyfit(uvb["w"][k0] - 5560, uvb["f"][k0], 1), 0.0))
        uvb["f"] = uvb["f"] * scale
        return segments, {"uvb_to_vis_scale": float(scale)}
    if entry["dataset"] == "sdss":
        spectrum = read_sdss_spectrum(directory / f"spec-{entry['sdss']}.fits")
        return [dict(w=spectrum.wavelength_angstrom, f=spectrum.flux_lambda, iv=spectrum.inverse_variance,
                     R=SDSS_RESOLVING_POWER, arm="sdss")], {}
    data = np.load(directory / f"desi-{entry['desi'][0]}.npz")
    segments = []
    for arm, resolving_power in DESI_RESOLVING_POWER.items():
        iv = data[f"{arm}_ivar"] * 1e34
        if f"{arm}_mask" in data:
            iv = np.where(data[f"{arm}_mask"] == 0, iv, 0.0)
        segments.append(dict(w=data[f"{arm}_wavelength"], f=data[f"{arm}_flux"] * 1e-17, iv=iv,
                             R=resolving_power, arm=arm))
    segments[0]["iv"] = np.where(segments[0]["w"] > 5790, 0, segments[0]["iv"])     # keep the higher-R arm
    segments[1]["iv"] = np.where(segments[1]["w"] > 7570, 0, segments[1]["iv"])     # in each overlap
    return segments, {}


def binned_observation(segments, edges):
    """Mean observed flux in wavelength bins (valid, unmasked pixels; NaN with fewer than 10 pixels)."""
    w = np.concatenate([s["w"][(s["iv"] > 0) & np.isfinite(s["f"])] for s in segments])
    f = np.concatenate([s["f"][(s["iv"] > 0) & np.isfinite(s["f"])] for s in segments])
    keep = unmasked(w) & ((w < XSHOOTER_JUNCTION[0]) | (w > XSHOOTER_JUNCTION[1]))
    w, f = w[keep], f[keep]
    index = np.digitize(w, edges)
    count = np.bincount(index, minlength=edges.size + 1)[1:edges.size]
    total = np.bincount(index, weights=f, minlength=edges.size + 1)[1:edges.size]
    return np.where(count >= 10, total / np.maximum(count, 1), np.nan)


def continuum_correction(segments, reference):
    """Quadratic in wavelength fitted to (segments / reference) in 30-A bins, normalized at 5500 A.

    Bin averages barely depend on resolution, so the two observations' lines cancel in the
    ratio; three passes of 3-sigma (MAD) clipping remove the bins where they do not."""
    edges = np.arange(3800.0, 8800.0 + 1e-9, 30.0)
    x = (0.5 * (edges[1:] + edges[:-1]) - 6000.0) / 1000.0
    ratio = binned_observation(segments, edges) / binned_observation(reference, edges)
    keep = np.isfinite(ratio) & (ratio > 0)
    for _ in range(3):
        coefficients = np.polyfit(x[keep], ratio[keep], 2)
        residual = ratio - np.polyval(coefficients, x)
        scatter = 1.4826 * np.median(np.abs(residual[keep]))
        keep = np.isfinite(ratio) & (ratio > 0) & (np.abs(residual) < 3 * scatter)
    coefficients = coefficients / np.polyval(coefficients, -0.5)
    return lambda w: np.polyval(coefficients, (np.asarray(w) - 6000.0) / 1000.0)


def broaden(wm, fm, resolving_power, lo, hi):
    """Gaussian line-spread function at constant resolving power (velocity search only)."""
    keep = (wm > lo * 0.98) & (wm < hi * 1.02)
    x = np.log(wm[keep])
    step = 1.0 / (resolving_power * 12.0)
    grid = np.arange(x[0], x[-1], step)
    y = np.interp(grid, x, fm[keep])
    sigma = 1.0 / (2.354820045 * resolving_power) / step
    half = int(5 * sigma) + 1
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    return np.exp(grid), np.convolve(np.pad(y, half, mode="edge"), kernel / kernel.sum(), mode="valid")


def local_fit(w, f, iv, model, a, b):
    """obs = model (c0 + c1 x) in [a, b], with iterative 5-sigma clipping."""
    sel = (w > a) & (w < b) & (iv > 0) & np.isfinite(model) & np.isfinite(f) & unmasked(w)
    if sel.sum() < 10:
        return None
    x = (w[sel] - 0.5 * (a + b)) / (b - a)
    design = np.column_stack((model[sel], model[sel] * x))
    weight = np.sqrt(iv[sel])
    keep = np.ones(sel.sum(), bool)
    for _ in range(3):
        coefficients, *_ = np.linalg.lstsq(design[keep] * weight[keep, None], f[sel][keep] * weight[keep], rcond=None)
        fit = design @ coefficients
        z = (f[sel] - fit) * weight
        keep = np.abs(z) < max(5 * 1.4826 * np.median(np.abs(z[keep])), 5)
    return sel, keep, fit


def radial_velocity(segments, wm, fm):
    """Median of chi^2 minima over the strong-line windows with a defined minimum (> 10% contrast)."""
    broadened = [broaden(wm, fm, s["R"], s["w"].min(), s["w"].max()) for s in segments]
    grid = np.arange(-150.0, 251.0, 5.0)
    votes = []
    for a, b in VELOCITY_WINDOWS:
        cost, used = np.zeros(grid.size), False
        for i, v in enumerate(grid):
            for segment, (mw, mf) in zip(segments, broadened):
                model = np.interp(segment["w"], mw * (1 + v / C_KMS), mf, left=np.nan, right=np.nan)
                fitted = local_fit(segment["w"], segment["f"], segment["iv"], model, a - 15, b + 15)
                if fitted is None:
                    continue
                sel, keep, fit = fitted
                used = True
                cost[i] += np.sum(((segment["f"][sel] - fit) ** 2 * segment["iv"][sel])[keep])
        if not used or cost.min() <= 0 or cost.max() / cost.min() < 1.1:
            continue
        k = int(np.argmin(cost))
        if 0 < k < grid.size - 1:
            c0, c1, c2 = cost[k - 1:k + 2]
            curvature = c0 - 2 * c1 + c2
            votes.append(grid[k] + (0.5 * 5.0 * (c0 - c2) / curvature if curvature > 0 else 0.0))
    return float(np.median(votes)) if votes else 0.0


def binned(w, *values):
    edges = np.arange(WAVELENGTH_LIMITS[0], WAVELENGTH_LIMITS[1] + BIN, BIN)
    index = np.digitize(w, edges)
    count = np.bincount(index, minlength=edges.size + 1)[1:edges.size]
    out = [0.5 * (edges[1:] + edges[:-1])]
    for y in values:
        total = np.bincount(index, weights=y, minlength=edges.size + 1)[1:edges.size]
        out.append(np.where(count > 0, total / np.maximum(count, 1), np.nan))
    return tuple(out)


def panel(entry, model_dir, data_dir):
    with np.load(model_dir / "spectrum_fine.npz") as model:
        wm, fm = np.asarray(model["wavelength"], float), np.asarray(model["flux"], float)
    segments, notes = read_segments(entry, data_dir)
    if entry.get("continuum_reference"):
        reference, _ = read_segments(dict(entry, dataset=entry["continuum_reference"]), data_dir)
        correction = continuum_correction(segments, reference)
        for segment in segments:
            factor = correction(segment["w"])
            segment["f"], segment["iv"] = segment["f"] / factor, segment["iv"] * factor**2
        notes["continuum_correction_at"] = {str(w): float(correction(w)) for w in (4000, 4500, 5500, 6500, 7000)}
    velocity = radial_velocity(segments, wm, fm)
    ws, fs, ranges = [], [], []
    for segment in segments:
        rest = segment["w"] / (1 + velocity / C_KMS)
        ok = ((segment["iv"] > 0) & np.isfinite(segment["f"]) & unmasked(segment["w"])
              & (rest >= WAVELENGTH_LIMITS[0]) & (rest <= WAVELENGTH_LIMITS[1]))
        if entry["dataset"] == "xshooter":
            ok &= (rest < XSHOOTER_JUNCTION[0]) | (rest > XSHOOTER_JUNCTION[1])
        if ok.sum() < 50:
            continue
        ws.append(rest[ok])
        fs.append(segment["f"][ok])
        ranges.append((rest[ok].min(), rest[ok].max(), segment["R"]))
    # continuous model: each grid point at the resolution of the covering (or nearest) segment
    grid = np.arange(WAVELENGTH_LIMITS[0] - 20.0, WAVELENGTH_LIMITS[1] + 20.0, 0.1)
    distance = np.array([np.maximum(np.maximum(lo - grid, grid - hi), 0.0) for lo, hi, _ in ranges])
    resolving_power = np.array([r for _, _, r in ranges])[np.argmin(distance, axis=0)]
    source = (wm > grid[0] - 30) & (wm < grid[-1] + 30)
    model_grid = convolve_variable_log_gaussian(wm[source], fm[source], grid,
                                                1.0 / (2.354820045 * resolving_power))
    w, f = np.concatenate(ws), np.concatenate(fs)
    order = np.argsort(w)
    w, f = w[order], f[order]
    scale = float(np.sum(f) / np.sum(np.interp(w, grid, model_grid)))
    bw, bf = binned(w, f)
    _, bm = binned(grid, model_grid * scale)
    norm = float(np.nanmedian(bf))
    good = np.isfinite(bf) & (bf > 0) & (bm > 0)
    return dict(w=bw, obs=bf / norm, model=bm / norm, velocity_km_s=velocity, model_scale=scale,
                rms_log_ratio_binned=float(np.sqrt(np.mean(np.log(bf[good] / bm[good]) ** 2))), **notes)


def abundance_label(entry):
    if entry.get("label_calcium_only"):
        return rf"$\log N(\mathrm{{Ca}})/N(\mathrm{{H}})={entry['abundances']['Ca']:.2f}$; others CI-chondritic"
    prefix = ""
    if entry["host"] == "He":
        prefix = (rf"$\log N(\mathrm{{H}})/N(\mathrm{{He}})={entry['log_h']:.2f}$; " if entry["log_h"] is not None
                  else "pure He; ")
    denominator = r"\mathrm{He}" if entry["host"] == "He" else r"\mathrm{H}"
    values = ", ".join(f"{element}={value:.2f}" for element, value in entry["abundances"].items())
    return prefix + rf"$\log N(Z)/N({denominator})$: " + values


def make_figure(models, data, output):
    objects = json.loads((HERE / "objects.json").read_text())["objects"]
    figure, axes = plt.subplots(len(ORDER), 1, figsize=(8.0, 10.0), sharex=True, gridspec_kw={"hspace": 0.0})
    record = {}
    for axis, key in zip(axes, ORDER):
        entry, model_dir = objects[key], models / key
        if not json.loads((model_dir / "summary.json").read_text()).get("verified"):
            raise RuntimeError(f"{model_dir}: the model did not pass its convergence certificate")
        config = json.loads((model_dir / "metadata.json").read_text())["config"]
        # the plotted model must be at the published parameters and abundances
        assert abs(config["effective_temperature"] - entry["teff"]) < 1e-6, key
        assert abs(config["logg"] - entry["logg"]) < 1e-9, key
        assert config["abundances"] == entry["abundances"], key
        assert config.get("log_hydrogen_abundance") == entry["log_h"], key
        d = panel(entry, model_dir, data / key)
        axis.plot(d["w"], d["obs"], color="0.30", lw=0.5, rasterized=True)
        axis.plot(d["w"], d["model"], color="#d95f02", lw=0.6)
        top = 1.32 * np.nanpercentile(np.r_[d["obs"], d["model"]], 99.7)
        axis.set_ylim(0.0, top)
        axis.set_yticks(np.arange(0.0, 0.8 * top, 0.5))           # keep tick labels clear of the panel joins
        axis.set_xlim(*WAVELENGTH_LIMITS)
        axis.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=10, length=4)
        axis.text(0.985, 0.965, rf"{entry['name']} ({INSTRUMENT_LABELS[entry['dataset']]})   "
                  rf"$T_{{\rm eff}}={entry['teff']:.0f}\,$K, $\log g={entry['logg']:.2f}$",
                  transform=axis.transAxes, ha="right", va="top", fontsize=11, zorder=5)
        axis.text(0.985, 0.835, abundance_label(entry), transform=axis.transAxes, ha="right", va="top",
                  fontsize=7.6, zorder=5)
        record[key] = {k: v for k, v in d.items() if k not in ("w", "obs", "model")}
        print(f"{key:7s} v={d['velocity_km_s']:+6.1f} km/s  rms(log, {BIN:.0f} A bins)={d['rms_log_ratio_binned']:.3f}")
    axes[0].legend(handles=(Line2D([], [], color="0.30", lw=0.8, label="Observation"),
                            Line2D([], [], color="#d95f02", lw=0.8, label="OpenWD")),
                   loc="lower right", frameon=False, fontsize=10, ncol=2, handlelength=2.3, columnspacing=1.4)
    axes[-1].set_xlabel(r"vacuum rest wavelength [$\mathrm{\AA}$]", fontsize=11)
    figure.supylabel(r"relative $f_\lambda$", fontsize=11, x=0.015)
    figure.subplots_adjust(left=0.085, right=0.975, bottom=0.065, top=0.995)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(figure)
    output.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    make_figure(args.models, args.data, args.output)
