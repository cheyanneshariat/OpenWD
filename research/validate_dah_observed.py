#!/usr/bin/env python3
"""Cold-start DAH models for the observed validation targets and score them.

By default each target uses its published Teff, log g and field geometry;
``--set`` explicitly overrides those values. One radial velocity and one flux scale are nuisance
operations (see ``wd_spectra.validation.magnetic_da``).  Targets run one at a
time in this process.  Each writes ``OUTPUT/KEY/`` (model, scores.json,
comparison.png); ``OUTPUT/summary.json`` collects the scores.

    PYTHONPATH=src OMP_NUM_THREADS=1 OPENWD_NUM_THREADS=1 \\
        python research/validate_dah_observed.py results/dah-validation \\
        --quality standard [--targets j2149-0728 j1033+2309] [--set key=value]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from wd_spectra.models import DAHConfig, compute_dah, save_model_result
from wd_spectra.validation.magnetic_da import (
    DAH_VALIDATION_BY_KEY,
    DAH_VALIDATION_TARGETS,
    DAH_WEAK_FIELD_BY_KEY,
    comparison_arrays,
    score_dah_spectrum,
    score_dah_weak_field_profiles,
)

# UVES resolves the weak-field components; sample the lines finely.
WEAK_WAVELENGTH = np.unique(
    np.concatenate(
        [np.arange(3_900.0, 6_800.0001, 0.5)]
        + [np.arange(c - 90.0, c + 90.0001, 0.02) for c in (6564.636, 4862.694, 4341.691, 4102.898)]
    )
)

WAVELENGTH = np.arange(3_600.0, 7_000.001, 0.5)


def plot(key: str, spectrum, score: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arrays = comparison_arrays(
        key,
        spectrum.wavelength_angstrom,
        spectrum.surface_flux_lambda,
        velocity_kms=float(score["velocity_kms"]),
    )
    target = DAH_VALIDATION_BY_KEY[key]
    figure, axis = plt.subplots(figsize=(11, 4.2))
    axis.plot(arrays["wavelength_angstrom"], arrays["observed"], color="0.15", lw=0.7, label="observed")
    if "hardy" in arrays:
        axis.plot(arrays["wavelength_angstrom"], arrays["hardy"], color="#1b9e77", lw=0.8,
                  label=f"Hardy et al. (broad {score['hardy']['broad_rms']:.3f})")
    axis.plot(arrays["wavelength_angstrom"], arrays["model"], color="#d95f02", lw=0.9,
              label=f"OpenWD DAH (broad {score['broad_rms']:.3f}, feature {score['feature_rms']:.3f})")
    axis.set_xlim(3_800.0, 7_000.0)
    axis.set_xlabel("rest vacuum wavelength (A)")
    axis.set_ylabel("F_nu / median(5200-6100 A)")
    axis.set_title(
        f"{target.display}: {target.effective_temperature:.0f} K, log g {target.logg:.2f}, "
        f"Bp {target.polar_field_megagauss:g} MG, i {target.inclination_deg:g} deg"
    )
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def run_weak(key: str, args, overrides: dict) -> None:
    """Cold start with full IQUV, then scalar Stokes I on the same structure."""

    target = DAH_WEAK_FIELD_BY_KEY[key]
    directory = args.output / key
    if (directory / "scores.json").is_file():
        print(f"{key}: already scored, skipping", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    try:
        (directory / "claimed").open("x").close()
    except FileExistsError:
        print(f"{key}: claimed by another runner, skipping", flush=True)
        return
    start = time.time()

    def iteration(index, atmosphere, status):
        print(
            f"[{key} {time.time() - start:7.0f}s] iteration {index} "
            f"flux={status.get('maximum_all_depth_total_flux_residual')} "
            f"local={status.get('maximum_relative_cell_energy_balance_residual')}",
            flush=True,
        )

    scores = {}
    atmosphere = None
    for transfer in ("full-stokes-iquv", "scalar-stokes-i"):
        config = DAHConfig(**{
            "quality": args.quality, **target.config_kwargs(), **overrides,
            "polarized_transfer": transfer,
        })
        result = compute_dah(
            config,
            WEAK_WAVELENGTH,
            initial_atmosphere=atmosphere,
            relax_atmosphere=atmosphere is None,
            iteration_callback=iteration,
        )
        if atmosphere is None:
            atmosphere = result.atmosphere
            status = result.metadata["atmosphere_convergence_status"]
        save_model_result(result, directory / transfer)
        score = score_dah_weak_field_profiles(
            key, result.spectrum.wavelength_angstrom, result.spectrum.surface_flux_lambda
        )
        score["structure_convergence"] = status
        scores[transfer] = score
        print(
            f"{key} {transfer}: profile mean {score['mean_rms']:.4f} "
            + " ".join(f"{n} {v['rms']:.4f}" for n, v in score["lines"].items()),
            flush=True,
        )
    scores["seconds"] = time.time() - start
    (directory / "scores.json").write_text(json.dumps(scores, indent=1, default=float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--quality", default="standard")
    parser.add_argument("--targets", nargs="*", default=[t.key for t in DAH_VALIDATION_TARGETS])
    parser.add_argument("--set", nargs="*", default=[], help="extra DAHConfig key=json-value")
    args = parser.parse_args()
    overrides = {}
    for item in args.set:
        name, value = item.split("=", 1)
        overrides[name] = json.loads(value)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    for key in args.targets:
        if key in DAH_WEAK_FIELD_BY_KEY:
            run_weak(key, args, overrides)
            continue
        target = DAH_VALIDATION_BY_KEY[key]
        directory = args.output / key
        if (directory / "scores.json").is_file():
            print(f"{key}: already scored, skipping", flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        try:
            # One worker per target when several runners share OUTPUT.
            (directory / "claimed").open("x").close()
        except FileExistsError:
            print(f"{key}: claimed by another runner, skipping", flush=True)
            continue
        config = DAHConfig(**{"quality": args.quality, **target.config_kwargs(), **overrides})
        start = time.time()

        def iteration(index, atmosphere, status):
            print(
                f"[{key} {time.time() - start:7.0f}s] iteration {index} "
                f"phase={status.get('solver_phase')} "
                f"flux={status.get('maximum_all_depth_total_flux_residual')} "
                f"local={status.get('maximum_relative_cell_energy_balance_residual')} "
                f"dlnT={status.get('maximum_log_temperature_correction')}",
                flush=True,
            )

        def progress(cell, cells):
            print(f"[{key} {time.time() - start:7.0f}s] synthesis cell {cell}/{cells}", flush=True)

        result = compute_dah(config, WAVELENGTH, iteration_callback=iteration, progress=progress)
        save_model_result(result, directory / "model")
        score = score_dah_spectrum(
            key, result.spectrum.wavelength_angstrom, result.spectrum.surface_flux_lambda
        )
        score["convergence"] = result.metadata["atmosphere_convergence_status"]
        score["seconds"] = time.time() - start
        score["overrides"] = overrides
        (directory / "scores.json").write_text(json.dumps(score, indent=1, default=float))
        plot(key, result.spectrum, score, directory / "comparison.png")
        summary = {
            path.parent.name: json.loads(path.read_text())
            for path in sorted(args.output.glob("*/scores.json"))
        }
        summary_path.write_text(json.dumps(summary, indent=1, default=float))
        print(
            f"{key}: broad {score['broad_rms']:.4f} feature {score['feature_rms']:.4f} "
            f"blue {score['blue_ratio']:.3f} ({score['convergence']}, {score['seconds']:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
