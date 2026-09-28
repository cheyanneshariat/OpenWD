"""Reproducible DAH comparison experiments with explicit checkpoint provenance.

Use --checkpoint for a fixed-structure diagnostic, add --relax to re-solve,
or omit the checkpoint for a cold start. Stellar parameters default to the
bundled published values; every override is saved with the result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from wd_spectra import DAHConfig, compute_dah
from wd_spectra.models.common import _jsonable, load_atmosphere_checkpoint, save_model_result
from wd_spectra.validation.magnetic_da import (
    DAH_VALIDATION_BY_KEY, DAH_WEAK_FIELD_BY_KEY, DAH_WEAK_LINE_WINDOWS,
    comparison_arrays, score_dah_spectrum, score_dah_weak_field_profiles,
    weak_field_profile_arrays,
)


def save_checkpoint(atmosphere, path):
    arrays = {name: getattr(atmosphere, name) for name in (
        "effective_temperature", "logg", "rosseland_optical_depth", "column_mass",
        "temperature", "gas_pressure", "mass_density", "electron_density",
    )}
    np.savez_compressed(path, **arrays,
        atmosphere_metadata_json=np.asarray(json.dumps(_jsonable(atmosphere.metadata))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target")
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--quality", default="standard")
    parser.add_argument("--step", type=float, default=1.)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "scores.json").exists():
        raise SystemExit("Output already scored; use a fresh experiment directory")
    weak = args.target in DAH_WEAK_FIELD_BY_KEY
    target = (DAH_WEAK_FIELD_BY_KEY if weak else DAH_VALIDATION_BY_KEY)[args.target]
    overrides = dict((k, json.loads(v)) for k, v in (item.split("=", 1) for item in args.set))
    config = DAHConfig(**{"quality": args.quality, **target.config_kwargs(), **overrides})
    wave = np.arange(3600., 7000.001, args.step)
    if weak:
        wave = np.unique(np.round(np.concatenate([wave] + [
            np.arange(center - 30, center + 30.001, .05)
            for _, center, _, _ in DAH_WEAK_LINE_WINDOWS
        ]), 6))
    package = Path(__file__).resolve().parents[1] / "src/wd_spectra"
    sources = [package / name for name in (
        "magnetic.py", "magnetic_atomic.py", "magnetic_continuum.py", "magnetic_eos.py",
        "radiative_transfer.py", "models/dah.py", "validation/magnetic_da.py",
    )]
    manifest = {
        "config": asdict(config), "target": args.target, "reference": target.reference,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "relax_atmosphere": args.relax or args.checkpoint is None,
        "wavelength_step": args.step, "wavelength_count": wave.size,
        "source_sha256": {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    (args.output / "request.json").write_text(json.dumps(manifest, indent=2))
    atmosphere = None
    if args.checkpoint:
        atmosphere = load_atmosphere_checkpoint(
            args.checkpoint, config.effective_temperature, config.logg, "hydrogen",
            include_molecules=config.effective_temperature <= 12000 and weak,
            include_negative_hydrogen=config.effective_temperature <= 12000 and weak,
        )
    start = time.monotonic()

    def iteration(index, atm, diagnostics):
        save_checkpoint(atm, args.output / "latest-atmosphere.npz")
        status = {"iteration": index, "seconds": time.monotonic() - start, **_jsonable(diagnostics)}
        with (args.output / "iterations.jsonl").open("a") as handle:
            handle.write(json.dumps(status) + "\n")
        print(args.target, index, round(status["seconds"], 1),
              diagnostics.get("maximum_all_depth_total_flux_residual"),
              diagnostics.get("maximum_relative_cell_energy_balance_residual"), flush=True)

    def progress(cell, cells):
        print(args.target, "cell", cell, "/", cells, round(time.monotonic() - start, 1), flush=True)

    result = compute_dah(config, wave, initial_atmosphere=atmosphere,
                         relax_atmosphere=manifest["relax_atmosphere"],
                         iteration_callback=iteration, progress=progress)
    save_model_result(result, args.output / "model")
    scorer = score_dah_weak_field_profiles if weak else score_dah_spectrum
    score = scorer(args.target, wave, result.spectrum.surface_flux_lambda)
    score.update(convergence=result.metadata["atmosphere_convergence_status"],
                 seconds=time.monotonic() - start, fixed_structure=not manifest["relax_atmosphere"])
    (args.output / "scores.json").write_text(json.dumps(score, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if weak:
        arrays = weak_field_profile_arrays(args.target, wave, result.spectrum.surface_flux_lambda,
                                           score["epoch_velocities_kms"])
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (name, (x, observed, model)) in zip(axes.flat, arrays.items()):
            ax.plot(x, observed, color="0.3", lw=.6, label="UVES")
            ax.plot(x, model, color="#b74d16", lw=1.1, label="OpenWD")
            ax.set_title(f"{name}: RMS {score['lines'][name]['rms']:.4f}")
            ax.set_xlabel("Vacuum wavelength (Å)")
            ax.set_ylabel("Normalized flux")
            ax.legend()
    else:
        arrays = comparison_arrays(args.target, wave, result.spectrum.surface_flux_lambda,
                                   velocity_kms=score["velocity_kms"])
        np.savez_compressed(args.output / "comparison-arrays.npz", **arrays)
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        x = arrays["wavelength_angstrom"]
        axes[0].plot(x, arrays["observed"], color="0.3", lw=.6, label="Observed")
        if "hardy" in arrays:
            axes[0].plot(x, arrays["hardy"], color="#008572", lw=1, label="Hardy et al.")
        axes[0].plot(x, arrays["model"], color="#b74d16", lw=1.1, label="OpenWD")
        axes[0].set_ylabel("Scaled Fν")
        axes[0].legend()
        axes[1].plot(x, arrays["model"] / arrays["observed"] - 1, color="#b74d16", lw=.6)
        axes[1].axhline(0, color="0.3", lw=.7)
        axes[1].set_ylim(-.3, .3)
        axes[1].set_ylabel("Model/data − 1")
        axes[1].set_xlabel("Vacuum wavelength (Å)")
        axes[1].set_xlim(3820, 6950)
    fig.suptitle(f"{target.display} — {args.output.name}; {score['convergence']}")
    fig.tight_layout()
    fig.savefig(args.output / "comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps(score, indent=2), flush=True)


if __name__ == "__main__":
    main()
