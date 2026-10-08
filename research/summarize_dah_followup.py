"""Collect observational experiment scores and plot the three-object follow-up."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wd_spectra import DAHConfig
from wd_spectra.validation.magnetic_da import (
    DAH_VALIDATION_BY_KEY, DAH_WEAK_FIELD_BY_KEY, weak_field_profile_arrays,
)
from wd_spectra.validation.observed import gaussian_smooth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--gd9-only", action="store_true", help="plot GD 9 while strong-field solves run")
    args = parser.parse_args()
    root = args.root
    policy_path = root / "comparison-policy.json"
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
    excluded = set(policy.get("excluded_parameter_experiments", []))
    summary = {}
    fixed_parameter_verification = {}
    physical_keys = ("effective_temperature", "logg", "magnetic_field_megagauss",
                     "field_geometry", "field_angle_deg", "field_strength_definition",
                     "dipole_inclination_deg", "dipole_offset_radius")
    for path in sorted(root.glob("*/*/scores.json")):
        score = json.loads(path.read_text())
        metadata = json.loads((path.parent / "model/metadata.json").read_text())
        atmosphere = metadata["atmosphere_metadata"]
        score["config"] = metadata["config"]
        target = DAH_WEAK_FIELD_BY_KEY.get(score["target"]) or DAH_VALIDATION_BY_KEY[score["target"]]
        expected = json.loads(json.dumps(asdict(DAHConfig(**target.config_kwargs()))))
        differences = {k: {"requested": score["config"][k], "published": expected[k]}
                       for k in physical_keys if score["config"][k] != expected[k]}
        tag = str(path.parent.relative_to(root))
        score["included_in_fixed_parameter_comparison"] = not differences and tag not in excluded
        fixed_parameter_verification[tag] = {
            "included": score["included_in_fixed_parameter_comparison"],
            "stellar_or_field_parameter_differences": differences,
        }
        score["depth_count"] = int(np.load(path.parent / "model/atmosphere.npz")["temperature"].size)
        score["equilibrium"] = {
            key: atmosphere.get(key) for key in (
                "maximum_all_depth_total_flux_residual",
                "maximum_relative_cell_energy_balance_residual",
                "radiative_equilibrium_maximum_log_temperature_correction",
            )
        }
        summary.setdefault(path.parent.parent.name, {})[path.parent.name] = score
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    (root / "fixed-parameter-verification.json").write_text(json.dumps(fixed_parameter_verification, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    velocity = summary["gd9"]["production-current"]["epoch_velocities_kms"]
    experiments = (
        ("baseline", "Old saved structure", "#9b6b50", "--"),
        ("production-current", "Updated, published parameters", "#2463a0", "-"),
    )
    for index, (experiment, label, color, linestyle) in enumerate(experiments):
        wave, flux = np.loadtxt(root / "gd9" / experiment / "model/spectrum.txt", unpack=True)
        arrays = weak_field_profile_arrays("gd9", wave, flux, velocity)
        for ax, (name, (x, observed, model)) in zip(axes.flat, arrays.items()):
            if index == 0:
                ax.plot(x, observed, color="0.65", lw=.45, label="UVES")
            ax.plot(x, model, color=color, lw=1.25, ls=linestyle, label=label)
            ax.set_title(name)
            ax.set_xlabel("Vacuum wavelength (Å)")
            ax.set_ylabel("Normalized flux")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("GD 9: atmosphere update at unchanged published parameters")
    fig.tight_layout()
    fig.savefig(root / "gd9-improvement.png", dpi=160)
    plt.close(fig)
    if args.gd9_only:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex="col")
    for column, target in enumerate(("j2149-0728", "j0732+3646")):
        base = np.load(root / target / "baseline/comparison-arrays.npz")
        update_tag = "production-current" if "production-current" in summary[target] else "relaxed-current"
        update = np.load(root / target / update_tag / "comparison-arrays.npz")
        update_score = summary[target][update_tag]
        x = base["wavelength_angstrom"]
        axes[0, column].plot(x, base["observed"], color="0.6", lw=.6, label="Observed")
        axes[0, column].plot(x, base["hardy"], color="#398b71", lw=1., label="Hardy reference")
        for arrays, label, color, style in (
            (base, "Old saved structure", "#9b6b50", "--"),
            (update, f"Updated structure ({update_score['depth_count']} depths)", "#2463a0", "-"),
        ):
            wave = arrays["wavelength_angstrom"]
            axes[0, column].plot(wave, arrays["model"], color=color, lw=1.2, ls=style, label=label)
            ratio = gaussian_smooth(arrays["model"], 12.) / gaussian_smooth(arrays["observed"], 12.)
            axes[1, column].plot(wave, ratio - 1, color=color, lw=1., ls=style)
        status = update_score["convergence"]
        axes[0, column].set_title(f"{target.upper()} — {status}")
        axes[0, column].set_ylabel("Scaled Fν")
        axes[0, column].legend(fontsize=8)
        axes[1, column].axhline(0, color="0.3", lw=.7)
        axes[1, column].set_ylim(-.25, .25)
        axes[1, column].set_xlim(3820, 6950)
        axes[1, column].set_ylabel("Smoothed ratio − 1")
        axes[1, column].set_xlabel("Vacuum wavelength (Å)")
    fig.suptitle("Strong-field comparisons at unchanged published parameters")
    fig.tight_layout()
    fig.savefig(root / "strong-field-comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, (target, experiment) in zip(axes, (
        ("gd9", "cold-current"), ("j2149-0728", "relaxed-current"),
        ("j0732+3646", "relaxed-current"),
    )):
        records = [json.loads(line) for line in (root / target / experiment / "iterations.jsonl").read_text().splitlines()]
        for key, label, color in (
            ("maximum_all_depth_total_flux_residual", "Total flux", "#2463a0"),
            ("maximum_relative_cell_energy_balance_residual", "Local energy", "#b95f24"),
        ):
            valid = [r for r in records if r.get(key) is not None]
            ax.semilogy([r["seconds"] / 60 for r in valid], [r[key] for r in valid],
                        color=color, label=label)
        ax.axhline(.002, color="0.5", ls=":", lw=1, label="0.2% tolerance")
        ax.set_title(target.upper())
        ax.set_xlabel("Elapsed atmosphere solve (minutes)")
        ax.set_ylabel("Maximum fractional residual")
        ax.legend(fontsize=8)
    fig.suptitle("Flux balance alone can miss substantial local energy imbalance")
    fig.tight_layout()
    fig.savefig(root / "convergence-history.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
