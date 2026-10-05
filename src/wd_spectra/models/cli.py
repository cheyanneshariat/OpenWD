"""Command-line interface shared by the one-shot examples."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from .common import ModelData, load_atmosphere_checkpoint, save_model_result
from .automatic import run_model
from .d6 import D6Config
from .dah import DAHConfig
from .pg1159 import PG1159Config
from .stellar import (
    DAConfig,
    DABConfig,
    DBConfig,
    DZConfig,
    compute_da,
    compute_dab,
    compute_db,
    compute_dz,
)


def _optional_float(text: str) -> float | None:
    if text.strip().lower() == "none":
        return None
    return float(text)


def _assignment(value: str) -> tuple[str, float]:
    try:
        element, raw = value.split("=", 1)
        return element.strip(), float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected Element=value") from exc


def _quicklook(spectrum, config, spectral_type: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wavelength = spectrum.wavelength_angstrom
    flux = spectrum.surface_flux_lambda
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].loglog(wavelength, wavelength * flux, color="#d55e00", lw=0.9)
    axes[0].set_ylabel(r"$\lambda F_\lambda$")
    axes[0].set_title(
        f"{spectral_type}: "
        f"Teff={config.effective_temperature:.0f} K, "
        f"log g={config.logg:.2f}"
    )
    optical = (wavelength >= 3400.0) & (wavelength <= 7500.0)
    if np.any(optical):
        normalizer = np.nanpercentile(flux[optical], 95.0)
        axes[1].plot(wavelength[optical], flux[optical] / normalizer,
                     color="#d55e00", lw=0.7)
        axes[1].set_xlim(3400.0, 7500.0)
        axes[1].set_ylabel("Relative surface flux")
    axes[1].set_xlabel(r"Vacuum wavelength [$\AA$]")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def one_shot_main(spectral_type: str) -> None:
    """Run a DA, DB, DAB, or DZ atmosphere and formal spectrum."""

    kind = spectral_type.upper()
    if kind not in {"DA", "DB", "DAB", "DZ", "PG1159", "D6", "DAH"}:
        raise ValueError(f"unsupported spectral type {spectral_type!r}")
    parser = argparse.ArgumentParser(
        description=f"Produce one self-consistent {kind} atmosphere and spectrum."
    )
    parser.add_argument("--teff", type=float)
    parser.add_argument("--logg", type=float)
    parser.add_argument("--quality", choices=("quick", "standard", "production"),
                        default="standard")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="optional external data root; bundled data are the default")
    parser.add_argument("--output", type=Path,
                        default=Path("results/one-shot") / kind.lower())
    parser.add_argument("--wavelength-min", type=float)
    parser.add_argument("--wavelength-max", type=float)
    parser.add_argument("--wavelength-step", type=float)
    parser.add_argument("--synthesize-atmosphere", "--restart-atmosphere", type=Path,
                        dest="restart_atmosphere",
                        help="diagnostic spectrum on a fixed atmosphere; does not resume a solve")
    parser.add_argument("--research-data", type=Path,
                        help="directory containing the qualified molecular DAB tables")
    parser.add_argument("--require-convergence", action="store_true")
    if kind in {"DA", "DAB", "DZ"}:
        parser.add_argument("--lyman-profiles", choices=("allard", "stark"),
                            default=(DZConfig().lyman_profile_source
                                     if kind == "DZ" else "allard"))
    if kind == "DA":
        parser.add_argument("--h3plus-partition",
                            choices=("neale-tennyson-1995", "none"),
                            default="neale-tennyson-1995")
    if kind == "DAB":
        parser.add_argument("--log-h-he", type=float, default=-2.0)
    if kind == "DZ":
        parser.add_argument("--log-h-he", type=_optional_float,
                            default=DZConfig().log_hydrogen_abundance,
                            help="log10 N(H)/N(He), or 'none' for a hydrogen-free atmosphere")
    if kind == "DZ":
        parser.add_argument("--abundance", action="append", type=_assignment,
                            help="replace defaults with Element=log10(N/He)")
        parser.add_argument("--dense-helium-eos", choices=("ideal", "reos3"),
                            default=DZConfig().dense_helium_eos)
        parser.add_argument("--strong-line-atomic-data",
                            choices=("stout", "nist-asd"),
                            default=DZConfig().strong_line_atomic_data)
    if kind == "D6":
        parser.add_argument("--abundance", action="append", type=_assignment,
                            help="Element=log10 N(X)/N(reference); replaces the J1637 defaults")
        parser.add_argument("--reference-element", default="C")
        parser.add_argument("--compare-hollands2025", action="store_true",
                            help="score the spectrum against the bundled J1637 digitization")
    if kind == "DAH":
        parser.add_argument("--field-mg", type=float,
                            help="uniform field, or dipole polar field (MG)")
        parser.add_argument("--geometry", choices=("uniform", "dipole"), default="uniform")
        parser.add_argument("--field-angle", type=float, default=None,
                            help="uniform field-to-line-of-sight angle; default isotropic")
        parser.add_argument("--inclination", type=float, default=DAHConfig().dipole_inclination_deg)
        parser.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                            metavar=("AX", "AY", "AZ"),
                            help="dipole offset in stellar radii, magnetic-axis frame")
        parser.add_argument("--transfer", choices=("full-stokes-iquv", "scalar-stokes-i"),
                            default=DAHConfig().polarized_transfer)
        parser.add_argument("--compare", default=None,
                            help="score against a bundled validation target, e.g. j2149-0728")
    if kind == "PG1159":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        parser.add_argument("--mass-fraction", action="append", type=_assignment)
        parser.add_argument("--target-name", default="PG 1424+535")
        parser.add_argument("--oxygen-atom", choices=("compact14", "extended54-complete"), default="extended54-complete")
        parser.add_argument("--refine-upper-atmosphere", action="store_true",
                            help="converge tau_Ross < 1e-2 to local radiative equilibrium and re-certify")
        parser.add_argument("--radiative-acceleration", action="store_true",
                            help="include radiation pressure in hydrostatic equilibrium")
    args = parser.parse_args()

    supplied_grid = (args.wavelength_min, args.wavelength_max, args.wavelength_step)
    if any(value is not None for value in supplied_grid):
        if not all(value is not None for value in supplied_grid):
            parser.error("supply all three wavelength options together")
        if (args.wavelength_min <= 0 or args.wavelength_max <= args.wavelength_min
                or args.wavelength_step <= 0):
            parser.error("wavelength bounds and step must be positive and increasing")
        wavelength = np.arange(args.wavelength_min,
                               args.wavelength_max + 0.5 * args.wavelength_step,
                               args.wavelength_step)
    else:
        wavelength = None

    data = ModelData.default(args.data_root)
    if kind == "PG1159":
        if args.restart_atmosphere is not None:
            parser.error("PG1159 requires a cold start")
        defaults = PG1159Config()
        config = PG1159Config(
            defaults.effective_temperature if args.teff is None else args.teff,
            defaults.logg if args.logg is None else args.logg,
            dict(args.mass_fraction) if args.mass_fraction else defaults.mass_fractions,
            args.target_name, args.quality, args.oxygen_atom,
            include_radiative_acceleration=args.radiative_acceleration,
            refine_upper_atmosphere=args.refine_upper_atmosphere)
        def progress(iteration, atmosphere, diagnostics):
            print(
                "PG1159 "
                f"{diagnostics.get('continuation_stage', 'stage')} "
                f"iteration {iteration}: "
                f"surface_flux={abs(diagnostics.get('surface_flux_ratio', float('nan')) - 1.0):.4g}, "
                f"local={diagnostics.get('maximum_relative_cell_energy_balance_residual', float('nan')):.4g}, "
                f"population={diagnostics.get('nlte_maximum_relative_population_change', float('nan')):.4g}",
                flush=True,
            )

        run = run_model(
            config, args.output, wavelength=wavelength, data=data,
            iteration_callback=progress, require_convergence=args.require_convergence,
        )
        directory = run.output_directory
        _quicklook(run.spectrum, config, kind, directory / "spectrum.png")
        print(f"Wrote {directory}")
        return
    if kind == "DAH":
        if args.restart_atmosphere is not None:
            parser.error("DAH one-shot runs are cold starts")
        defaults = DAHConfig()
        config = DAHConfig(
            effective_temperature=defaults.effective_temperature if args.teff is None else args.teff,
            logg=defaults.logg if args.logg is None else args.logg,
            magnetic_field_megagauss=(
                defaults.magnetic_field_megagauss if args.field_mg is None else args.field_mg
            ),
            field_geometry=args.geometry,
            field_angle_deg=args.field_angle,
            dipole_inclination_deg=args.inclination,
            dipole_offset_radius=tuple(args.offset),
            polarized_transfer=args.transfer,
            quality=args.quality,
        )

        def dah_progress(iteration, atmosphere, diagnostics):
            print(
                f"DAH iteration {iteration}: "
                f"flux={diagnostics.get('maximum_all_depth_total_flux_residual', float('nan')):.4g}, "
                f"local={diagnostics.get('maximum_relative_cell_energy_balance_residual', float('nan')):.4g}",
                flush=True,
            )

        run = run_model(
            config, args.output, wavelength=wavelength, data=data,
            iteration_callback=dah_progress, require_convergence=args.require_convergence,
        )
        directory = run.output_directory
        _quicklook(run.spectrum, config, kind, directory / "spectrum.png")
        if args.compare:
            import json

            from ..validation.magnetic_da import score_dah_spectrum

            scores = score_dah_spectrum(
                args.compare,
                run.spectrum.wavelength_angstrom,
                run.spectrum.surface_flux_lambda,
            )
            (directory / "validation_scores.json").write_text(
                json.dumps(scores, indent=2, default=float) + "\n", encoding="utf-8"
            )
            print(json.dumps(scores, indent=2, default=float))
        print(f"Wrote {directory}")
        return
    if kind == "D6":
        if args.restart_atmosphere is not None:
            parser.error("D6 one-shot runs are cold starts")
        defaults = D6Config()
        config = D6Config(
            effective_temperature=defaults.effective_temperature if args.teff is None else args.teff,
            logg=defaults.logg if args.logg is None else args.logg,
            abundances=dict(args.abundance) if args.abundance else defaults.abundances,
            reference_element=args.reference_element,
            quality=args.quality,
        )

        def d6_progress(iteration, atmosphere, diagnostics):
            print(
                f"D6 {diagnostics.get('solver_phase', 'solver')} iteration {iteration}: "
                f"flux={diagnostics.get('maximum_all_depth_total_flux_residual', float('nan')):.4g}, "
                f"local={diagnostics.get('maximum_relative_cell_energy_balance_residual', float('nan')):.4g}",
                flush=True,
            )

        run = run_model(config, args.output, wavelength=wavelength, data=data,
                        iteration_callback=d6_progress, require_convergence=args.require_convergence)
        directory = run.output_directory
        spectrum = run.spectrum
        _quicklook(spectrum, config, kind, directory / "spectrum.png")
        if args.compare_hollands2025:
            import json

            from ..validation.hollands_d6 import hollands2025_scores

            scores = hollands2025_scores(
                spectrum.wavelength_angstrom, spectrum.surface_flux_lambda
            )
            (directory / "hollands2025_scores.json").write_text(
                json.dumps(scores, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(scores, indent=2))
        print(f"Wrote {directory}")
        return
    if kind == "DA":
        defaults = DAConfig()
        config = DAConfig(
            effective_temperature=defaults.effective_temperature if args.teff is None else args.teff,
            logg=defaults.logg if args.logg is None else args.logg,
            quality=args.quality,
            lyman_profile_source=args.lyman_profiles,
            h3plus_partition_model=args.h3plus_partition,
        )
        composition = "hydrogen"
        compute = compute_da
    elif kind == "DB":
        defaults = DBConfig()
        config = DBConfig(
            effective_temperature=defaults.effective_temperature if args.teff is None else args.teff,
            logg=defaults.logg if args.logg is None else args.logg,
            quality=args.quality,
        )
        composition = "helium"
        compute = compute_db
    elif kind == "DAB":
        defaults = DABConfig()
        config = DABConfig(
            effective_temperature=defaults.effective_temperature if args.teff is None else args.teff,
            logg=defaults.logg if args.logg is None else args.logg,
            log_hydrogen_to_helium=args.log_h_he,
            quality=args.quality,
            lyman_profile_source=args.lyman_profiles,
        )
        composition = "mixed"
        compute = compute_dab
    else:
        defaults = DZConfig()
        config = DZConfig(
            effective_temperature=defaults.effective_temperature if args.teff is None else args.teff,
            logg=defaults.logg if args.logg is None else args.logg,
            abundances=dict(args.abundance) if args.abundance else defaults.abundances,
            log_hydrogen_abundance=args.log_h_he,
            quality=args.quality,
            dense_helium_eos=args.dense_helium_eos,
            strong_line_atomic_data=args.strong_line_atomic_data,
            lyman_profile_source=args.lyman_profiles,
        )
        composition = "helium"
        compute = compute_dz

    if args.restart_atmosphere is None:
        run = run_model(config, args.output, data=data, wavelength=wavelength,
                        research_data=args.research_data,
                        require_convergence=args.require_convergence)
        _quicklook(run.spectrum, config, kind, run.output_directory / "spectrum.png")
        print(f"Wrote {run.output_directory}")
        return
    if args.require_convergence:
        parser.error("--require-convergence applies to cold runs, not diagnostic fixed synthesis")
    if args.output.exists():
        parser.error("choose a new output directory; existing results are never overwritten")
    atmosphere = None
    if args.restart_atmosphere is not None:
        checkpoint_kwargs = {}
        if kind == "DAB":
            checkpoint_kwargs["log_hydrogen_to_helium"] = config.log_hydrogen_to_helium
        if kind == "DA":
            molecular = config.effective_temperature <= 12_000.0
            checkpoint_kwargs.update(
                include_molecules=molecular,
                include_negative_hydrogen=molecular,
                trihydrogen_ion_partition_model=(
                    None if config.h3plus_partition_model == "none"
                    else config.h3plus_partition_model
                ),
            )
        atmosphere = load_atmosphere_checkpoint(
            args.restart_atmosphere,
            config.effective_temperature,
            config.logg,
            composition,
            **checkpoint_kwargs,
        )
    result = compute(
        config,
        wavelength,
        data=data,
        initial_atmosphere=atmosphere,
        relax_atmosphere=atmosphere is None,
    )
    directory = save_model_result(result, args.output)
    _quicklook(result.spectrum, config, kind, directory / "spectrum.png")
    print(f"Wrote {directory}")
