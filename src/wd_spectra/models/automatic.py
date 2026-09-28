"""Automatic, auditable workflow dispatch with isolated cool-model workers.

Existing compute_da/db/dab/dz remain explicit presets. run_model is the
automatic file-oriented entry point. Packaged cool callbacks run in a
child process; DQ's package-local adapters also run in an isolated worker.
No failed solver selects another EOS, no saved atmosphere is discovered
automatically, and no flux is scaled.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings
import numpy as np

from .common import (
    ModelData,
    AtmosphereConvergenceWarning,
    _jsonable,
    save_model_result,
    validate_wavelength,
)
from .stellar import (
    DAConfig,
    DBConfig,
    DABConfig,
    DZConfig,
    compute_da,
    compute_db,
    compute_dab,
    compute_dz,
)
from .selection import select_physics, PhysicsSelectionPolicy, PhysicsSelection
from .daz import DAZConfig, compute_daz
from .dq import DQConfig, compute_dq
from .hot import DOConfig, DAOConfig, compute_do, compute_dao
from .pg1159 import PG1159Config, compute_pg1159
from .d6 import D6Config, compute_d6
from ..spectrum import Spectrum


@dataclass(frozen=True)
class ModelRun:
    """Artifacts with explicit physics and certification, not a reconstructed EOS."""

    output_directory: Path
    selection: PhysicsSelection
    spectrum_path: Path
    convergence_verified: bool

    @property
    def spectrum(self):
        values = np.loadtxt(self.spectrum_path)
        return Spectrum(
            values[:, 0],
            values[:, 1],
            dict(
                flux_convention="surface F_lambda",
                wavelength_medium="vacuum",
                flux_unit="erg s^-1 cm^-2 Angstrom^-1",
                source_path=str(self.spectrum_path),
                convergence_verified=self.convergence_verified,
                selected_physics=self.selection.workflow,
            ),
        )


def _cool_commands(config, selection, directory):
    # These constraints are limitations of the preserved successful drivers,
    # not assertions that the physics ceases to apply elsewhere.
    if config.logg != 8.0 or config.quality != "production":
        raise ValueError(
            'The automatic cool workflow currently requires logg=8 and quality="production"; no different gravity/resolution is substituted'
        )
    defaults = type(config)()
    for key in (
        "mixing_length_alpha",
        "atmosphere_solver",
        "neutral_broadening",
    ):
        if getattr(config, key) != getattr(defaults, key):
            raise ValueError(
                f"Automatic cool workflow does not implement an override of {key}; use an explicit preset for controlled experiments"
            )
    temperature = str(int(config.effective_temperature))
    if float(temperature) != config.effective_temperature:
        raise ValueError(
            "Preserved cool drivers currently require integer-K input; no temperature rounding is applied"
        )
    worker = directory / "worker"
    if selection.workflow == "dense-db":
        # Importing this command factory is unnecessary: keep the recipe in
        # one canonical script and execute that script in its isolated worker.
        return [
            [
                sys.executable,
                "-m", "wd_spectra._cool.run_cool_db",
                temperature,
                "--output-root",
                str(worker),
                "--allow-unqualified",
            ]
        ], worker / temperature / "experimental-spectrum.txt"
    if config.h2_he_cia_path is not None:
        raise ValueError(
            "Automatic molecular recipe uses checksum-pinned research data; supply its directory through research_data, not an unverified CIA override"
        )
    for key in (
        "lyman_profile_source",
        "allard_minimum_effective_temperature",
        "balmer_self_broadening_prescription",
        "balmer_self_broadening_truncation_closure",
    ):
        if getattr(config, key) != getattr(defaults, key):
            raise ValueError(
                f"Automatic molecular workflow cannot ignore an explicit {key} override"
            )
    command = [
        sys.executable,
        "-m", "wd_spectra._cool.run_molecular_dab_mass_experiment",
        "--physical-detuning-lyman",
        "--consistent-stark-edge",
        temperature,
        "--log-h-he",
        str(config.log_hydrogen_to_helium),
        "--physical-only",
        "--stable-transfer",
        "--step-method",
        "nonlinear-convection-current-energy",
        "--inner-scaling",
        "svd",
        "--inner-max-evaluations",
        "1000",
        "--max-iterations",
        "35",
        "--no-continuations",
        "--state-sum-h2",
        "--output-root",
        str(worker),
    ]
    command.extend(["--pseudo-sweeps", "40", "--relax-bottom"])
    return [command], worker / temperature / "spectrum.txt"


def run_model(
    config,
    output_directory,
    *,
    data=None,
    research_data=None,
    require_convergence=False,
    policy=PhysicsSelectionPolicy(),
    wavelength=None,
    iteration_callback=None,
):
    """Choose physics automatically and write a reproducible model run.

    Cool recipes currently require production resolution and logg=8. Public model runs are cold starts; saved atmospheres and
    neighboring stellar models are not accepted. Selection uses material diagnostics, not the list of tested Teff
    points. Failed/uncertified outputs are retained and warned about; optional
    require_convergence makes an uncertified completed run raise an error.
    """
    if not isinstance(require_convergence, bool):
        raise TypeError("require_convergence must be a bool")
    wave = None if wavelength is None else validate_wavelength(wavelength)
    data = ModelData.default() if data is None else data
    directory = Path(output_directory).expanduser().resolve()
    if directory.exists():
        raise FileExistsError(
            f"Choose a new output directory; never overwrite {directory}"
        )
    print(
        "OpenWD: constructing the provisional material screen (not a converged atmosphere)",
        flush=True,
    )
    selection = select_physics(config, data=data, policy=policy)
    print(
        f"OpenWD physics selection: {selection.workflow}: {selection.reason}",
        flush=True,
    )
    print(json.dumps(_jsonable(asdict(selection)), sort_keys=True), flush=True)
    commands = []
    environment = os.environ.copy()
    # Every worker imports this installed package and runs in its own process.
    isolated_worker = selection.workflow in ("dense-db", "molecular-dab")
    if isolated_worker:
        if wave is not None or iteration_callback is not None:
            raise ValueError("Cool workers currently use their qualified output grid and stdout progress")
        commands, spectrum_path = _cool_commands(config, selection, directory)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        environment["OPENWD_DATA"] = str(data.root)
        if research_data is not None:
            environment["OPENWD_RESEARCH_DATA"] = str(
                Path(research_data).expanduser().resolve()
            )
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "OPENWD_NUM_THREADS",
        ):
            environment[key] = "1"
        if selection.workflow == "molecular-dab":
            # Validation only; no download or alternate data lookup.
            subprocess.run(
                [sys.executable, "-m", "wd_spectra._cool.check_data"],
                env=environment,
                check=True,
            )
    directory.mkdir(parents=True)
    manifest = dict(
        schema=1,
        request=_jsonable(config),
        selection=_jsonable(asdict(selection)),
        initial_checkpoint=None,
        cold_start=True,
        data_directory=str(data.root),
        research_data_directory=environment.get("OPENWD_RESEARCH_DATA"),
        independent_depth_grid_validation=False,
        full_physics_validation=False,
        commands=commands,
        status="running",
        physics_changed_after_failure=False,
    )

    def record():
        (directory / "model-run.json").write_text(json.dumps(manifest, indent=2) + "\n")

    record()
    try:
        if not isolated_worker:
            compute = {
                DOConfig: compute_do,
                DAOConfig: compute_dao,
                PG1159Config: compute_pg1159,
                D6Config: compute_d6,
                DAConfig: compute_da,
                DAZConfig: compute_daz,
                DQConfig: compute_dq,
                DBConfig: compute_db,
                DABConfig: compute_dab,
                DZConfig: compute_dz,
            }[type(config)]

            started = time.monotonic()

            def progress(iteration, atmosphere, diagnostic):
                print(
                    json.dumps(
                        dict(
                            iteration=iteration,
                            elapsed_seconds=time.monotonic() - started,
                            phase=diagnostic.get("solver_phase"),
                            local_energy=diagnostic.get(
                                "maximum_relative_cell_energy_balance_residual"
                            ),
                            flux=diagnostic.get(
                                "maximum_all_depth_total_flux_residual"
                            ),
                            correction=diagnostic.get(
                                "maximum_log_temperature_correction"
                            ),
                        )
                    ),
                    flush=True,
                )

            if isinstance(config, DQConfig):
                if iteration_callback is not None:
                    raise ValueError("DQ worker reports progress to stdout; callbacks are not supported")
                options = {} if wave is None else {"wavelength": wave}
                result = compute(config, data=data, output_directory=directory / 'worker', **options)
            else:
                options = dict(
                    data=data,
                    iteration_callback=progress if iteration_callback is None else iteration_callback,
                )
                if not isinstance(config, PG1159Config):
                    options["initial_atmosphere"] = None
                if wave is not None:
                    options["wavelength"] = wave
                result = compute(
                    config,
                    **options,
                )
            save_model_result(result, directory)
            spectrum_path = directory / "spectrum.txt"
            qualified = result.metadata["atmosphere_convergence_status"] == "converged"
        else:
            for command in commands:
                subprocess.run(command, env=environment, check=True)
            temperature = str(int(config.effective_temperature))
            if selection.workflow == "dense-db":
                qualified = (
                    json.loads(
                        (
                            directory
                            / "worker/audit"
                            / temperature
                            / "qualification.json"
                        ).read_text()
                    )["numerically_qualified_for_declared_experimental_physics"]
                    is True
                )
            else:
                check = subprocess.run(
                    [
                        sys.executable,
                        "-m", "wd_spectra._cool.check_molecular_dab_result",
                        str(directory / "worker" / temperature),
                    ],
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                print(check.stdout, end="", flush=True)
                if check.stderr:
                    print(check.stderr, file=sys.stderr, end="", flush=True)
                try:
                    report = json.loads(check.stdout.strip().splitlines()[-1])
                except (ValueError, IndexError) as exc:
                    raise RuntimeError(
                        "Molecular qualification checker did not return a result"
                    ) from exc
                if check.returncode not in (0, 1) or not isinstance(
                    report.get("structure_grid_convergence_verified"), bool
                ):
                    raise RuntimeError(
                        "Molecular qualification checker failed to execute"
                    )
                qualified = report["structure_grid_convergence_verified"]
                if (check.returncode == 0) != qualified:
                    raise RuntimeError("Inconsistent molecular qualification report")
        spectrum_values = np.loadtxt(spectrum_path)
        if (
            spectrum_values.ndim != 2
            or spectrum_values.shape[0] < 2
            or spectrum_values.shape[1] != 2
            or not np.all(np.isfinite(spectrum_values))
            or np.any(spectrum_values[:, 0] <= 0)
            or np.any(np.diff(spectrum_values[:, 0]) <= 0)
            or np.any(spectrum_values[:, 1] < 0)
        ):
            raise ValueError(
                "Worker did not produce a finite, ordered, nonnegative two-column spectrum"
            )
        manifest.update(
            status="completed",
            convergence_verified=qualified,
            spectrum_path=str(spectrum_path),
        )
        record()
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record()
        raise
    if not qualified:
        message = f"Model completed without full equilibrium certification; exploratory outputs retained at {directory}"
        if require_convergence:
            raise RuntimeError(message)
        warnings.warn(message, AtmosphereConvergenceWarning, stacklevel=2)
    return ModelRun(directory, selection, spectrum_path, qualified)
