"""Compare the provisional LTE flux handoff through complete solves.

Baseline passes use_preconditioner_flux_handoff=False; candidate passes True to the
production adaptive adapter. The benchmark does not implement a controller or
change driver acceptance. Both arms retain complete physical equations, phase
routing, final five gates, and identical shared budgets. A driver wrapper only
observes history and already-requested initial R/J. Fixed narrow public synthesis
is diagnostic: reduced fixtures do not match standard request fingerprints.

Run under an external wall timeout with all six numerical thread variables=1.
Example from the checkout (no atmosphere work occurs for --help):
PYTHONPATH=src python benchmarks/benchmark_preconditioner_flux_handoff.py \
  --case d6 --depths 16 --metal-lines 128 --checkpoint accepted-state.npz \
  --samples 2 --preconditioner-steps 8 --steps 24 --output fresh-report.json
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch
import warnings

ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_OPTION = "use_preconditioner_flux_handoff"
THREADS = ("OPENWD_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_inventory():
    paths = list((ROOT / "src/wd_spectra").rglob("*.py"))
    paths += list((ROOT / "csrc").glob("*.[ch]"))
    paths += [ROOT / "setup.py", ROOT / "MANIFEST.in", Path(__file__),
              ROOT / "benchmarks/benchmark_lte_probe_reuse.py"]
    def key(path):
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return {key(path): sha(path) for path in sorted(set(paths)) if path.is_file()}


def checkpoint_fixture(args):
    """Rebuild public D6 materials on a sampled, recorded accepted state.

    Only the public adapter's initial T, mass and optical-depth inputs are
    supplied. It recomputes the EOS and all materials at the reduced grid and
    line budget; checkpoint diagnostics do not transfer to this fixture.
    """
    if args.checkpoint is None:
        return physical_fixture(args.case, args.depths, args.continuum, args.metal_lines), None
    with np.load(args.checkpoint, allow_pickle=False) as saved:
        size = saved["temperature"].size
        if args.depths > size:
            raise ValueError("Checkpoint sampling cannot add depths")
        indices = np.rint(np.linspace(0, size - 1, args.depths)).astype(int)
        fields = ("rosseland_optical_depth", "column_mass", "temperature", "gas_pressure",
                  "mass_density", "neutral_h_density", "proton_density", "electron_density")
        seed = Atmosphere(effective_temperature=float(saved["effective_temperature"]),
            logg=float(saved["logg"]), metadata=json.loads(str(saved["atmosphere_metadata_json"])),
            **{name: saved[name][indices].copy() for name in fields})

    def supplied_seed(teff, logg, *positional, **keywords):
        if teff != seed.effective_temperature or logg != seed.logg:
            raise ValueError("Checkpoint and public D6 preset parameters differ")
        return seed

    with patch.object(d6, "gray_d6_atmosphere", supplied_seed):
        fixture = physical_fixture(args.case, args.depths, args.continuum, args.metal_lines)
    for name in ("temperature", "column_mass", "rosseland_optical_depth"):
        np.testing.assert_array_equal(getattr(fixture["seed"], name), getattr(seed, name))
    return fixture, dict(path=str(args.checkpoint.resolve()),
        sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        original_depths=size, sampling_indices=indices.tolist(),
        scope="Checkpoint-derived reduced diagnostic; EOS/materials rebuilt. Not a public cold start.")

def physical_errors(ev):
    p = ev.payload
    target = float(p["target_flux"]) if "target_flux" in p else None
    if target is None:
        from wd_spectra.constants import STEFAN_BOLTZMANN
        target = STEFAN_BOLTZMANN * p["atmosphere"].effective_temperature**4
    return {
        "flux": float(np.max(np.abs(p["total_flux_interface"] / target - 1))),
        "energy": float(np.max(np.abs(p["cell_energy_balance_relative_residual"]))),
        "source_closure": float(p["scattering_source_maximum_relative_residual"]),
        "phase_residual": float(np.max(np.abs(ev.residual))),
    }


def options(fixture, args, candidate):
    controls = dict(fixture["options"], max_iterations=args.steps,
        maximum_convective_preconditioner_iterations=args.preconditioner_steps,
        use_convective_gradient_preconditioner=True,
        resume_supplied_structure_in_formal_flux_phase=False,
        iteration_callback=None)
    # The handoff is the production default; the baseline disables it
    # explicitly so captured adapter options cannot silently change either arm.
    controls[ALGORITHM_OPTION] = bool(candidate)
    return controls


def synthesize(case, atmosphere, args, wave):
    from wd_spectra.models import (DBConfig, DAZConfig, D6Config,
                                  compute_db, compute_daz, compute_d6)
    presets = {
        "db": (compute_db, DBConfig(quality="quick")),
        "daz": (compute_daz, DAZConfig(quality="quick", lyman_profile_source="stark",
            structure_maximum_metal_lines=args.metal_lines,
            formal_maximum_metal_lines=args.metal_lines)),
        "d6": (compute_d6, D6Config(quality="quick",
            structure_maximum_metal_lines=args.metal_lines,
            formal_maximum_metal_lines=args.metal_lines)),
    }
    compute, config = presets[case]
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = compute(config, wave, initial_atmosphere=atmosphere,
                         relax_atmosphere=False)
    return result, [str(w.message) for w in records]


def measure(fixture, args, candidate, flush, row, arrays, label):
    driver = adaptive.solve_trust_region_newton
    segments = []
    counts = {}
    last = {}
    request_counts = {}
    row["all_structure_requests"] = request_counts
    row["material_and_transfer_calls"] = counts
    phase_stage = "solve"
    source, source_start = inspect.getsourcelines(adaptive.solve_adaptive_lte_structure)
    definitions = [node for node in ast.walk(ast.parse("".join(source)))
                   if isinstance(node, ast.FunctionDef) and node.name == "evaluate_log_temperature"]
    if len(definitions) != 2:
        raise RuntimeError("Expected exactly two production evaluator definitions")
    timed_line = source_start + max(n.lineno for n in definitions) - 1
    source_path = str(Path(adaptive.__file__).resolve())

    def profile(frame, event, arg):
        if (event == "call" and frame.f_code.co_filename == source_path
            and frame.f_code.co_firstlineno == timed_line):
            phase = frame.f_locals["solver_phase"]
            key = phase_stage + ":" + phase + (":J" if frame.f_locals["need_jacobian"] else ":R")
            request_counts[key] = request_counts.get(key, 0) + 1

    def counted(name, function):
        def call(*positional, **keywords):
            key = phase_stage + ":" + name
            counts[key] = counts.get(key, 0) + 1
            return function(*positional, **keywords)
        return call

    def segment(initial, evaluate, **controls):
        index = len(segments)
        if index == 0:
            arrays[label+"_initial_driver_state"] = initial.copy()
            row["initial_driver_noncallback_controls"] = {
                key: value for key, value in controls.items()
                if key != "accepted_state_handoff" and not callable(value)}
        observed_initial = set()

        def observed(current, need):
            ev = evaluate(current, need)
            key = "J" if need else "R"
            if index == 0 and key not in observed_initial:
                np.testing.assert_array_equal(current, initial)
                arrays[label+"_initial_driver_"+key] = (ev.jacobian if need else ev.residual).copy()
                observed_initial.add(key)
            return ev
        original_callback = controls.get("callback")

        def callback(record, current, ev):
            row["history"].append(dict(segment=index, **asdict(record), **physical_errors(ev)))
            if original_callback is not None:
                original_callback(record, current, ev)

        # Observation only: preserve any production accepted-state hook and
        # all controls. No benchmark controller or convergence predicate.
        controls = dict(controls, callback=callback)
        result = driver(initial, observed, **controls)
        if result.diagnostics.terminal_reason == "accepted-state-phase-handoff":
            assert index == 0 and candidate and not result.converged
        segments.append(dict(index=index, iterations=result.iterations,
            converged=result.converged, diagnostics=asdict(result.diagnostics),
            production_accepted_state_hook_present=controls.get("accepted_state_handoff") is not None,
            **physical_errors(result.evaluation)))
        last.update(state=result.state.copy(), evaluate=observed, controls=controls)
        row["segments"] = segments
        flush()
        return result

    model_options = options(fixture, args, candidate)
    row["algorithm_option"] = dict(name=ALGORITHM_OPTION,
        explicitly_passed=ALGORITHM_OPTION in model_options,
        effective_value=bool(model_options.get(ALGORITHM_OPTION, False)))
    if (not model_options["enforce_local_energy_balance"]
        or model_options["mixing_length_alpha"] is None):
        raise RuntimeError("This experiment requires ordinary convective LTE with energy enforcement")
    for name in ("with_temperature", "true_absorption", "scattering_opacity",
                 "rosseland_opacity", "thermodynamics"):
        model_options[name] = counted(name, model_options[name])
    if sys.getprofile() is not None:
        raise RuntimeError("Refuse to overwrite an existing Python profiler")
    start = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(adaptive, "solve_trust_region_newton", segment))
        for module in (d6, metals):
            stack.enter_context(patch.object(module, "metal_line_mass_absorption_coefficient",
                counted("metal_opacity_build", module.metal_line_mass_absorption_coefficient)))
        for name in ("coherent_scattering_feautrier_field", "feautrier_radiation_field",
                     "cancellation_safe_field", "cancellation_safe_scalar_field",
                     "integrated_coherent_scattering_feautrier_state_response",
                     "integrated_feautrier_interface_state_response",
                     "cancellation_safe_response", "integrated_radiative_cell_energy_state_response"):
            stack.enter_context(patch.object(adaptive, name, counted(name, getattr(adaptive, name))))
        sys.setprofile(profile)
        try:
            atmosphere = adaptive.solve_adaptive_lte_structure(
                fixture["seed"], fixture["wavelength"], **model_options)
            row["solve_seconds"] = time.perf_counter() - start
            row["solve_timer_scope"] = "Complete solver including Python profiling and per-segment report writes; setup, fresh diagnostic and synthesis excluded"
            row["canonical_certificate"] = equilibrium_certificate(atmosphere.metadata,
                flux_tolerance=model_options["flux_tolerance"],
                temperature_tolerance=model_options["temperature_tolerance"])
            assert row["canonical_certificate"] == atmosphere.metadata["equilibrium_certificate"]
            row["final_metadata"] = atmosphere.metadata
            metadata_key = "convective_preconditioner_flux_handoff"
            hook_presence = [segment["production_accepted_state_hook_present"] for segment in segments]
            if candidate:
                handoff_metadata = atmosphere.metadata.get(metadata_key)
                if not isinstance(handoff_metadata, dict):
                    raise RuntimeError("Candidate must retain current-solve handoff metadata")
                assert handoff_metadata["requested"] is True and handoff_metadata["used"] is True
                assert handoff_metadata["certifies_equilibrium"] is False
                assert hook_presence == [True] + [False] * (len(segments)-1)
                triggered = handoff_metadata["triggered"]
                assert isinstance(triggered, bool)
                assert triggered == (segments[0]["diagnostics"]["terminal_reason"] == "accepted-state-phase-handoff")
                if triggered:
                    assert segments[0]["converged"] is False
                row["production_algorithm_metadata"] = handoff_metadata
            else:
                assert metadata_key not in atmosphere.metadata, "Baseline inherited current-solve handoff metadata"
                assert not any(hook_presence), "Disabled baseline must not install an accepted-state hook"
                row["production_algorithm_metadata"] = None
            row["production_algorithm_contract_verified"] = True
            phase_stage = "fresh-final-diagnostic"
            row["status"] = phase_stage
            flush()
            began = time.perf_counter()
            ev = last["evaluate"](last["state"], True)
            for name in ("temperature", "column_mass", "gas_pressure", "mass_density", "electron_density"):
                np.testing.assert_array_equal(getattr(ev.payload["atmosphere"], name), getattr(atmosphere, name))
            builder = last["controls"].get("step_builder")
            if atmosphere.metadata["local_energy_completion_used"]:
                # This is the final production unpenalized energy direction,
                # with freshly frozen physical weights; no truncation policy.
                if builder is None:
                    raise RuntimeError("Expected final energy step builder")
                raw = builder(last["state"], ev, ev.jacobian, np.inf)
                linear_policy = "production unpenalized energy builder"
            else:
                scale = np.maximum(np.max(abs(ev.jacobian), axis=1), np.finfo(float).tiny)
                raw, _, rank, singular = np.linalg.lstsq(ev.jacobian/scale[:, None],
                    -ev.residual/scale, rcond=None)
                linear_policy = dict(row_equilibrated=True, rcond=None,
                    rank=int(rank), singular_values=singular.tolist(),
                    scope="Additive unpenalized/default-rcond diagnostic, distinct from any production regularization/truncation")
            if hasattr(raw, "direction"):
                raw = raw.direction
            fresh_step = float(last["controls"]["step_measure"](last["state"], last["state"]+raw))
            fresh = dict(**physical_errors(ev), boundary=float(ev.payload["lower_boundary_absorption_escape_bound"]),
                unrestricted_log_temperature_correction=fresh_step, linear_policy=linear_policy)
            fresh["five_measured_gates_pass"] = bool(fresh["flux"] < model_options["flux_tolerance"]
                and fresh["energy"] < model_options["flux_tolerance"]
                and fresh_step < model_options["temperature_tolerance"]
                and fresh["source_closure"] < 1e-6 and fresh["boundary"] < model_options["flux_tolerance"])
            row["fresh_final"] = fresh
            row["fresh_final_scope"] = (
                "Production evaluator requested at exact returned anchor; exact-state "
                "cached data may be reused. Separate unpenalized correction, "
                "not an independent material or transfer implementation."
            )
            row["fresh_final_seconds"] = time.perf_counter()-began
            for name in ("temperature", "gas_pressure", "mass_density", "column_mass",
                         "rosseland_optical_depth", "electron_density"):
                arrays[label+"_final_"+name] = np.asarray(getattr(atmosphere, name)).copy()
            arrays[label+"_final_state"] = last["state"]
            arrays[label+"_final_R"] = ev.residual.copy()
            arrays[label+"_final_J"] = ev.jacobian.copy()
            arrays[label+"_final_raw_direction"] = raw.copy()
            arrays[label+"_final_flux"] = ev.payload["total_flux_interface"].copy()
            arrays[label+"_final_energy"] = ev.payload["cell_energy_balance_relative_residual"].copy()
        finally:
            sys.setprofile(None)
    row["all_structure_requests"] = request_counts
    row["material_and_transfer_calls"] = counts
    row["source_profiler_restored"] = sys.getprofile() is None
    row["physical_arrays_finite_positive"] = bool(all(
        np.all(np.isfinite(getattr(atmosphere, name))) and np.all(getattr(atmosphere, name)>0)
        for name in ("temperature", "gas_pressure", "mass_density", "column_mass", "electron_density")))
    row["qualified_reduced_structure"] = bool(row["canonical_certificate"]["verified"]
        and row["fresh_final"]["five_measured_gates_pass"] and row["physical_arrays_finite_positive"])
    row["request_count_scope"] = "All timed production evaluator requests including initialization, thermal sweeps, completion and fresh final diagnostics. Python profiler overhead included."
    return atmosphere


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("db", "daz", "d6"), required=True)
    parser.add_argument("--depths", type=int, default=8)
    parser.add_argument("--continuum", type=int, default=80)
    parser.add_argument("--metal-lines", type=int, default=16)
    parser.add_argument("--preconditioner-steps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path,
        help="D6-only accepted checkpoint; sampled grid and rebuilt materials are diagnostic")
    parser.add_argument("--source-root", type=Path,
        help="Explicit checkout root, useful when reviewing this proposed file outside benchmarks")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (not 3 <= args.depths <= 24 or args.continuum < 80 or args.metal_lines < 0
        or not 2 <= args.preconditioner_steps <= 24
        or not 4 <= args.steps <= 64 or not 1 <= args.samples <= 3):
        parser.error("Invalid reduced-grid or bounded iteration/sample controls")
    if args.checkpoint is not None and args.case != "d6":
        parser.error("Checkpoint fixtures are D6-only")
    if any(path.exists() for path in (args.output, args.output.with_suffix(".npz"),
        args.output.with_suffix(".json.tmp"), args.output.with_suffix(".npz.tmp"))):
        parser.error("Outputs and temporary paths must be fresh")
    if any(os.environ.get(name) != "1" for name in THREADS):
        parser.error("Pin all six numerical thread variables to one")
    global ROOT, np, adaptive, d6, metals, physical_fixture, Atmosphere, equilibrium_certificate
    ROOT = (args.source_root.resolve() if args.source_root is not None else ROOT)
    if not (ROOT / "src/wd_spectra/adaptive_structure.py").is_file():
        parser.error("Expected an OpenWD checkout; use --source-root for scratch proposals")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "benchmarks"))
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    from wd_spectra import adaptive_structure as adaptive, d6, metals
    from wd_spectra._convergence import equilibrium_certificate
    from wd_spectra._provenance import model_data_identity
    from wd_spectra.models.common import ModelData
    from wd_spectra.atmosphere import Atmosphere
    from benchmark_lte_probe_reuse import physical_fixture, runtime_metadata
    if Path(adaptive.__file__).resolve() != ROOT / "src/wd_spectra/adaptive_structure.py":
        raise RuntimeError("Imported OpenWD does not match requested source root")
    arrays = {}
    report = dict(status="running", protocol=vars(args), records=[],
        DQ_excluded=True, DQ_model_exercised=False,
        production_sources_changed_relative_to_merged_baseline=True,
        shared_algorithm_default_enabled=True,
        algorithm_option=ALGORITHM_OPTION,
        scope="Complete reduced LTE solves comparing the production algorithm option disabled and enabled and fixed diagnostic synthesis; not a released-quality cold model or independent grid validation")
    report["protocol"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    def flush():
        save(args.output, report)

    def inventory():
        return source_inventory()

    def save_arrays():
        with args.output.with_suffix(".npz.tmp").open("wb") as stream:
            np.savez(stream, **arrays)
        args.output.with_suffix(".npz.tmp").replace(args.output.with_suffix(".npz"))
    try:
        parameter = inspect.signature(adaptive.solve_adaptive_lte_structure).parameters.get(ALGORITHM_OPTION)
        if parameter is None or parameter.default is not True:
            raise RuntimeError("Production algorithm option must exist with exact default True")
        report["production_default_true_verified"] = True
        report["runtime"] = runtime_metadata()
        native_path = report["runtime"]["native_extension"]
        if native_path is None:
            raise RuntimeError("Native extension required for this experiment")
        report["native_sha256"] = sha(native_path)
        report["source_before"] = inventory()
        data = ModelData.default()
        report["data_before"] = model_data_identity(data)
        fixture, checkpoint = checkpoint_fixture(args)
        report["checkpoint"] = checkpoint
        report["wavelength_points"] = len(fixture["wavelength"])
        report["tolerances"] = {k: fixture["options"][k] for k in ("flux_tolerance", "temperature_tolerance")}
        report["common_budget_controls"] = dict(max_iterations=args.steps,
            maximum_convective_preconditioner_iterations=args.preconditioner_steps,
            resume_supplied_structure_in_formal_flux_phase=False,
            maximum_formal_flux_continuations=fixture["options"].get("maximum_formal_flux_continuations", 2),
            energy_thermal_sweeps=fixture["options"].get("energy_thermal_sweeps", 40))
        spectral_wave = np.unique(np.r_[np.linspace(3600, 7000, 128),
            np.linspace(3928, 3940, 17), np.linspace(4470, 4475, 17),
            np.linspace(4858, 4865, 17), np.linspace(6558, 6568, 21)])
        arrays["structure_wave"] = fixture["wavelength"].copy()
        arrays["spectral_wave"] = spectral_wave
        for sample in range(args.samples):
            order = (False, True) if sample % 2 == 0 else (True, False)
            for candidate in order:
                variant = "handoff" if candidate else "baseline"
                label = f"sample{sample+1}_{variant}"
                row = dict(sample=sample+1, variant=variant, status="setting-up", history=[])
                report["records"].append(row)
                flush()
                began = time.perf_counter()
                fresh, identity = checkpoint_fixture(args)
                assert identity == checkpoint
                for name in ("temperature", "column_mass", "gas_pressure", "mass_density", "rosseland_optical_depth"):
                    np.testing.assert_array_equal(getattr(fresh["seed"], name), getattr(fixture["seed"], name))
                np.testing.assert_array_equal(fresh["wavelength"], fixture["wavelength"])
                row["setup_seconds"] = time.perf_counter()-began
                row["identical_seed_and_grid_verified"] = True
                row["status"] = "solving"
                flush()
                atmosphere = measure(fresh, args, candidate, flush, row, arrays, label)
                save_arrays()
                row["status"] = "synthesizing"
                flush()
                began = time.perf_counter()
                result, notices = synthesize(args.case, atmosphere, args, spectral_wave)
                row["synthesis_seconds"] = time.perf_counter()-began
                row["synthesis_warnings"] = notices
                row["fixed_synthesis_request_verified"] = result.atmosphere.metadata.get("fixed_synthesis_request_verified")
                row["synthesis_scope"] = "Public fixed synthesis on reduced structure; request mismatch correctly retained; not released-quality or bolometric grid validation"
                flux = result.spectrum.surface_flux_lambda
                row["spectrum_finite_positive"] = bool(np.all(np.isfinite(flux)) and np.all(flux > 0))
                arrays[label+"_spectrum"] = flux.copy()
                row["status"] = "completed"
                save_arrays()
                flush()
        comparisons=[]
        for sample in range(args.samples):
            pair = [row for row in report["records"] if row["sample"] == sample+1]
            assert len(pair) == 2
            assert pair[0]["initial_driver_noncallback_controls"] == pair[1]["initial_driver_noncallback_controls"]
            for key in ("state", "R", "J"):
                old = f"sample{sample+1}_baseline_initial_driver_"+key
                new = f"sample{sample+1}_handoff_initial_driver_"+key
                assert (old in arrays) == (new in arrays), "Initial driver request structure differs"
                if old in arrays:
                    np.testing.assert_array_equal(arrays[old], arrays[new])
            baseline = arrays[f"sample{sample+1}_baseline_spectrum"]
            candidate = arrays[f"sample{sample+1}_handoff_spectrum"]
            comparisons.append(dict(sample=sample+1,
                maximum_pointwise_relative_flux_difference=float(np.max(abs(candidate/baseline-1))),
                maximum_absolute_flux_difference=float(np.max(abs(candidate-baseline))),
                independent_grid_or_spectral_accuracy_qualified=False))
        report["spectral_comparisons"] = comparisons
        report["identical_initial_driver_states_and_measured_R_J_verified"] = True
        report["identical_initial_driver_noncallback_controls_verified"] = True
        report["baseline_disables_algorithm_option"] = all(
            row["algorithm_option"]["explicitly_passed"]
            and row["algorithm_option"]["effective_value"] is False
            for row in report["records"] if row["variant"] == "baseline")
        assert report["baseline_disables_algorithm_option"]
        report["source_after"] = inventory()
        report["data_after"] = model_data_identity(data)
        report["identities_unchanged"] = bool(report["source_before"] == report["source_after"]
            and report["data_before"] == report["data_after"] and sha(native_path) == report["native_sha256"])
        if not report["identities_unchanged"]:
            raise RuntimeError("Source/data/native identity changed")
        report["array_sha256"] = sha(args.output.with_suffix(".npz"))
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        flush()
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
