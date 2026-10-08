"""One reduced real DO/DAO Jacobian with canonical physics reuse on or off.

This uses only the source in this checkout. ``baseline`` disables
reuse_accepted_state in that same source; it is not an immutable release
comparison. ``candidate`` enables it. Both otherwise use the same real bundled
atoms, deterministic gray seed (60000 K/logg=8/tau_max=1000), geometric 80-point
input wavelength grid, He I 14/He II 3 and, for DAO, H 3 with log(H/He)=2.
Quality quick gives two angles. Temperatures and populations remain joint
unknowns; "fixed state" means a single initial anchor, not an atmosphere solve.

Run sequentially on an idle machine with numerical threads set externally:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENWD_NUM_THREADS=1 PYTHONPATH=src \
    python benchmarks/benchmark_hot_cache_reuse.py --variant baseline \
    --depths 8 --output /tmp/do-baseline.json
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENWD_NUM_THREADS=1 PYTHONPATH=src \
    python benchmarks/benchmark_hot_cache_reuse.py --variant candidate \
    --depths 8 --output /tmp/do-candidate.json

Add --mixed for DAO. Alternate variants in fresh processes and compare seed,
source/data/native/thread identities and state/residual/J arrays before timing.
Each invocation performs one untimed Jacobian warmup, constructs fresh measured
equations, and resets call counters before the one measured evaluate request.
Imports, seed/initial-state setup, checkpoint IO and hashing are outside that
timer. Peak RSS includes warmup and retained traces, not just the measured call.
No streaming, custom proposal, cold solve or convergence claim is included.
Atomic JSON/NPZ checkpoints preserve incomplete progress under an outer cap.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time


THREAD_FIELDS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OPENWD_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
SEED_FIELDS = ("effective_temperature", "logg", "rosseland_optical_depth", "column_mass",
               "temperature", "gas_pressure", "mass_density", "neutral_h_density",
               "proton_density", "electron_density")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode()); digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode()); digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=("baseline", "candidate"), default="candidate")
    parser.add_argument("--depths", type=int, choices=(8, 16), default=8)
    parser.add_argument("--mixed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root, output = Path(__file__).resolve().parents[1], args.output.resolve()
    numerical_path, seed_path = output.with_suffix(".npz"), output.with_suffix(".seed.npz")
    if output.suffix != ".json" or any(p.exists() for p in (output, numerical_path, seed_path)):
        parser.error("output must be a fresh .json path with fresh sibling NPZ paths")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = dict(schema="openwd-hot-cache-reuse-jacobian-v1", status="setup-started",
        created_utc=datetime.now(timezone.utc).isoformat(), command_line=sys.argv,
        benchmark_sha256=file_digest(__file__), source_root=str(source_root), source_selection="current checkout only",
        variant=args.variant, reuse_accepted_state=args.variant == "candidate", mixed=args.mixed, depths=args.depths,
        effective_temperature=60000., logg=8., helium_i_bound_levels=14, helium_ii_bound_levels=3,
        hydrogen_bound_levels=3 if args.mixed else 0, nlte_fraction=1., fixed_temperature=False,
        baseline_semantics="Same candidate source with canonical reuse disabled; not the immutable released source",
        scope="One joint reduced real-atom initial Jacobian; no nonlinear solve or physical model qualification",
        untimed_warmups=1, cold_atmosphere_solves=0, custom_proposal=False,
        full_physics_validation=False, independent_grid_validation=False,
        seed_npz=str(seed_path), numerical_npz=str(numerical_path),
        thread_environment={name: os.environ.get(name) for name in THREAD_FIELDS}, checkpoint_io_seconds=0.,
        timing_scope="One measured HotEquations.evaluate(state,True); excludes setup, warmup, IO/hashing and instrumentation installation; lightweight call-count wrappers included",
        peak_rss_scope="Entire fresh process, including warmup and retained trace arrays; macOS ru_maxrss bytes/Linux KiB converted to bytes")
    arrays = {}

    def atomic_npz(path, values):
        import numpy as np
        temporary = path.with_name(path.name + ".writing")
        with temporary.open("wb") as stream:
            np.savez(stream, **values)
        temporary.replace(path)

    def persist(*, numerical=False):
        began = time.perf_counter()
        if numerical:
            atomic_npz(numerical_path, arrays)
            report["persisted_array_names"] = sorted(arrays)
        report.update(elapsed_seconds=time.perf_counter()-started, peak_rss_bytes=peak_rss_bytes(),
                      retained_array_bytes=sum(a.nbytes for a in arrays.values()))
        temporary = output.with_name(output.name + ".writing")
        temporary.write_text(json.dumps(json_value(report), indent=2, allow_nan=False) + "\n")
        temporary.replace(output)
        report["checkpoint_io_seconds"] += time.perf_counter()-began

    persist()
    try:
        sys.path.insert(0, str(source_root / "src"))
        import numpy as np
        import scipy
        import wd_spectra._hot_structure as hot
        import wd_spectra.opacity as opacity
        from wd_spectra import multilevel_nlte as hydrogen
        from wd_spectra.atmosphere import gray_helium_atmosphere, gray_hydrogen_helium_atmosphere
        from wd_spectra.hot_nlte import HotNLTEModel
        from wd_spectra.models.common import ModelData
        from wd_spectra.models.hot import DOConfig, DAOConfig, _model_from_config
        if Path(hot.__file__).resolve() != source_root / "src/wd_spectra/_hot_structure.py":
            raise RuntimeError("Imported solver does not belong to this checkout")
        config = (DAOConfig if args.mixed else DOConfig)(effective_temperature=60000., logg=8., quality="quick",
            maximum_helium_ii_level=3, **({"maximum_hydrogen_level": 3} if args.mixed else {}))
        data = ModelData.default()
        model = _model_from_config(config, data)
        source_paths = sorted((source_root / "src/wd_spectra").rglob("*.py"))
        source_hashes = {str(p.relative_to(source_root)): file_digest(p) for p in source_paths}
        data_paths = dict(ccc_hydrogen_collisions=data.ccc_hydrogen_collisions, tlusty_source=data.tlusty_source,
            tlusty_helium_atom=data.tlusty_helium_atom,
            helium_i_profile=data.cache / "helium-stark" / config.helium_i_profile,
            helium_ii_profile=data.helium_ii_stark)
        data_hashes = {name: file_digest(path) for name, path in data_paths.items()}
        native_path = Path(opacity._rt.__file__).resolve() if opacity._rt is not None else None
        native_hash = file_digest(native_path) if native_path else None
        report.update(status="seed-construction-started", config=asdict(config), n_angle=model.n_angle,
            source_hashes=source_hashes, data_hashes=data_hashes, data_paths=data_paths,
            native_extension_available=native_path is not None, native_extension_path=native_path,
            native_extension_sha256=native_hash,
            runtime=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                         numpy=np.__version__, scipy=scipy.__version__),
            seed_function=("wd_spectra.atmosphere.gray_hydrogen_helium_atmosphere" if args.mixed else
                           "wd_spectra.atmosphere.gray_helium_atmosphere"),
            seed_parameters=dict(effective_temperature=60000., logg=8., n_depth=args.depths,
                tau_min=1e-8, tau_max=1000., rosseland_opacity=.1, hopf_constant=2/3, depth_concentration=0.,
                **({"log_hydrogen_to_helium": 2.} if args.mixed else {})),
            input_wave_function="numpy.geomspace(25,100000,80)")
        persist()
        seed = (gray_hydrogen_helium_atmosphere(60000., 8., 2., n_depth=args.depths) if args.mixed else
                gray_helium_atmosphere(60000., 8., n_depth=args.depths))
        input_wave = np.geomspace(25., 100000., 80)
        seed_arrays = {name: np.asarray(getattr(seed, name)) for name in SEED_FIELDS}
        seed_arrays["structure_input_wavelength"] = input_wave
        atomic_npz(seed_path, seed_arrays)
        report.update(status="seed-persisted", seed_metadata=seed.metadata,
            seed_array_sha256=array_digest(seed_arrays), input_wavelength_count=len(input_wave),
            input_wavelength_sha256=array_digest({"wavelength": input_wave}))
        try:
            from threadpoolctl import threadpool_info
            report["runtime"]["numerical_libraries"] = threadpool_info()
        except ImportError:
            pass
        report["setup_seconds_before_warmup"] = time.perf_counter()-started
        persist()
        options = dict(reuse_accepted_state=args.variant == "candidate")
        warm = hot.HotEquations(seed, model, input_wave, **options)
        warm_state = warm.initial_state()
        arrays["initial_state"] = warm_state.copy()
        arrays["structure_wavelength"] = warm.wave.copy()
        report.update(status="warmup-started", unknowns=len(warm_state), structure_wavelength_count=len(warm.wave),
            initial_state_sha256=array_digest({"state": warm_state}),
            structure_wavelength_sha256=array_digest({"wavelength": warm.wave}))
        persist(numerical=True)
        began = time.perf_counter()
        warmed = warm.evaluate(warm_state, True)
        report["untimed_warmup_evaluate_seconds"] = time.perf_counter()-began
        report["warmup_output_sha256"] = array_digest(
            {"state": warm_state, "residual": warmed.residual, "jacobian": warmed.jacobian})
        del warm, warmed, warm_state
        report["status"] = "measured-equations-initializing"
        persist()
        began = time.perf_counter()
        equations = hot.HotEquations(seed, model, input_wave, **options)
        state = equations.initial_state()
        report["measured_equations_setup_seconds"] = time.perf_counter()-began
        if equations.jacobian_key is not None or not np.array_equal(state, arrays["initial_state"]):
            raise RuntimeError("Measured equations must be fresh at the identical deterministic initial state")
        if not np.array_equal(equations.wave, arrays["structure_wavelength"]):
            raise RuntimeError("Fresh measured wavelength grid differs from warmup")
        report.update(fresh_measured_equations=True, measured_jacobian_cache_initially_empty=True,
            call_counts_scope="Measured evaluate request only; excludes both initial_state setups and warmup")
        counts = {name: 0 for name in ("base_transfer", "transfer_coefficients", "hydrogen_state", "hydrogen_rates")}
        report["call_counts"] = counts
        original_transfer = hot.transfer_field
        original_coefficients = HotNLTEModel.transfer_coefficients
        original_hydrogen = hydrogen.solve_multilevel_hydrogen_statistical_equilibrium

        def transfer(*positional, **keywords):
            counts["base_transfer"] += 1
            return original_transfer(*positional, **keywords)

        def coefficients(*positional, **keywords):
            counts["transfer_coefficients"] += 1
            return original_coefficients(*positional, **keywords)

        def hydrogen_solve(*positional, **keywords):
            counts["hydrogen_rates" if keywords.get("_return_rate_matrix") else "hydrogen_state"] += 1
            return original_hydrogen(*positional, **keywords)

        report["status"] = "measured-jacobian-started"
        persist()
        hot.transfer_field = transfer
        HotNLTEModel.transfer_coefficients = coefficients
        hydrogen.solve_multilevel_hydrogen_statistical_equilibrium = hydrogen_solve
        before = equations.evaluations
        began = time.perf_counter()
        try:
            result = equations.evaluate(state, True)
        finally:
            report["measured_evaluate_seconds"] = time.perf_counter()-began
            hot.transfer_field = original_transfer
            HotNLTEModel.transfer_coefficients = original_coefficients
            hydrogen.solve_multilevel_hydrogen_statistical_equilibrium = original_hydrogen
            report["instrumentation_restored"] = True
            report["measured_physical_residual_builds"] = equations.evaluations-before
        arrays.update(state=state.copy(), residual=result.residual.copy(), jacobian=result.jacobian.copy())
        output_digest = array_digest({name: arrays[name] for name in ("state", "residual", "jacobian")})
        report.update(status="complete", output_array_sha256=output_digest,
            measured_output_bitwise_matches_warmup_digest=output_digest == report["warmup_output_sha256"],
            physical_diagnostics=json_value(result.payload[2]),
            retained_canonical_physics_bundle=equations.last_physics is not None,
            array_summaries={name: dict(shape=list(value.shape), dtype=str(value.dtype),
                sum=float(value.sum()), maximum_absolute=float(np.max(abs(value))))
                for name, value in arrays.items()},
            identity_unchanged_after_measurement=dict(
                source_hashes=source_hashes == {str(p.relative_to(source_root)): file_digest(p)
                    for p in sorted((source_root / "src/wd_spectra").rglob("*.py"))},
                data_hashes=data_hashes == {name: file_digest(path) for name, path in data_paths.items()},
                native_extension_sha256=native_hash == (file_digest(native_path) if native_path else None),
                benchmark_sha256=report["benchmark_sha256"] == file_digest(__file__),
                thread_environment=report["thread_environment"] == {name: os.environ.get(name) for name in THREAD_FIELDS}))
        report["identity_unchanged"] = all(report["identity_unchanged_after_measurement"].values())
        persist(numerical=True)
    except BaseException as exc:
        report.update(status="failed", exception_type=type(exc).__name__, exception=str(exc))
        persist(numerical=bool(arrays))
        raise
    print(json.dumps(json_value(report), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
