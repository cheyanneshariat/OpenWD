"""Compare one fixed-state LTE Jacobian using public model material callbacks.

Run from the checkout with numerical threads pinned, for example:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
PYTHONPATH=src python benchmarks/benchmark_lte_probe_reuse.py --case daz

This captures the real public preset's materials on a contained 8-depth,
80-continuum-point, 16-metal-line structure fixture. It never runs Newton
iterations, qualifies an atmosphere, or synthesizes a spectrum. Compilation
and fixture construction are excluded. Constitutive data are read through
ModelData; missing data raise normally and are never downloaded.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
import platform
import sys
import time
from unittest.mock import patch

import numpy as np
import scipy

from wd_spectra import adaptive_structure as adaptive
from wd_spectra import d6 as d6_physics, helium, metals, opacity, radiative_transfer
from wd_spectra.models import (
    DAConfig, DBConfig, DZConfig, DAZConfig, D6Config,
    compute_da, compute_db, compute_dz, compute_daz, compute_d6,
)


class FixtureCaptured(Exception):
    pass


def physical_fixture(case, depths=8, continuum=80, metal_lines=16):
    """Capture exactly the callbacks passed by a public ordinary LTE adapter."""
    presets = {
        "da": (compute_da, DAConfig(quality="quick", lyman_profile_source="stark"),
               "radiative_equilibrium_hydrogen_atmosphere"),
        "db": (compute_db, DBConfig(quality="quick"),
               "radiative_equilibrium_helium_atmosphere"),
        "daz": (compute_daz, DAZConfig(quality="quick", lyman_profile_source="stark",
                    structure_maximum_metal_lines=metal_lines),
                "radiative_equilibrium_hydrogen_atmosphere"),
        "dz": (compute_dz, DZConfig(quality="quick",
                    structure_maximum_metal_lines=metal_lines),
               "radiative_equilibrium_helium_atmosphere"),
        "d6": (compute_d6, D6Config(quality="quick",
                    structure_maximum_metal_lines=metal_lines),
               "radiative_equilibrium_d6_atmosphere"),
    }
    compute, config, name = presets[case]
    ordinary_adapter = compute.__globals__[name]
    captured = {}

    def compact_adapter(*args, **options):
        options.update(n_depth=depths, n_continuum_wavelength=continuum,
                       max_iterations=1)
        if case == "d6":
            # Supply a declared-physics grey state to bound fixture creation,
            # rather than iterating the continuum initializer.
            from wd_spectra.d6 import gray_d6_atmosphere
            seed = gray_d6_atmosphere(
                args[0], args[1], args[2], args[4], n_depth=depths,
                reference_element=options.get("reference_element", "C"),
            )
            options.update(initial_temperature=seed.temperature,
                           initial_column_mass=seed.column_mass,
                           initial_rosseland_optical_depth=seed.rosseland_optical_depth)
        return ordinary_adapter(*args, **options)

    def capture(seed, wave, **options):
        captured.update(seed=seed, wavelength=wave, options=options)
        raise FixtureCaptured

    with patch.dict(compute.__globals__, {name: compact_adapter}), \
            patch.object(adaptive, "solve_adaptive_lte_structure", capture):
        try:
            compute(config, np.array([4000., 6000.]))
        except FixtureCaptured:
            pass
    if not captured:
        raise RuntimeError("Public preset did not enter the shared LTE adapter")
    return captured


def benchmark_fixture(fixture, variants=(False, True), samples=5):
    original = adaptive.solve_adaptive_lte_structure
    counts = {}
    prepared = {}
    rows = {reuse: {"variant": "candidate" if reuse else "baseline",
                   "jacobian_seconds": [], "material_calls": []}
            for reuse in variants}
    evaluations = {}
    sample_order = []

    def counted(name, function):
        def call(*args, **kwargs):
            counts[name] = counts.get(name, 0)+1
            return function(*args, **kwargs)
        return call

    def capture_evaluator(initial, evaluate, **options):
        prepared[reuse] = (initial.copy(), evaluate)
        raise FixtureCaptured

    options = dict(fixture["options"],
        use_convective_gradient_preconditioner=False,
        project_initial_convective_gradient=False,
        use_initial_bolometric_rescaling=False, enforce_local_energy_balance=True,
        max_iterations=1)
    with ExitStack() as stack:
        stack.enter_context(patch.object(adaptive, "solve_trust_region_newton", capture_evaluator))
        for module, attribute, name in (
            (metals, "metal_line_mass_absorption_coefficient", "metal_line_builds"),
            (d6_physics, "metal_line_mass_absorption_coefficient", "metal_line_builds"),
            (opacity, "hydrogen_continuum_mass_absorption_coefficient", "hydrogen_continuum_calls"),
            (helium, "helium_continuum_mass_absorption_coefficient", "helium_continuum_calls"),
        ):
            if hasattr(module, attribute):
                stack.enter_context(patch.object(module, attribute,
                    counted(name, getattr(module, attribute))))
        for reuse in variants:
            try:
                original(fixture["seed"], fixture["wavelength"], **dict(
                    options, reuse_material_probe_rosseland=reuse))
            except FixtureCaptured:
                pass
        # Warm every retained evaluator before taking any timings. Each has
        # its own base transfer cache; both use the same material callbacks.
        for initial, evaluate in prepared.values():
            evaluate(initial, False)
            evaluate(initial, True)
        for sample in range(samples):
            # Reverse each paired sample to counter a consistent ordering bias.
            order = variants if sample % 2 == 0 else tuple(reversed(variants))
            for reuse in order:
                initial, evaluate = prepared[reuse]
                counts.clear()
                started = time.perf_counter()
                ev = evaluate(initial, True)
                rows[reuse]["jacobian_seconds"].append(time.perf_counter()-started)
                rows[reuse]["material_calls"].append(dict(counts))
                evaluations[reuse] = ev
                sample_order.append(dict(sample=sample+1, variant=rows[reuse]["variant"]))
    for row in rows.values():
        row["median_jacobian_seconds"] = float(np.median(row["jacobian_seconds"]))
        row["samples"] = samples
        keys = set().union(*(calls.keys() for calls in row["material_calls"]))
        row["material_calls_per_jacobian"] = {
            key: float(np.mean([calls.get(key, 0) for calls in row["material_calls"]]))
            for key in keys}
    return list(rows.values()), evaluations, sample_order


def runtime_metadata():
    try:
        from threadpoolctl import threadpool_info
        threadpools = threadpool_info()
    except ImportError:
        threadpools = None
    native = radiative_transfer._rt
    return dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        transfer_backend="native available" if native is not None else "python",
        native_extension=None if native is None else native.__file__,
        thread_environment={name: os.environ.get(name) for name in (
            "OPENWD_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        numerical_threadpools=threadpools)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("da", "db", "daz", "dz", "d6"), default="daz")
    parser.add_argument("--variant", choices=("baseline", "candidate", "both"), default="both")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--depths", type=int, default=8)
    parser.add_argument("--continuum", type=int, default=80)
    parser.add_argument("--metal-lines", type=int, default=16)
    parser.add_argument("--fixture-only", action="store_true",
                        help="Validate physical fixture construction without timing")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.depths < 3 or args.continuum < 80 or args.metal_lines < 0:
        parser.error("Require depths >= 3, continuum >= 80 and metal-lines >= 0")
    try:
        fixture = physical_fixture(args.case, args.depths, args.continuum, args.metal_lines)
    except FileNotFoundError as error:
        parser.exit(2, f"Missing constitutive data for {args.case}: {error}\n")
    if args.fixture_only:
        print(json.dumps({"case": args.case, "depths": fixture["seed"].n_depth,
            "wavelengths": len(fixture["wavelength"]), "fixture_constructed": True,
            "jacobian_or_cold_solve_measured": False}, indent=2))
        return
    variants = (False, True) if args.variant == "both" else (args.variant == "candidate",)
    rows, evaluations, sample_order = benchmark_fixture(fixture, variants, args.samples)
    equal = None
    if len(evaluations) == 2:
        np.testing.assert_array_equal(evaluations[False].residual, evaluations[True].residual)
        np.testing.assert_array_equal(evaluations[False].jacobian, evaluations[True].jacobian)
        equal = True
    print(json.dumps({"case": args.case, "depths": fixture["seed"].n_depth,
        "wavelengths": len(fixture["wavelength"]), "results": rows,
        "sample_order": sample_order, "runtime": runtime_metadata(),
        "residual_and_jacobian_bitwise_equal": equal, "compilation_included": False,
        "cold_solve_or_spectrum_measured": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
