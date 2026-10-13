"""Full-resolution D6 operations shared by frozen tests and their offline builder."""

from contextlib import contextmanager, nullcontext
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import numpy as np

from wd_spectra import D6Config, compute_d6
from wd_spectra.d6 import _bulk_atmosphere, radiative_equilibrium_d6_atmosphere
from wd_spectra.models.common import ModelData, numerical_resolution
from wd_spectra.models.d6 import _atomic_inputs, _N_DEPTH, _STRUCTURE_LINES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from uv_atomic_control_legacy import legacy_uv_atomic_controls

STEPS = 2
ARRAYS = ("temperature", "gas_pressure", "mass_density", "electron_density")
METRICS = (
    "maximum_all_depth_total_flux_residual",
    "maximum_relative_cell_energy_balance_residual",
    "maximum_log_temperature_correction",
    "scattering_source_maximum_relative_residual",
)
TRAJECTORY_RTOL = dict(temperature=2e-5, gas_pressure=1e-12,
                       mass_density=1e-4, electron_density=1e-4)
METRIC_RTOL = 5e-3
METRIC_ATOL = (1e-7, 1e-6, 1e-6, 1e-10)


@lru_cache(maxsize=1)
def atomic_inputs():
    return _atomic_inputs(ModelData.default(), tuple(D6Config().abundances))


@contextmanager
def atomic_input_scope(*, legacy=False):
    """Keep cached atoms local to a current or historical regression scope.

    The immutable pre-Fe controls need their original Fe II atom. A cache
    populated before the patch, or retained after it, would silently mix
    physics identities. Clear it on both boundaries, including exceptions.
    """
    atomic_inputs.cache_clear()
    try:
        with legacy_uv_atomic_controls() if legacy else nullcontext():
            yield
    finally:
        atomic_inputs.cache_clear()


def assert_trajectory_matches(records, saved):
    """Apply the unchanged warm-v1 guards and return measured differences.

    The explicit offline audit and the regression test use the same bounds;
    this helper neither chooses a new reference nor claims convergence.
    """
    assert len(records) == STEPS
    np.testing.assert_array_equal([r["iteration"] for r in records], saved["iterations"])
    np.testing.assert_array_equal([r["phase"] for r in records], saved["phases"])
    differences = {}

    def measured_difference(actual, expected):
        return {
            "maximum_absolute_difference": float(np.max(np.abs(actual - expected))),
            "maximum_relative_difference": float(np.max(
                np.abs(actual - expected) / np.maximum(
                    np.abs(expected), np.finfo(np.float64).tiny,
                )
            )),
        }

    # Complete profiles protect more than scalar maxima. These are the
    # original cross-platform floating-point/linear-algebra tolerances.
    for key in ARRAYS:
        actual = np.stack([r[key] for r in records])
        assert np.all(np.isfinite(actual)) and np.all(actual > 0)
        np.testing.assert_allclose(actual, saved[key], rtol=TRAJECTORY_RTOL[key], atol=0)
        differences[key] = measured_difference(actual, saved[key])
    metrics = np.stack([r["metrics"] for r in records])
    assert np.all(np.isfinite(metrics)) and np.all(metrics >= 0)
    differences["metrics"] = {}
    for index, absolute in enumerate(METRIC_ATOL):
        actual, expected = metrics[:, index], saved["metrics"][:, index]
        np.testing.assert_allclose(actual, expected, rtol=METRIC_RTOL, atol=absolute)
        differences["metrics"][METRICS[index]] = measured_difference(actual, expected)
    return differences


def solve_structure(column_mass, optical_depth, temperature, callback=None):
    """Recompute the public standard material closures, with an explicit warm seed.

    The low-level API explicitly selects the conservative warm-start phase.
    Frozen test inputs do not claim a matching current-code checkpoint hash.
    """
    config = D6Config()
    database, photoionization, topbase = atomic_inputs()
    resolution = numerical_resolution(config.quality)
    return radiative_equilibrium_d6_atmosphere(
        config.effective_temperature, config.logg, database, photoionization,
        config.abundances, reference_element=config.reference_element,
        n_depth=_N_DEPTH[config.quality], max_iterations=resolution.maximum_iterations,
        maximum_metal_lines=_STRUCTURE_LINES[config.quality],
        minimum_metal_oscillator_strength=1.0e-4,
        include_rydberg_dissolution=config.include_rydberg_dissolution,
        include_metal_series_pseudocontinuum=config.include_metal_series_pseudocontinuum,
        metal_series_pseudocontinuum_elements=config.metal_series_pseudocontinuum_elements,
        topbase_photoionization_database=topbase,
        include_linear_stark_quasistatic=config.include_linear_stark_quasistatic,
        linear_stark_profile=config.linear_stark_profile,
        microturbulent_velocity_kms=config.microturbulent_velocity_kms,
        include_oxygen_i_series_stark=config.include_oxygen_i_series_stark,
        profile_edge_optical_depth=config.line_profile_edge_optical_depth,
        n_angle=min(resolution.n_angle, 3), mixing_length_alpha=config.mixing_length_alpha,
        initial_temperature=temperature, initial_column_mass=column_mass,
        initial_rosseland_optical_depth=optical_depth,
        resume_supplied_structure_in_formal_flux_phase=True,
        iteration_callback=callback,
    )


class TrajectoryComplete(Exception):
    """Stop after the requested measured updates, without claiming convergence."""


def trajectory(column_mass, optical_depth, temperature):
    records = []
    started = time.monotonic()

    def record(iteration, atmosphere, diagnostic):
        records.append(dict(
            iteration=iteration, phase=diagnostic["solver_phase"],
            metrics=np.array([diagnostic[key] for key in METRICS]),
            **{key: getattr(atmosphere, key).copy() for key in ARRAYS},
        ))
        print(json.dumps(dict(iteration=iteration, phase=diagnostic["solver_phase"],
                              elapsed_seconds=time.monotonic() - started,
                              **{key: diagnostic[key] for key in METRICS})), flush=True)
        if len(records) == STEPS:
            raise TrajectoryComplete

    try:
        solve_structure(column_mass, optical_depth, temperature, record)
    except TrajectoryComplete:
        return records
    raise AssertionError(f"D6 returned before {STEPS} measured trajectory steps")


def fixed_spectrum(column_mass, optical_depth, temperature, wavelength):
    config = D6Config()
    database, _, _ = atomic_inputs()
    atmosphere, _ = _bulk_atmosphere(
        config.effective_temperature, config.logg, optical_depth, column_mass,
        temperature, database, config.abundances,
        reference_element=config.reference_element,
        metadata={"model": "frozen-d6-regression-input",
                  "radiative_equilibrium_converged": False},
    )
    return compute_d6(config, wavelength, initial_atmosphere=atmosphere,
                      relax_atmosphere=False).spectrum
