"""Cold-start equilibrium AND spectral controls, without external atmospheres.

Reviewed corrected outputs are separate from the untouched historical controls.
No reference is supplied to an atmosphere solver. Fixed-state synthesis has
separate, tighter checks in test_spectral_regressions.py.
"""

import logging
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from wd_spectra._compat import trapezoid
from wd_spectra.constants import STEFAN_BOLTZMANN

from wd_spectra.models import (
    AtmosphereConvergenceWarning,
    DAConfig,
    DBConfig,
    compute_da,
    compute_db,
    save_model_result,
)
from wd_spectra.models.common import _jsonable


pytestmark = pytest.mark.canary

_FORMAL_WAVELENGTH = np.unique(
    np.r_[np.geomspace(900.0, 300000.0, 1100), np.arange(3700.0, 7000.0, 2.0)]
)


def _assert_solver_control(result, case, step_tolerance, caught):
    if os.environ.get("OPENWD_TEST_ARTIFACTS"):
        save_model_result(result, Path(os.environ["OPENWD_TEST_ARTIFACTS"]) / case)
    m = result.atmosphere.metadata
    assert m["radiative_equilibrium_solver_converged"]
    history = [
        entry
        for segment in m["nonlinear_solver_segments"]
        for entry in segment["iteration_history"]
    ]
    assert history and history[-1]["maximum_step"] < step_tolerance
    with np.load(
        Path(__file__).parent / "data/approved_regressions/cold" / (case + ".npz")
    ) as old:
        for name in ("gas_pressure", "column_mass"):
            np.testing.assert_allclose(
                getattr(result.atmosphere, name)[: len(old[name])],
                old[name],
                rtol=2e-5,
                atol=0.0,
            )
        wave, i, j = np.intersect1d(
            result.spectrum.wavelength_angstrom,
            old["wavelength"],
            return_indices=True,
        )
        expected = old["surface_flux"][j]
        actual = result.spectrum.surface_flux_lambda[i]
        # Absolute spectral changes are measured on the stellar energy scale,
        # not relative to near-zero Wien-tail bins. The bound is the existing
        # 0.3% equilibrium flux accuracy, common to every model, not a fitted
        # per-star spectral tolerance. Fixed-atmosphere tests remain stricter.
        assert np.max(wave * abs(actual - expected)) / np.max(wave * expected) < 3e-3
        target = STEFAN_BOLTZMANN * result.atmosphere.effective_temperature**4
        assert trapezoid(abs(actual - expected), wave) / target < 3e-3
        for lo, hi in ((1150, 3000), (3500, 7000), (7000, 300000)):
            take = (wave >= lo) & (wave <= hi)
            if np.sum(take) < 2:
                continue
            old_band = trapezoid(expected[take], wave[take])
            if old_band / target >= 1e-3:
                change = trapezoid((actual - expected)[take], wave[take]) / old_band
                assert abs(change) < 3e-3
    certificate = m["equilibrium_certificate"]
    assert certificate["verified"], certificate["failures"]
    assert m["temperature_correction_measured"]
    assert m["radiative_equilibrium_converged"] == certificate["verified"]
    assert result.metadata["atmosphere_convergence_status"] == (
        "converged" if certificate["verified"] else "unconverged"
    )
    convergence_warnings = [
        w for w in caught if issubclass(w.category, AtmosphereConvergenceWarning)
    ]
    assert bool(convergence_warnings) == (not certificate["verified"])
    assert m["nonlinear_solver_terminal_reason"] != "initial-state-converged"
    assert result.spectrum.metadata["source_converged"]


@pytest.fixture(autouse=True)
def _live_nonlinear_diagnostics(capsys, request):
    """Also show Jacobian work and rejected directions between accepted steps."""

    class LiveHandler(logging.Handler):
        def emit(self, record):
            with capsys.disabled():
                print(f"[{request.node.name}] {record.getMessage()}", flush=True)

    logger = logging.getLogger("wd_spectra.nonlinear")
    handler = LiveHandler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _progress_callback(label, capsys):
    """Emit live convergence telemetry even while pytest capture is active."""

    start = time.monotonic()
    with capsys.disabled():
        print(f"[{label}] starting cold atmosphere (no checkpoint)", flush=True)

    def report(iteration, _atmosphere, diagnostics):
        artifact_root = os.environ.get("OPENWD_TEST_ARTIFACTS")
        if artifact_root:
            directory = Path(artifact_root)
            directory.mkdir(parents=True, exist_ok=True)
            record = dict(
                iteration=iteration,
                elapsed_seconds=time.monotonic() - start,
                diagnostic_only=True,
                diagnostics=diagnostics,
            )
            with (directory / "iterations.jsonl").open("a") as stream:
                stream.write(json.dumps(_jsonable(record)) + "\n")
            # Last accepted state for diagnosis only; no test ever loads it.
            temporary = directory / "last-iteration.tmp.npz"
            np.savez_compressed(
                temporary,
                temperature=_atmosphere.temperature,
                column_mass=_atmosphere.column_mass,
                gas_pressure=_atmosphere.gas_pressure,
                diagnostic_only=True,
            )
            temporary.replace(directory / "last-iteration.npz")
        maximum_flux = diagnostics.get(
            "maximum_all_depth_total_flux_residual",
            diagnostics.get("maximum_total_flux_residual", np.nan),
        )
        message = (
            f"[{label}] elapsed={time.monotonic() - start:.1f}s "
            f"iteration={iteration} "
            f"phase={diagnostics.get('solver_phase', 'unknown')} "
            f"max_flux={float(maximum_flux):.6g} "
            "max_dlnT="
            f"{float(diagnostics.get('maximum_log_temperature_correction', np.nan)):.6g} "
            f"trust={float(diagnostics.get('trust_radius', np.nan)):.6g} "
            f"line_search={float(diagnostics.get('line_search_factor', np.nan)):.6g}"
        )
        # sys.__stdout__ still points at pytest's captured file descriptor.
        # Explicitly suspend capture so a stalled canary remains observable.
        with capsys.disabled():
            print(message, flush=True)

    return report


@pytest.mark.parametrize(
    "effective_temperature,maximum_expected_iterations",
    # Work guards, not physical tolerances. The flux-conserving
    # photosphere-concentrated depths (2026-10-01) took 82 and 55 iterations;
    # with uniform depths the 10 kK model needed 158 and did not certify.
    [(10_000.0, 90), (22_000.0, 90)],
)
def test_protected_db_cold_starts_converge_without_fallback(
    effective_temperature,
    maximum_expected_iterations,
    capsys,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AtmosphereConvergenceWarning)
        result = compute_db(
            DBConfig(
                effective_temperature=effective_temperature,
                quality="production",
            ),
            _FORMAL_WAVELENGTH,
            iteration_callback=_progress_callback(
                f"DB-{effective_temperature:g}K-production", capsys
            ),
        )

    metadata = result.atmosphere.metadata
    assert 80 <= result.atmosphere.n_depth <= 160
    _assert_solver_control(result, f"db-{effective_temperature:g}", 3e-4, caught)
    assert metadata["maximum_all_depth_total_flux_residual"] < 3.0e-3
    assert metadata["radiative_equilibrium_iterations_including_domain_adaptation"] <= (
        maximum_expected_iterations
    )
    assert metadata["cold_start"]
    assert not result.metadata["checkpoint_matches_model_request"]


def test_standard_db_22000_enters_exact_flux_verification(capsys):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AtmosphereConvergenceWarning)
        result = compute_db(
            DBConfig(effective_temperature=22_000.0, quality="standard"),
            _FORMAL_WAVELENGTH,
            iteration_callback=_progress_callback("DB-22000K-standard", capsys),
        )

    metadata = result.atmosphere.metadata
    assert result.atmosphere.n_depth == 40
    _assert_solver_control(result, "db-22000-standard", 3e-4, caught)
    assert metadata["maximum_all_depth_total_flux_residual"] < 3.0e-3
    assert (
        metadata["radiative_equilibrium_iterations_including_domain_adaptation"] <= 60
    )
    assert metadata["nonlinear_solver_segments"][-1]["phase"] in (
        "formal-radiative-flux-completion",
        "local-energy-completion",
    )
    assert metadata["nonlinear_solver_terminal_reason"] in (
        "initial-state-converged",
        "residual-and-step-converged",
        "stationary-residual-converged",
    )
    assert not metadata["initial_temperature_was_supplied"]


@pytest.mark.parametrize(
    "effective_temperature,maximum_expected_iterations",
    # Work guards, not physical tolerances. Photosphere-concentrated depths
    # (2026-10-01) took 64 and 50 iterations, against 45 and 33 uniform.
    [(5_000.0, 70), (20_000.0, 70)],
)
def test_protected_da_cold_starts_converge_without_fallback(
    effective_temperature,
    maximum_expected_iterations,
    capsys,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AtmosphereConvergenceWarning)
        result = compute_da(
            DAConfig(
                effective_temperature=effective_temperature,
                quality="production",
            ),
            _FORMAL_WAVELENGTH,
            iteration_callback=_progress_callback(
                f"DA-{effective_temperature:g}K-production", capsys
            ),
        )

    metadata = result.atmosphere.metadata
    assert result.atmosphere.n_depth == 100
    _assert_solver_control(result, f"da-{effective_temperature:g}", 3e-4, caught)
    assert metadata["maximum_total_flux_residual"] < 3.0e-3
    assert metadata["radiative_equilibrium_iterations"] <= (maximum_expected_iterations)
    assert result.metadata["atmosphere_initialization"] == "gray"


@pytest.mark.parametrize("effective_temperature", [3_000.0, 4_000.0])
def test_ultracool_da_cold_starts_converge_with_exact_flux_verification(
    effective_temperature,
    capsys,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AtmosphereConvergenceWarning)
        result = compute_da(
            DAConfig(
                effective_temperature=effective_temperature,
                quality="production",
            ),
            _FORMAL_WAVELENGTH,
            iteration_callback=_progress_callback(
                f"DA-{effective_temperature:g}K-production", capsys
            ),
        )

    metadata = result.atmosphere.metadata
    assert result.atmosphere.n_depth == 100
    _assert_solver_control(result, f"da-{effective_temperature:g}", 2e-4, caught)
    assert metadata["maximum_all_depth_total_flux_residual"] < 2.0e-3
    # A shared work guard for the two 100-layer ultracool controls, including
    # conditioning and the final stationarity probe. Exact iteration counts
    # vary with floating-point arithmetic; this is not a physical tolerance.
    assert metadata["radiative_equilibrium_iterations"] <= 120
    assert metadata["formal_flux_completion_used"]
    assert metadata["adiabatic_asymptotic_conditioning_enabled"]
    assert metadata["adiabatic_asymptotic_interfaces"] > 0
    assert metadata["initial_convective_gradient_projection_mode"] == (
        "interface-transport"
    )
    assert not metadata["initial_bolometric_rescaling_enabled"]
    assert not metadata["initial_temperature_was_supplied"]
    assert result.metadata["atmosphere_initialization"] == "gray"


def test_standard_dab_12000_completes_thin_layer_energy_balance(capsys):
    """A converged flux residual must not hide a non-stationary surface mode.

    In optically thin outer layers the total flux barely constrains T, so the
    formal-flux phase can accept a line-searched step at round-off residual
    while its unrestricted Newton proposal is a 0.6% surface sawtooth.  That
    state must be completed with local energy balance, not handed back to fail
    the stationarity certificate (2026-10-01).
    """
    from wd_spectra.models import DABConfig, compute_dab

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AtmosphereConvergenceWarning)
        result = compute_dab(
            DABConfig(
                effective_temperature=12_000.0,
                logg=8.0,
                log_hydrogen_to_helium=-2.0,
                quality="standard",
            ),
            _FORMAL_WAVELENGTH,
            iteration_callback=_progress_callback("DAB-12000K-standard", capsys),
        )
    metadata = result.atmosphere.metadata
    certificate = metadata["equilibrium_certificate"]
    assert certificate["verified"], certificate["failures"]
    assert metadata["local_energy_completion_used"]
    assert not [w for w in caught if issubclass(w.category, AtmosphereConvergenceWarning)]
