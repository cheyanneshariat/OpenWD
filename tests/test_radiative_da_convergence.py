"""Cool, nonmagnetic DA with convection explicitly disabled."""
import json
import os
from pathlib import Path
import time
import warnings
from dataclasses import replace

import numpy as np
import pytest

from wd_spectra import DAConfig, compute_da
from wd_spectra.models import AtmosphereConvergenceWarning, save_model_result
from wd_spectra.models.common import _jsonable


@pytest.mark.parametrize("coarse_ok,fine_ok", [(True, True), (False, True), (False, False)])
def test_radiative_refinement_keeps_physics_and_requires_a_fresh_certificate(
    monkeypatch, coarse_ok, fine_ok
):
    """Synthetic certificates test wiring; the canary below tests actual physics."""
    from wd_spectra.atmosphere import gray_hydrogen_atmosphere
    from wd_spectra._convergence import equilibrium_certificate
    from wd_spectra.models import stellar
    from wd_spectra.spectrum import Spectrum

    calls, states, callbacks = [], [], []

    def structure(teff, logg, **options):
        calls.append(options)
        assert (teff, logg) == (6680.0, 7.96)
        assert options["mixing_length_alpha"] is None
        assert options["include_molecules"] and options["include_negative_hydrogen"]
        assert options["n_continuum_wavelength"] == 600
        if len(calls) == 2:
            np.testing.assert_array_equal(options["initial_temperature"], states[0].temperature)
            np.testing.assert_array_equal(options["initial_column_mass"], states[0].column_mass)
        passed = coarse_ok if len(calls) == 1 else fine_ok
        metadata = dict(
            radiative_equilibrium_solver_converged=passed,
            radiative_equilibrium_converged=passed,
            radiative_equilibrium_iterations=5,
            maximum_all_depth_total_flux_residual=1e-5 if passed else .2,
            maximum_relative_cell_energy_balance_residual=1e-5,
            maximum_unrestricted_log_temperature_correction=1e-5,
            temperature_correction_measured=True,
            electron_scattering_source_final_maximum_relative_residual=1e-12,
            lower_boundary_absorption_escape_bound=1e-5,
        )
        metadata["equilibrium_certificate"] = equilibrium_certificate(metadata)
        a = replace(gray_hydrogen_atmosphere(teff, logg, n_depth=options["n_depth"]), metadata=metadata)
        states.append(a)
        options["iteration_callback"](1, a, {})
        return a

    monkeypatch.setattr(stellar, "radiative_equilibrium_hydrogen_atmosphere", structure)
    monkeypatch.setattr(stellar, "synthesize_hydrogen_spectrum", lambda a, w, **k:
                        Spectrum(np.asarray(w), np.ones(len(w)), {}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AtmosphereConvergenceWarning)
        result = compute_da(
            DAConfig(effective_temperature=6680., logg=7.96, quality="production",
                     mixing_length_alpha=None, include_molecules=True),
            np.array([4000., 5000.]),
            iteration_callback=lambda i, a, s: callbacks.append((i, a.n_depth)),
        )
    assert [c["n_depth"] for c in calls] == ([100] if coarse_ok else [100, 200])
    assert callbacks == ([(1, 100)] if coarse_ok else [(1, 100), (6, 200)])
    expected = coarse_ok or fine_ok
    assert (result.metadata["atmosphere_convergence_status"] == "converged") == expected
    assert bool([w for w in caught if issubclass(w.category, AtmosphereConvergenceWarning)]) == (not expected)


def test_radiative_refined_restart_is_not_compressed(monkeypatch):
    from wd_spectra.atmosphere import gray_hydrogen_atmosphere
    from wd_spectra.models import stellar

    class Captured(Exception):
        pass

    def structure(*args, **options):
        assert options["n_depth"] == 200
        raise Captured

    monkeypatch.setattr(stellar, "radiative_equilibrium_hydrogen_atmosphere", structure)
    initial = gray_hydrogen_atmosphere(6680., 7.96, n_depth=200)
    with pytest.raises(Captured):
        compute_da(DAConfig(effective_temperature=6680., logg=7.96,
                           mixing_length_alpha=None, quality="production"),
                   np.array([4000., 5000.]), initial_atmosphere=initial)


@pytest.mark.canary
def test_g76_48_radiative_da_cold_start():
    """The fixed G 76-48 parameters must work before any magnetic synthesis."""
    output = os.environ.get("OPENWD_TEST_ARTIFACTS")
    directory = Path(output) / "g76-48" if output else None
    started = time.monotonic()
    records = []

    def progress(iteration, atmosphere, status):
        record = dict(iteration=iteration, seconds=time.monotonic() - started,
                      **_jsonable(status))
        records.append(record)
        print(json.dumps(record), flush=True)
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "iterations.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        assert len(records) <= 200, "radiative DA solve failed its work budget"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AtmosphereConvergenceWarning)
        result = compute_da(
            DAConfig(effective_temperature=6680.0, logg=7.96,
                     quality="production", mixing_length_alpha=None,
                     include_molecules=True),
            np.array([4000.0, 5000.0]),
            iteration_callback=progress,
        )
    if directory:
        save_model_result(result, directory / "nonmagnetic-structure")
    m = result.atmosphere.metadata
    assert result.atmosphere.n_depth == 200
    assert result.atmosphere.effective_temperature == 6680.0
    assert result.atmosphere.logg == 7.96
    assert m["maximum_convective_flux_fraction"] == 0.0
    assert m["convective_preconditioner_iterations"] == 0
    assert m["radiative_depth_refinement"]["initial_depth_points"] == 100
    assert m["radiative_depth_refinement"]["stellar_parameters_changed"] is False
    assert m["equilibrium_certificate"]["verified"], m["equilibrium_certificate"]
    assert m["maximum_all_depth_total_flux_residual"] < 2e-3
    assert m["maximum_relative_cell_energy_balance_residual"] < 2e-3
    assert m["temperature_correction_measured"]
    assert m["maximum_unrestricted_log_temperature_correction"] < 2e-4
    assert result.metadata["atmosphere_initialization"] == "gray"
    assert result.metadata["atmosphere_convergence_status"] == "converged"
    assert np.all(np.isfinite(result.spectrum.surface_flux_lambda))
    assert np.all(result.spectrum.surface_flux_lambda > 0)
    assert not any(issubclass(w.category, AtmosphereConvergenceWarning) for w in caught)
