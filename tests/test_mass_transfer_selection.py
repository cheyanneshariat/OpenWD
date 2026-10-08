"""One solve owns its discretization: no shared last-opacity state."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from wd_spectra import adaptive_structure as adaptive
from wd_spectra.atmosphere import Atmosphere


def solve(mode, *, scattering=0.9, local_energy=True, backend=None):
    n = 12
    mass = np.geomspace(1e-8, 100., n)
    seed = Atmosphere(8000., 8., mass, mass, np.full(n, 7000.), 1e8*mass,
                      np.ones(n), np.zeros(n), np.zeros(n), np.ones(n), {})
    extinction = np.broadcast_to(1+np.exp(np.sin(np.arange(n))), (5, n))
    return adaptive.solve_adaptive_lte_structure(
        seed, np.array([1000., 2000., 4000., 8000., 20000.]),
        with_temperature=lambda t: replace(seed, temperature=t),
        true_absorption=lambda a: (1-scattering)*extinction,
        scattering_opacity=lambda a: scattering*extinction,
        rosseland_opacity=lambda a: np.ones(n), thermodynamics=lambda a: None,
        mixing_length_alpha=None, max_iterations=1, temperature_tolerance=3e-4,
        flux_tolerance=3e-3, n_angle=2, initial_temperature_was_supplied=False,
        use_initial_bolometric_rescaling=False, maximum_formal_flux_continuations=0,
        compute_local_energy_response=local_energy,
        **({} if mode is None else dict(transfer_discretization=mode)),
        **({} if backend is None else dict(column_mass_radiation_backend=backend)))


@pytest.mark.parametrize("scattering", [0., 0.9])
@pytest.mark.parametrize("local_energy", [False, True])
def test_explicit_mass_operator_is_used_and_default_survives(monkeypatch, scattering, local_energy):
    from wd_spectra import _mass_feautrier as mass
    calls = []
    original = mass.mass_field
    def field(*args, **kwargs):
        calls.append(kwargs["column_mass"].copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(mass, "mass_field", field)
    before = solve(None, scattering=scattering, local_energy=local_energy)
    assert not calls
    changed = solve("column-mass", scattering=scattering, local_energy=local_energy)
    assert calls
    assert changed.metadata["transfer_discretization"] == "column-mass"
    assert not np.array_equal(before.temperature, changed.temperature)
    count = len(calls)
    after = solve("optical-depth", scattering=scattering, local_energy=local_energy)
    assert len(calls) == count
    np.testing.assert_array_equal(before.temperature, after.temperature)
    from wd_spectra.models.common import _jsonable
    assert _jsonable(before.metadata) == _jsonable(after.metadata)


def test_concurrent_calls_do_not_share_transfer_state():
    expected = {mode: solve(mode) for mode in ("column-mass", "optical-depth")}
    modes = ["column-mass", "optical-depth"]*2
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(solve, modes))
    for mode, result in zip(modes, actual):
        np.testing.assert_array_equal(result.temperature, expected[mode].temperature)


def test_invalid_transfer_mode_fails_before_material_evaluation():
    with pytest.raises(ValueError, match="transfer discretization"):
        solve("automatic-temperature-switch")


def test_explicit_column_mass_backend_owns_every_transfer_operation():
    from wd_spectra import _mass_feautrier as mass

    class Backend:
        def __init__(self):
            self.calls = []

        def field(self, *args, **kwargs):
            self.calls.append("field")
            return mass.mass_field(*args, **kwargs)

        def response(self, *args, **kwargs):
            self.calls.append("response")
            return mass.mass_response(*args, **kwargs)

        def energy(self, *args, **kwargs):
            self.calls.append("energy")
            return mass.mass_energy(*args, **kwargs)

        def energy_response(self, *args, **kwargs):
            self.calls.append("energy_response")
            return mass.mass_energy_response(*args, **kwargs)

        def boundary(self, *args, **kwargs):
            self.calls.append("boundary")
            return adaptive._thermal_boundary_absorption_escape_bound(
                *args, **kwargs
            )

        def record_temperature_response_probes(self, probes):
            self.calls.append("temperature_response_probes")
            assert len(probes) == 3

    expected = solve("column-mass")
    backend = Backend()
    actual = solve("column-mass", backend=backend)
    np.testing.assert_array_equal(actual.temperature, expected.temperature)
    assert {"field", "energy", "energy_response", "boundary",
            "temperature_response_probes"}.issubset(backend.calls)
    with pytest.raises(ValueError, match="requires column-mass"):
        solve("optical-depth", backend=Backend())
