"""DA lower-boundary extension wiring. The structure solver is stubbed; this
checks when compute_da re-solves cold on a deeper seed, and with what."""

from dataclasses import replace

import pytest
from wd_spectra import gray_hydrogen_atmosphere
from wd_spectra.models import stellar
from wd_spectra.models.stellar import DAConfig, compute_da


class _Done(Exception):
    def __init__(self, atmosphere):
        self.atmosphere = atmosphere


def _run(monkeypatch, config, failures_by_call, initial_atmosphere=None):
    calls = []

    def solve(teff, logg, *, n_depth, tau_max, **kwargs):
        failures = failures_by_call[min(len(calls), len(failures_by_call) - 1)]
        calls.append(dict(kwargs, n_depth=n_depth, tau_max=tau_max))
        atmosphere = gray_hydrogen_atmosphere(teff, logg, n_depth=n_depth)
        return replace(atmosphere, metadata={
            "equilibrium_certificate": {"failures": list(failures)},
            "radiative_equilibrium_iterations": 10,
            "lower_boundary_absorption_escape_bound": 0.01 if failures else 1e-6,
        })

    def finish(atmosphere, kind):
        raise _Done(atmosphere)

    monkeypatch.setattr(stellar, "radiative_equilibrium_hydrogen_atmosphere", solve)
    monkeypatch.setattr(stellar, "atmosphere_with_model_request_fingerprint", lambda a, f: a)
    monkeypatch.setattr(stellar, "fixed_synthesis_atmosphere", lambda a, f: a)
    monkeypatch.setattr(stellar, "warn_if_atmosphere_not_converged", finish)
    with pytest.raises(_Done) as done:
        compute_da(config, initial_atmosphere=initial_atmosphere)
    return calls, done.value.atmosphere


_HOT = dict(
    effective_temperature=32_000.0, logg=6.0, quality="production",
    lyman_profile_source="stark", include_molecules=False,
)


def test_screen_failure_deepens_cold_with_proportional_layers(monkeypatch):
    calls, atmosphere = _run(
        monkeypatch, DAConfig(**_HOT),
        [["boundary_screening", "local_energy"], ["boundary_screening"], []],
    )
    assert [c["tau_max"] for c in calls] == pytest.approx([1e2, 4e2, 1.6e3])
    # Same log-tau spacing as the 100-layer, 1e-10..1e2 seed.
    assert [c["n_depth"] for c in calls] == [100, 106, 111]
    assert all(c["initial_temperature"] is None for c in calls)
    segments = atmosphere.metadata["adaptive_domain_segments"]
    assert [s["failures"] for s in segments][-1] == []
    assert atmosphere.metadata["adaptive_domain_expansions"] == 2
    assert atmosphere.metadata[
        "radiative_equilibrium_iterations_including_domain_adaptation"
    ] == 30


def test_other_failures_alone_do_not_deepen(monkeypatch):
    calls, atmosphere = _run(monkeypatch, DAConfig(**_HOT), [["local_energy"]])
    assert len(calls) == 1
    assert "adaptive_domain_segments" not in atmosphere.metadata


def test_extensions_are_bounded_and_can_be_disabled(monkeypatch):
    calls, _ = _run(
        monkeypatch, DAConfig(**_HOT, lower_boundary_extensions=2),
        [["boundary_screening"]],
    )
    assert len(calls) == 3
    calls, _ = _run(
        monkeypatch, DAConfig(**_HOT, lower_boundary_extensions=0),
        [["boundary_screening"]],
    )
    assert len(calls) == 1 and calls[0]["tau_max"] == pytest.approx(1e2)


def test_supplied_atmosphere_is_not_deepened(monkeypatch):
    seed = gray_hydrogen_atmosphere(32_000.0, 6.0, n_depth=100)
    calls, _ = _run(
        monkeypatch, DAConfig(**_HOT), [["boundary_screening"]],
        initial_atmosphere=seed,
    )
    assert len(calls) == 1


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_invalid_extension_count_is_rejected(bad):
    with pytest.raises(ValueError):
        compute_da(DAConfig(**_HOT, lower_boundary_extensions=bad))


def test_convective_seed_is_the_da_default_and_reaches_the_solver(monkeypatch):
    calls, _ = _run(monkeypatch, DAConfig(**_HOT), [[]])
    assert calls[0]["convective_seed"] is True
    calls, _ = _run(monkeypatch, DAConfig(**_HOT, convective_seed=False), [[]])
    assert calls[0]["convective_seed"] is False


def test_convective_seed_leaves_a_radiative_gray_seed_unchanged():
    import numpy as np
    from wd_spectra.atmosphere import hydrogen_continuum_atmosphere

    kw = dict(n_depth=30, tau_min=1e-8, tau_max=100.0)
    gray = hydrogen_continuum_atmosphere(20_000.0, 8.0, **kw)
    seed = hydrogen_continuum_atmosphere(20_000.0, 8.0, convective_seed=True, **kw)
    assert not seed.metadata["convective_seed_applied"]
    np.testing.assert_array_equal(seed.temperature, gray.temperature)
    np.testing.assert_array_equal(seed.column_mass, gray.column_mass)


def test_convective_seed_replaces_an_unstable_gray_interior():
    import numpy as np
    from wd_spectra.atmosphere import hydrogen_continuum_atmosphere

    kw = dict(n_depth=30, tau_min=1e-8, tau_max=100.0)
    gray = hydrogen_continuum_atmosphere(8_000.0, 8.0, **kw)
    seed = hydrogen_continuum_atmosphere(8_000.0, 8.0, convective_seed=True, **kw)
    assert seed.metadata["convective_seed_applied"]
    k = int(np.searchsorted(gray.rosseland_optical_depth, 1.0))
    # Unchanged above depth one; cooler and deeper in mass below it.
    np.testing.assert_array_equal(seed.temperature[: k + 1], gray.temperature[: k + 1])
    assert seed.temperature[-1] < gray.temperature[-1]
    assert seed.column_mass[-1] > gray.column_mass[-1]
    assert seed.rosseland_optical_depth[-1] == pytest.approx(100.0, rel=1e-6)
    for array in (seed.rosseland_optical_depth, seed.column_mass, seed.gas_pressure):
        assert np.all(np.diff(array) > 0)
