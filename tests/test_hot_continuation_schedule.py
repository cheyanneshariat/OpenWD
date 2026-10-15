"""Hot NLTE continuation schedule; the stage solver is stubbed."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import wd_spectra._hot_structure as hot_structure
from wd_spectra.hot_nlte import solve_hot_nlte_atmosphere


@dataclass
class _Atmosphere:
    metadata: dict = field(default_factory=dict)


@dataclass
class _Result:
    atmosphere: _Atmosphere
    population_state: object
    nonlinear_result: object


def _fractions(monkeypatch, log_hydrogen_to_helium, maximum_iterations=120):
    seen = []

    def stage(seed, model, wave, *, nlte_fraction, **kwargs):
        seen.append(nlte_fraction)
        return _Result(_Atmosphere(), None, SimpleNamespace(converged=True, iterations=1))

    monkeypatch.setattr(hot_structure, "solve", stage)
    model = SimpleNamespace(
        log_hydrogen_to_helium=log_hydrogen_to_helium,
        population_maximum_iterations=120, population_tolerance=1e-4,
    )
    solve_hot_nlte_atmosphere(_Atmosphere(), model, np.geomspace(100, 1e5, 10),
                              maximum_iterations=maximum_iterations)
    return seen


def test_do_approaches_full_nlte_through_0p9_and_0p95(monkeypatch):
    assert _fractions(monkeypatch, None) == pytest.approx(
        [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])


def test_dao_adds_0p95_to_its_tenths(monkeypatch):
    fractions = _fractions(monkeypatch, 2.0)
    assert fractions == pytest.approx(list(np.linspace(0, 1, 11)[:-1]) + [0.95, 1.0])
    assert np.all(np.diff(fractions) > 0)


def test_two_step_smoke_budget_still_goes_straight_to_full_nlte(monkeypatch):
    assert _fractions(monkeypatch, None, maximum_iterations=2) == [1.0]
