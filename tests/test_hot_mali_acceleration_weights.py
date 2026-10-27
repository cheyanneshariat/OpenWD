"""Hot-MALI fit weights follow the existing physical population floor."""
from dataclasses import replace

import numpy as np
import pytest

from wd_spectra import multilevel_nlte as hydrogen
from wd_spectra.hot_nlte import population_arrays
from test_hot_mali import sdb, atom  # noqa: F401 (fixtures)


@pytest.mark.parametrize("accelerated_lambda", [False, True])
def test_only_mali_weights_the_fit_by_the_existing_population_floor(
    sdb, monkeypatch, accelerated_lambda,
):
    model, atmosphere = sdb
    model = replace(model, population_maximum_iterations=2, population_tolerance=1e-14)
    _, reference = population_arrays(model._rate_state(atmosphere))
    original = hydrogen._anderson_log_population_update
    calls = []

    def checked_fit(log_old, log_new, history, **kwargs):
        weights = kwargs.get("residual_weights")
        assert kwargs["depth"] == 4
        assert kwargs["mixing"] == 0.65
        assert kwargs["maximum_step"] == 0.75
        if accelerated_lambda:
            # Independently reconstruct the physical populations from the
            # supplied log departures, using the unchanged LTE reservoir.
            old, new = np.exp(log_old) * reference, np.exp(log_new) * reference
            floors = 1e-12 * reference.sum(axis=1)[:, None]
            expected = np.minimum(1.0, np.maximum(old, new) / floors)
            assert weights.shape == log_old.shape
            assert np.all(np.isfinite(weights))
            assert np.all((weights > 0.0) & (weights <= 1.0))
            assert np.any(weights < 1.0), "fixture must contain below-floor levels"
            np.testing.assert_allclose(weights, expected, rtol=2e-14, atol=0.0)
        else:
            assert weights is None
        calls.append(weights)
        return original(log_old, log_new, history, **kwargs)

    monkeypatch.setattr(hydrogen, "_anderson_log_population_update", checked_fit)
    result = model.solve_populations(atmosphere, accelerated_lambda=accelerated_lambda)
    assert result.iterations == 2
    assert len(calls) == 1
