"""DZ preset wiring of the lower-boundary deepening (no physics solve: the solver is stubbed)."""

import numpy as np
import pytest
from wd_spectra.models import stellar
from wd_spectra.models.stellar import DZConfig, compute_dz


class _Stop(Exception):
    pass


def test_dz_deepens_early_on_any_screen_failure_and_restarts(monkeypatch):
    calls = {}

    def solve(solver, *args, **kwargs):
        calls.update(kwargs)
        raise _Stop

    monkeypatch.setattr(stellar, "solve_with_screened_boundary", solve)
    with pytest.raises(_Stop):
        compute_dz(DZConfig(effective_temperature=5000.0, logg=8.0, abundances={"Ca": -10.0}),
                   np.array([4000.0, 5000.0]))
    assert calls["deepen_on_any_screen_failure"] is True
    assert calls["restart_after_deepening"] is True
    assert calls["deepening_pressure_factor"] == stellar._DZ_DEEPENING_PRESSURE_FACTOR > 1.0
    # far above the 0.003 screening tolerance, so mildly leaking bottoms still finish their first solve
    assert 0.003 < calls["early_deepening_escape_limit"] == stellar._DZ_EARLY_DEEPENING_ESCAPE_LIMIT < 0.3
