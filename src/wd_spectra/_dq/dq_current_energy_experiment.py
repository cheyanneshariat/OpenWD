"""Opt-in current-energy normalization, retaining physical equations/gates.

Use the existing explicit-gradient system's fully differentiated current
thermal-plus-convective units instead of the fixed-phase row experiment.
All patches are process-local. The normal cold path retains conditioning;
--direct-steady skips conditioning only for the initially imported saved
state, not later domain extensions. --steady-update-budget is an explicit
saved-state test limit.
"""
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

from . import phase_energy_scale as phase_units
from .dq_explicit_gradient import ExplicitGradientSystem


@contextmanager
def current_energy_phase_rows():
    """Keep the native current-unit equations and quotient-rule Jacobian."""
    original = ExplicitGradientSystem.evaluate
    token = phase_units.SCALES.set({})

    def evaluate(system, state, need):
        ev = original(system, state, need)
        units = phase_units.energy_units(ev.payload)
        # The phase controller also refreshes auxiliary coordinate units.
        # Its legacy energy-drift field must describe our CURRENT units;
        # there is no frozen energy weight to refresh in this experiment.
        return replace(ev, payload={**ev.payload,
            'dq_fixed_energy_scale': units,
            'dq_equation_form': 'integral flux minus local Q/current energy units',
            'dq_current_energy_units': True})

    try:
        with patch.object(ExplicitGradientSystem, 'evaluate', evaluate):
            yield
    finally:
        phase_units.SCALES.reset(token)








