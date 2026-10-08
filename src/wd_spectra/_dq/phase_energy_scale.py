"""Fixed energy units with explicit, between-solve refreshes only."""
from contextvars import ContextVar
import numpy as np

SCALES = ContextVar('dq_phase_energy_units', default=None)


def energy_units(payload):
    cv = payload['dq_augmented_auxiliary_flux']
    units = payload['thermal_cell_emission']+cv[:-1]+cv[1:]
    if np.any(~np.isfinite(units)) or np.any(units <= 0):
        raise ValueError('Positive finite energy units required')
    return units.copy()


def refresh_energy_units(system, payload):
    scales = SCALES.get()
    if scales is None:
        raise RuntimeError('Energy-unit refresh outside a phase experiment')
    scales[np.asarray(system.pressure).tobytes()] = energy_units(payload)


def scale_drift(payload):
    old = payload['dq_fixed_energy_scale']
    new = energy_units(payload)
    return float(np.max(abs(np.log(new/old))))


