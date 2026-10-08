"""One smooth H2 partition potential for mass action AND thermal energy.

Research-only SciPy interpolation. Published Barklem--Collet nodes are exact;
there is no atmospheric fit. The constant ground-state limit is retained and
the upper log-temperature tangent is an explicit extrapolation, not data.
"""
from contextlib import contextmanager, ExitStack
from functools import lru_cache
from unittest.mock import patch
import numpy as np
from scipy.interpolate import make_interp_spline
from wd_spectra import molecules, eos
from wd_spectra.constants import BOLTZMANN


@lru_cache(maxsize=1)
def potential():
    t, q = molecules._molecular_hydrogen_partition_table()
    first = np.flatnonzero(q != q[0])[0] - 1
    x, y = np.log(t[first:]), np.log(q[first:])
    curve = make_interp_spline(
        x, y, k=5, bc_type=([(1, 0.0), (2, 0.0)], [(2, 0.0), (3, 0.0)])
    )
    return x, y, curve


def partition_potential(t, derivative=0):
    t = molecules._positive_temperature(t)
    x, y, curve = potential()
    z = np.log(t)
    if derivative == 0:
        value = curve(np.clip(z, x[0], x[-1]))
        return np.where(z > x[-1], y[-1] + curve(x[-1], 1) * (z - x[-1]), value)
    value = curve(np.clip(z, x[0], x[-1]), derivative)
    value = np.where(z < x[0], 0.0, value)
    return np.where(z > x[-1], curve(x[-1], 1) if derivative == 1 else 0.0, value)


def partition(t):
    return np.exp(partition_potential(t))


def energy(t):
    return BOLTZMANN * np.asarray(t) * partition_potential(t, 1)


@contextmanager
def smooth_h2_experiment():
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                molecules, "molecular_hydrogen_internal_partition_function", partition
            )
        )
        for module in (molecules, eos):
            stack.enter_context(
                patch.object(module, "molecular_hydrogen_rovibrational_energy", energy)
            )
        yield
