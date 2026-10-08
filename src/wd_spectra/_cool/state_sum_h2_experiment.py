"""H2 Q and energy from the SAME published bound-state sum (research only).

Roueff et al. 2019, A&A 630 A58; ExoMol RACPPK. Cached data are unmodified
CC BY-SA 4.0 source data. Ground electronic state only, no quasibound levels
or nonideal level dissolution. No atmospheric fit, interpolation or T clamp.
"""
from contextlib import contextmanager, ExitStack
from functools import lru_cache
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from wd_spectra import molecules, eos
from wd_spectra.constants import BOLTZMANN, PLANCK, LIGHT_SPEED

from .research_paths import data_directory
DATA = data_directory() / "1H2__RACPPK.states.bz2"
DATA_SHA256 = "276f5a36d094e7e1417c44f11c1d173417e92629c5f747b6a862bcce5c374a81"


@lru_cache(maxsize=1)
def states():
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("RACPPK state data differ from the validated public source")
    data = np.loadtxt(DATA)
    e, g, j = data[:, 1], data[:, 2], data[:, 3].astype(int)
    if (
        len(data) != 302
        or e.min() != 0
        or np.any(e < 0)
        or not np.array_equal(g, (2 * j + 1) * np.where(j % 2, 3, 1))
    ):
        raise ValueError("Unexpected RACPPK bound-state data or statistical weights")
    levels, weights = PLANCK * LIGHT_SPEED * e / BOLTZMANN, g / 4.0
    levels.setflags(write=False)
    weights.setflags(write=False)
    return levels, weights


def moments(t):
    t = molecules._positive_temperature(t)
    e, g = states()
    w = g * np.exp(-e / t[..., None])
    q = np.sum(w, axis=-1)
    mean = np.sum(w * e, axis=-1) / q
    variance = np.sum(w * (e - mean[..., None]) ** 2, axis=-1) / q
    return q, BOLTZMANN * mean, BOLTZMANN * variance / t ** 2


def partition(t):
    t = molecules._positive_temperature(t)
    e, g = states()
    return np.sum(g * np.exp(-e / t[..., None]), axis=-1)


def energy(t):
    return moments(t)[1]


@contextmanager
def state_sum_h2_experiment():
    states()  # Missing data is an explicit failure before any atmosphere run.
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
