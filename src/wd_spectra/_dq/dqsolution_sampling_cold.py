"""Cold DQ opacity-sampling experiment using unsmeared thermal Swan lines.

No saved atmosphere, observation, pressure-width fit, or stellar fit enters
the calculation. This is a thermal-only numerical sensitivity experiment,
not a complete dense-fluid model or a proposed default. A structure-grid
certificate must not be confused with independent sampling convergence.

Motivation: MARCS (Gustafsson et al. 2008, A&A 486,951) samples individual
line opacity with trapezoidal wavelength quadrature, and tests different
sampling densities. Averaging opacity before transfer is not equivalent.
"""


import numpy as np


def sampling_grid(lo, hi, resolving_power, phase=0.0):
    if not (0 < lo < hi and resolving_power >= 1000 and 0 <= phase < 1):
        raise ValueError("Invalid opacity-sampling grid")
    count = int(np.ceil(resolving_power * np.log(hi / lo)))
    step = np.log(hi / lo) / count
    inside = lo * np.exp((np.arange(count) + phase) * step)
    return np.unique(np.r_[lo, inside, hi])






