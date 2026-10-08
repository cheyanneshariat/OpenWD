"""Keep the wavelength scale consistent with a density-clipped Stark table.

The data already include Doppler broadening. Holding their S(alpha) fixed
while scaling alpha with an out-of-range electron density spuriously sends
the thermal width to zero. This diagnostic instead holds the entire edge
profile fixed below the density grid. It is a table-edge approximation, not a new low-density Stark
calculation; the temperature clipping and table normalization are unchanged.
Inside the supplied density domain the original calculation is exact.
"""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
from wd_spectra.stark import StarkLine


@contextmanager
def consistent_stark_density_edge():
    original=StarkLine._local_profile_state
    def state(line,temperature,electron_density):
        # Preserve validation and every in-domain floating-point operation.
        if not np.isfinite(electron_density) or electron_density<=0:
            return original(line,temperature,electron_density)
        lower=10.**line.log_electron_density[0]
        return original(line,temperature,max(electron_density,lower))
    with patch.object(StarkLine,'_local_profile_state',state):
        yield
