"""Exact-temperature Swan strengths and prefix sums, with a byte-bounded LRU.

No line pruning, rounded temperatures, interpolation, or summation-order change.
Each cache belongs to one immutable line-list evaluator, never a global state.
"""
from collections import OrderedDict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import ndtr
from .dq_swan_depth_integral import SpatiallyIntegratedSwan
from .dqsolution_sampling import TRUNCATION_SIGMA


class CachedSwan(SpatiallyIntegratedSwan):
    def __init__(self, *args, strength_cache_bytes=1024**3, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(strength_cache_bytes) or strength_cache_bytes < 0:
            raise ValueError('Finite nonnegative cache budget required')
        self.strength_cache_bytes = int(strength_cache_bytes)
        self._thermal_cache = OrderedDict()
        self._thermal_bytes = 0
        self.thermal_hits = self.thermal_misses = 0

    def set_strength_cache_bytes(self, budget):
        """Change storage policy only; release entries before new allocations."""
        if not np.isfinite(budget) or budget < 0:
            raise ValueError('Finite nonnegative cache budget required')
        self.strength_cache_bytes = int(budget)
        while self._thermal_cache and self._thermal_bytes > self.strength_cache_bytes:
            _, old = self._thermal_cache.popitem(last=False)
            self._thermal_bytes -= sum(a.nbytes for a in old)

    def _thermal(self, temperature):
        key = float(temperature)
        if key in self._thermal_cache:
            self._thermal_cache.move_to_end(key)
            self.thermal_hits += 1
            return self._thermal_cache[key]
        self.thermal_misses += 1
        strength = super()._line_strength(temperature)
        prefix = np.r_[0., np.cumsum(strength)]
        strength.setflags(write=False)
        prefix.setflags(write=False)
        size = strength.nbytes + prefix.nbytes
        while self._thermal_cache and self._thermal_bytes + size > self.strength_cache_bytes:
            _, old = self._thermal_cache.popitem(last=False)
            self._thermal_bytes -= sum(a.nbytes for a in old)
        if size <= self.strength_cache_bytes:
            self._thermal_cache[key] = (strength, prefix)
            self._thermal_bytes += size
        return strength, prefix

    def _line_strength(self, temperature):
        return self._thermal(temperature)[0]

    def cumulative(self, query_wavenumber, temperature):
        q = np.asarray(query_wavenumber, dtype=float)
        if q.ndim != 1 or np.any(~np.isfinite(q)):
            raise ValueError('Finite 1D wavenumbers required')
        positive = q > 0
        out = np.zeros_like(q)
        if not np.any(positive):
            return out
        mapping = self._query_map(1e8/q[positive], maximum_temperature=temperature)
        velocity = self._velocity_fraction(temperature)
        strength, prefix = self._thermal(temperature)
        index = np.searchsorted(self.nu, q[positive], side='left')
        offset = mapping.fractional_offset
        inside = np.abs(offset) <= TRUNCATION_SIGMA*velocity
        corrections = np.zeros_like(offset)
        corrections[inside] = ndtr(offset[inside]/velocity) - (offset[inside] > 0)
        operator = csr_matrix((corrections, mapping.indices, mapping.indptr),
            shape=(len(mapping.wavelength), len(self.nu)), copy=False)
        out[positive] = prefix[index] + operator @ strength
        return out
