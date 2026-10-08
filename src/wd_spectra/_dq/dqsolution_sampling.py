from __future__ import annotations

"""Sparse, exact-line thermal Swan opacity for atmosphere feedback tests.

This research-only evaluator replaces the pre-binned Swan component by every
raw ExoMol 8states Swan transition evaluated with a thermal Gaussian,

    sigma_nu = nu_0 sqrt(k T / (24 m_H)) / c.

It deliberately includes no collision broadening or pressure shift and is not
a proposed physical default.  The other C2 systems retain their tabulated
baseline opacity.  A query/line CSR incidence map is constructed once using a
declared ceiling temperature; individual columns use their actual temperature
and discard the unnormalised Gaussian tail beyond eight sigma.  The omitted
two-sided Gaussian area is only erfc(8/sqrt(2)) = 1.24e-15 per line.

Returned columns are cached by the exact temperature and wavelength-grid
fingerprint.  Thus a finite-difference Jacobian that changes one layer only
evaluates one new molecular column.  No temperature or wavelength bucketing is
used.
"""


from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from wd_spectra.constants import BOLTZMANN, HYDROGEN_MASS


LIGHT_SPEED = 2.99792458e10
SECOND_RADIATION_CONSTANT_CM_K = 1.438776877
TRUNCATION_SIGMA = 8.0
GAUSSIAN_TAIL_AREA_BOUND = math.erfc(TRUNCATION_SIGMA / math.sqrt(2.0))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _grid_fingerprint(wavelength: np.ndarray) -> str:
    values = np.ascontiguousarray(wavelength, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class _QueryLineMap:
    wavelength: np.ndarray
    wavenumber: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    fractional_offset: np.ndarray
    fingerprint: str


class SparseResolvedSwan:
    """Return baseline non-Swan C2 plus all raw Swan thermal Gaussians.

    Parameters
    ----------
    table
        The original complete C2 cross-section table.  Its molecular partition
        function is used directly, without a Swan-only partition convention.
    maximum_temperature
        Ceiling used only to preselect possible query/line overlaps.  The
        default is the top of the supplied opacity-temperature grid, so the
        one-argument ``SparseResolvedSwan(table)`` interface is safe over the
        whole table.  A tighter explicitly declared ceiling saves memory.
    """

    def __init__(
        self,
        table,
        maximum_temperature: float | None = None,
        *,
        branches: Path,
        cache_columns: int = 192,
        cache_grids: int = 4,
        cache_map_bytes: int = 64 * 1024**2,
    ):
        if table.swan_cross_section is None:
            raise ValueError("split Swan opacity is required")
        if cache_columns < 0 or cache_grids < 1 or cache_map_bytes < 0:
            raise ValueError("invalid sparse-Swan cache size")
        branches = Path(branches)
        with np.load(branches, allow_pickle=False) as archive:
            lines = np.asarray(archive["lines"][:3], dtype=float)
        if (
            lines.ndim != 2
            or lines.shape[0] != 3
            or lines.shape[1] == 0
            or np.any(~np.isfinite(lines))
            or np.any(lines[0] <= 0)
            or np.any(lines[1] < 0)
            or np.any(lines[2] < 0)
        ):
            raise ValueError("invalid raw Swan line cache")
        order = np.argsort(lines[0], kind="stable")
        self.nu = np.ascontiguousarray(lines[0, order])
        self.ag = np.ascontiguousarray(lines[1, order])
        self.lower_energy = np.ascontiguousarray(lines[2, order])
        for values in (self.nu, self.ag, self.lower_energy):
            values.setflags(write=False)

        maximum = (
            float(table.temperature_K[-1])
            if maximum_temperature is None
            else float(maximum_temperature)
        )
        if (
            not np.isfinite(maximum)
            or maximum <= 0
            or maximum < table.temperature_K[0]
            or maximum > table.temperature_K[-1]
            or maximum > table.partition_temperature_K[-1]
        ):
            raise ValueError("maximum temperature outside supplied C2 data")
        self.table = table
        self.maximum_temperature = maximum
        self.maximum_velocity_fraction = self._velocity_fraction(maximum)
        if TRUNCATION_SIGMA * self.maximum_velocity_fraction >= 1:
            raise ValueError("temperature ceiling invalidates overlap bounds")

        non_swan = table.cross_section - table.swan_cross_section
        tolerance = 1e-12 * np.maximum(table.cross_section, 1e-300)
        if np.any(non_swan < -tolerance):
            raise ValueError("Swan opacity is not a subset of total C2 opacity")
        self.non_swan = np.maximum(non_swan, 0.0)
        self.non_swan.setflags(write=False)
        self.cache_columns = int(cache_columns)
        self.cache_grids = int(cache_grids)
        self.cache_map_bytes = int(cache_map_bytes)
        self._maps: OrderedDict[str, _QueryLineMap] = OrderedDict()
        self._columns: OrderedDict[tuple[str, float], np.ndarray] = OrderedDict()
        self._stats = dict(
            map_builds=0,
            map_hits=0,
            column_hits=0,
            column_misses=0,
            gaussian_pairs_evaluated=0,
        )

    @staticmethod
    def _velocity_fraction(temperature: float) -> float:
        return float(
            np.sqrt(BOLTZMANN * temperature / (24.0 * HYDROGEN_MASS))
            / LIGHT_SPEED
        )

    @staticmethod
    def _remember(cache: OrderedDict, key, value, limit: int):
        if limit == 0:
            return value
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _query_map(
        self, wavelength: np.ndarray, maximum_temperature: float | None = None
    ) -> _QueryLineMap:
        maximum = (
            self.maximum_temperature
            if maximum_temperature is None
            else float(maximum_temperature)
        )
        if (
            not np.isfinite(maximum)
            or maximum < self.table.temperature_K[0]
            or maximum > self.maximum_temperature
        ):
            raise ValueError("query-map temperature outside declared range")
        grid_fingerprint = _grid_fingerprint(wavelength)
        fingerprint = (
            grid_fingerprint
            if maximum == self.maximum_temperature
            else f"{grid_fingerprint}:{maximum.hex()}"
        )
        cached = self._maps.get(fingerprint)
        if cached is not None:
            # The cryptographic fingerprint is the cache key; the exact check
            # also makes accidental in-process misuse fail deterministically.
            if not np.array_equal(cached.wavelength, wavelength):
                raise RuntimeError("wavelength-grid fingerprint collision")
            self._maps.move_to_end(fingerprint)
            self._stats["map_hits"] += 1
            return cached

        query = 1e8 / wavelength
        extent = TRUNCATION_SIGMA * self._velocity_fraction(maximum)
        lower = np.searchsorted(self.nu, query / (1.0 + extent), side="left")
        upper = np.searchsorted(self.nu, query / (1.0 - extent), side="right")
        counts = upper - lower
        indptr = np.empty(len(query) + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        indices = np.empty(int(indptr[-1]), dtype=np.int32)
        cursor = 0
        for lo, hi in zip(lower, upper):
            count = int(hi - lo)
            if count:
                indices[cursor : cursor + count] = np.arange(lo, hi, dtype=np.int32)
                cursor += count
        repeated_query = np.repeat(query, counts)
        centers = self.nu[indices]
        fractional_offset = (repeated_query - centers) / centers
        saved_wave = np.array(wavelength, copy=True)
        saved_wave.setflags(write=False)
        for values in (query, indptr, indices, fractional_offset):
            values.setflags(write=False)
        result = _QueryLineMap(
            wavelength=saved_wave,
            wavenumber=query,
            indptr=indptr,
            indices=indices,
            fractional_offset=fractional_offset,
            fingerprint=fingerprint,
        )
        self._stats["map_builds"] += 1
        self._remember(self._maps, fingerprint, result, self.cache_grids)
        # Shifted query grids can contain very different numbers of line
        # overlaps. A count-only cache is not a bound on memory. Eviction
        # changes reuse only; the exact newly computed map is still returned.
        def map_bytes(value):
            return sum(x.nbytes for x in (
                value.wavelength,value.wavenumber,value.indptr,
                value.indices,value.fractional_offset))
        retained_bytes=sum(map_bytes(value) for value in self._maps.values())
        while retained_bytes>self.cache_map_bytes:
            _,removed=self._maps.popitem(last=False)
            retained_bytes-=map_bytes(removed)
        self._stats['map_cache_bytes']=retained_bytes
        # Removing a grid must also remove its dependent returned columns.
        retained = set(self._maps)
        for key in tuple(self._columns):
            if key[0] not in retained:
                del self._columns[key]
        return result

    def _line_strength(self, temperature: float) -> np.ndarray:
        partition = float(self.table.molecular_partition_function(temperature))
        strength = (
            self.ag
            * np.exp(
                -SECOND_RADIATION_CONSTANT_CM_K * self.lower_energy / temperature
            )
            * -np.expm1(-SECOND_RADIATION_CONSTANT_CM_K * self.nu / temperature)
            / (8.0 * np.pi * LIGHT_SPEED * self.nu**2 * partition)
        )
        if np.any(~np.isfinite(strength)) or np.any(strength < 0):
            raise FloatingPointError("invalid raw Swan integrated strength")
        return strength

    def _non_swan_column(self, wavelength: np.ndarray, temperature: float) -> np.ndarray:
        grid = self.table.temperature_K
        index = int(np.clip(np.searchsorted(grid, temperature) - 1, 0, len(grid) - 2))
        fraction = (temperature - grid[index]) / (grid[index + 1] - grid[index])
        low = np.interp(
            wavelength,
            self.table.wavelength_angstrom,
            self.non_swan[:, index],
            left=0.0,
            right=0.0,
        )
        high = np.interp(
            wavelength,
            self.table.wavelength_angstrom,
            self.non_swan[:, index + 1],
            left=0.0,
            right=0.0,
        )
        return (1.0 - fraction) * low + fraction * high

    def _swan_column_from_map(self, mapping: _QueryLineMap, temperature: float) -> np.ndarray:
        velocity = self._velocity_fraction(temperature)
        offset = mapping.fractional_offset
        inside = np.abs(offset) <= TRUNCATION_SIGMA * velocity
        data = np.zeros_like(offset)
        selected = mapping.indices[inside]
        data[inside] = (
            np.exp(-0.5 * (offset[inside] / velocity) ** 2)
            / (np.sqrt(2.0 * np.pi) * self.nu[selected] * velocity)
        )
        operator = csr_matrix(
            (data, mapping.indices, mapping.indptr),
            shape=(len(mapping.wavelength), len(self.nu)),
            copy=False,
        )
        self._stats["gaussian_pairs_evaluated"] += int(np.count_nonzero(inside))
        result = np.asarray(operator @ self._line_strength(temperature)).ravel()
        if np.any(~np.isfinite(result)) or np.any(result < 0):
            raise FloatingPointError("invalid sparse raw Swan cross section")
        return result

    def swan_column(self, wavelength, temperature: float) -> np.ndarray:
        """Evaluate only the raw Swan component for one exact temperature."""
        w = np.asarray(wavelength, dtype=float)
        t = float(temperature)
        if w.ndim != 1 or np.any(~np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("wavelength must be finite, positive and one-dimensional")
        if (
            not np.isfinite(t)
            or t < self.table.temperature_K[0]
            or t > self.maximum_temperature
            or t < self.table.partition_temperature_K[0]
        ):
            raise ValueError("temperature outside declared sparse-Swan range")
        return self._swan_column_from_map(
            self._query_map(w, maximum_temperature=t), t
        )

    def _column(self, mapping: _QueryLineMap, temperature: float) -> np.ndarray:
        key = (mapping.fingerprint, float(temperature))
        cached = self._columns.get(key)
        if cached is not None:
            self._columns.move_to_end(key)
            self._stats["column_hits"] += 1
            return cached
        self._stats["column_misses"] += 1
        result = self._non_swan_column(mapping.wavelength, temperature)
        result += self._swan_column_from_map(mapping, temperature)
        result.setflags(write=False)
        return self._remember(self._columns, key, result, self.cache_columns)

    def cross_section(
        self,
        wavelength,
        temperature,
        neutral_helium_density,
        *,
        wavenumber_shift=None,
    ) -> np.ndarray:
        """C2 cross section on arbitrary samples, one column per layer.

        Neutral-He density is validated for interface compatibility but does
        not enter this thermal-only diagnostic.  Pressure shifts are rejected.
        """
        w = np.asarray(wavelength, dtype=float)
        t = np.asarray(temperature, dtype=float)
        n = np.asarray(neutral_helium_density, dtype=float)
        if (
            w.ndim != 1
            or t.ndim != 1
            or n.shape != t.shape
            or np.any(~np.isfinite(w))
            or np.any(w <= 0)
            or np.any(~np.isfinite(t))
            or np.any(~np.isfinite(n))
            or np.any(n < 0)
        ):
            raise ValueError("invalid sparse-Swan wavelength/layer arrays")
        if (
            np.any(t < self.table.temperature_K[0])
            or np.any(t > self.maximum_temperature)
            or np.any(t < self.table.partition_temperature_K[0])
            or np.any(t > self.table.partition_temperature_K[-1])
        ):
            raise ValueError("temperature outside declared sparse-Swan range")
        if wavenumber_shift is not None:
            shift = np.asarray(wavenumber_shift, dtype=float)
            if shift.shape != t.shape or np.any(~np.isfinite(shift)) or np.any(shift != 0):
                raise ValueError("this thermal-only diagnostic does not implement shifts")
        mapping = self._query_map(w, maximum_temperature=float(np.max(t)))
        output = np.empty((len(w), len(t)))
        for column, value in enumerate(t):
            output[:, column] = self._column(mapping, float(value))
        return output

    def diagnostics(self) -> dict:
        """Return immutable-value geometry, truncation and cache diagnostics."""
        return {
            **self._stats,
            "raw_swan_transitions": int(len(self.nu)),
            "maximum_temperature_K": self.maximum_temperature,
            "maximum_velocity_fraction": self.maximum_velocity_fraction,
            "gaussian_truncation_sigma": TRUNCATION_SIGMA,
            "gaussian_omitted_area_bound_per_line": GAUSSIAN_TAIL_AREA_BOUND,
            "cached_grids": len(self._maps),
            "cached_columns": len(self._columns),
            "grid_nonzero_pairs": {
                key: int(value.indices.size) for key, value in self._maps.items()
            },
        }







