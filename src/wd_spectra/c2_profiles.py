"""DQ-only molecular profiles with widths independent of the sampling grid.

The standard *approximate* molecular neutral broadening prescription is from
SYNSPEC 54 (NSTPAR/PRFQUA/INMOLI): GWSTD=1e-7, He weight=0.42,
temperature exponent=0.45. It is not a measured C2--He coefficient. The
impact approximation itself can fail in dense helium; no DQp shift is made.

Thermal Doppler broadening is applied on log wavenumber. A cell-integrated
Lorentz kernel then supplies the local neutral-He impact width. FFTs compute
a linear, not circular, convolution; neither missing wings nor cut-off
opacity are renormalized. Cached columns only reuse identical local states.
"""

from collections import OrderedDict
from dataclasses import replace
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.ndimage import gaussian_filter1d

from .constants import BOLTZMANN, HYDROGEN_MASS

LIGHT_SPEED = 2.99792458e10
ROTATIONAL_PROFILE_DESCRIPTION = (
    "continuous rotational-branch Swan overlap; integrated line strength retained; "
    "non-Swan systems unchanged; approximate band treatment, not measured C2-He widths"
)
PROFILE_DESCRIPTION = (
    "thermal Doppler + standard SYNSPEC molecular He-impact approximation; "
    "not C2-specific calibrated broadening, no pressure shift"
)
EXOMOL_PROFILE_DESCRIPTION = (
    "thermal Doppler + ExoMol 8states definition defaults (0.07 cm^-1/bar at "
    "296 K, exponent 0.5) applied to neutral-He partial pressure; "
    "generic coefficients, not C2-He measurements; no pressure shift"
)


class C2RotationalProfiles:
    """Select precomputed overlap opacity, with optional local Swan translation.

    No finished-spectrum smoothing and no implicit legacy-table substitute.
    The same object supplies opacity to the structure and the final transfer.
    """

    def __init__(self, table, *, component="total"):
        overlap = table.swan_rotational_overlap_cross_section
        if overlap is None:
            raise ValueError(
                "C2 table lacks rotational_overlap data; rebuild with "
                "python tools/build_c2_opacity.py --output NEW_PATH, or explicitly "
                "select molecular_line_profile='line_bin' for legacy synthesis"
            )
        if component not in ("total", "swan"):
            raise ValueError("unknown C2 opacity component")
        cross = (
            overlap
            if component == "swan"
            else table.cross_section - table.swan_cross_section + overlap
        )
        self.table = replace(
            table,
            cross_section=cross,
            swan_cross_section=overlap,
            swan_rotational_overlap_cross_section=None,
        )

    def cross_section(
        self, wavelength, temperature, neutral_helium_density, *, wavenumber_shift=None
    ):
        w, t, n = map(
            lambda a: np.asarray(a, dtype=float),
            (wavelength, temperature, neutral_helium_density),
        )
        if (
            w.ndim != 1
            or t.ndim != 1
            or n.shape != t.shape
            or np.any(~np.isfinite(n))
            or np.any(n < 0)
        ):
            raise ValueError("invalid molecular opacity wavelength/layer arrays")
        # This validates both grids and retains exactly the existing unshifted
        # table interpolation. No independent discretization change is hidden.
        result = self.table.cross_section_for_wavelength_temperature(w, t)
        if wavenumber_shift is None:
            return result
        shift = np.asarray(wavenumber_shift, dtype=float)
        if shift.shape != t.shape or np.any(~np.isfinite(shift)) or np.any(shift < 0):
            raise ValueError(
                "finite nonnegative per-layer Swan wavenumber shifts required"
            )
        for j, delta in enumerate(shift):
            if delta == 0:
                continue
            nu = 1e8 / w - delta
            use = nu > 0
            result[:, j] = 0
            result[use, j] = self.table.cross_section_for_wavelength_temperature(
                1e8 / nu[use], t[j : j + 1]
            )[:, 0]
        return result


def swan_density_shift_wavenumber(mass_density):
    """Blouin & Dufour 2019: Delta E[eV]=0.2*rho[g/cm3], blueward.

    An empirical published band-energy prescription, not a new fitted
    coefficient and not a complete dense-fluid molecular equation of state.
    """
    rho = np.asarray(mass_density, dtype=float)
    if np.any(~np.isfinite(rho)) or np.any(rho < 0):
        raise ValueError("finite nonnegative density required for Swan shift")
    return 0.2 * rho * 8065.543937


def helium_impact_hwhm(
    temperature, neutral_helium_density, prescription="synspec_standard"
):
    """Lorentz half width in cm^-1; SYNSPEC gamma/(4*pi*c)."""
    t, n = np.broadcast_arrays(temperature, neutral_helium_density)
    if (
        np.any(~np.isfinite(t))
        or np.any(t <= 0)
        or np.any(~np.isfinite(n))
        or np.any(n < 0)
    ):
        raise ValueError(
            "finite positive temperature and nonnegative He density required"
        )
    if prescription == "exomol_default":
        # Defaults in the checksum-pinned 8states .def file; no fitted scale.
        # Reference pressure is bar, not atmosphere.
        return 0.07 * (296.0 / t) ** 0.5 * (n * BOLTZMANN * t / 1e6)
    if prescription != "synspec_standard":
        raise ValueError("unknown molecular impact prescription")
    return 1e-7 * 0.42 * n * (t / 1e4) ** 0.45 / (4 * np.pi * LIGHT_SPEED)


def lorentz_cell_kernel(size, spacing, gamma):
    """Integrated probability of each positive-offset cell, without rescaling."""
    if (
        size < 1
        or not np.isfinite(spacing)
        or spacing <= 0
        or not np.isfinite(gamma)
        or gamma < 0
    ):
        raise ValueError("invalid Lorentz kernel geometry")
    if gamma == 0:
        out = np.zeros(size)
        out[0] = 1
        return out
    k = np.arange(size, dtype=float)
    # atan2 avoids catastrophic cancellation for remote wings.
    return np.arctan2(spacing * gamma, gamma**2 + (k * k - 0.25) * spacing**2) / np.pi


class C2ImpactProfiles:
    """Convolve a line-strength table at the actual layer temperature/density.

    ``wavenumber_step`` is numerical, not a width. Tables must be refined
    independently of this parameter. Large arrays are bounded LRU caches,
    not a density/temperature quantization or a reuse of earlier atmospheres.
    """

    def __init__(
        self,
        table,
        *,
        wavenumber_step=0.5,
        cache_columns=192,
        prescription="synspec_standard",
        component="total",
    ):
        if (
            not np.isfinite(wavenumber_step)
            or wavenumber_step <= 0
            or cache_columns < 0
        ):
            raise ValueError("invalid profile resolution/cache size")
        self.table = table
        if component not in ("total", "swan"):
            raise ValueError("unknown C2 opacity component")
        if component == "swan" and table.swan_cross_section is None:
            raise ValueError(
                "Swan density shifts require a table built with --split-swan"
            )
        self.component = component
        self.strengths = (
            table.cross_section if component == "total" else table.swan_cross_section
        )
        helium_impact_hwhm(7000, 0, prescription)
        self.prescription = prescription
        self.native = 1e8 / table.wavelength_angstrom[::-1]
        self.log_step = np.diff(np.log(self.native))
        if not np.allclose(self.log_step, self.log_step[0], rtol=1e-6, atol=1e-12):
            raise ValueError(
                "C2 impact profiles require a constant-log-wavenumber table"
            )
        self.edges = np.exp(
            np.r_[
                np.log(self.native) - self.log_step[0] / 2,
                np.log(self.native[-1]) + self.log_step[0] / 2,
            ]
        )
        count = int(np.ceil((self.edges[-1] - self.edges[0]) / wavenumber_step))
        self.uniform_edges = np.linspace(self.edges[0], self.edges[-1], count + 1)
        self.nu = (self.uniform_edges[1:] + self.uniform_edges[:-1]) / 2
        self.spacing = self.uniform_edges[1] - self.uniform_edges[0]
        self.fft_size = next_fast_len(2 * count - 1)
        self.sources, self.columns = OrderedDict(), OrderedDict()
        self.cache_columns = cache_columns

    @staticmethod
    def _remember(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def _source_transform(self, index):
        if index in self.sources:
            self.sources.move_to_end(index)
            return self.sources[index]
        sigma_log = (
            np.sqrt(BOLTZMANN * self.table.temperature_K[index] / (24 * HYDROGEN_MASS))
            / LIGHT_SPEED
        )
        # A density per log(nu), not per nu, has a constant Doppler width.
        integrated = self.strengths[::-1, index] * np.diff(self.edges)
        broadened = gaussian_filter1d(
            integrated, sigma_log / self.log_step[0], mode="constant", truncate=6
        )
        cumulative = np.r_[0.0, np.cumsum(broadened)]
        uniform = (
            np.diff(np.interp(self.uniform_edges, self.edges, cumulative))
            / self.spacing
        )
        transformed = rfft(uniform, self.fft_size)
        # A sweep visits cool surface and hot bottom brackets in order. An
        # eight-entry cache repeatedly evicts every useful bracket during a
        # Jacobian build. Retain the finite supplied temperature grid; no
        # approximate temperature/density bucketing is introduced.
        return self._remember(
            self.sources, index, transformed, len(self.table.temperature_K)
        )

    def _column(self, temperature, density):
        key = (float(temperature), float(density))
        if key in self.columns:
            self.columns.move_to_end(key)
            return self.columns[key]
        grid = self.table.temperature_K
        i = int(np.clip(np.searchsorted(grid, temperature) - 1, 0, len(grid) - 2))
        fraction = (temperature - grid[i]) / (grid[i + 1] - grid[i])
        source = (1 - fraction) * self._source_transform(
            i
        ) + fraction * self._source_transform(i + 1)
        positive = lorentz_cell_kernel(
            len(self.nu),
            self.spacing,
            float(helium_impact_hwhm(temperature, density, self.prescription)),
        )
        kernel = np.zeros(self.fft_size)
        kernel[: len(positive)] = positive
        kernel[-len(positive) + 1 :] = positive[1:][::-1]
        values = irfft(source * rfft(kernel), self.fft_size)[: len(self.nu)]
        # Remove floating point negative roundoff only, not physical opacity.
        if np.min(values) < -1e-12 * max(float(np.max(values)), 1e-300):
            raise FloatingPointError("negative broadened C2 opacity")
        values = np.maximum(values, 0)
        return self._remember(self.columns, key, values, self.cache_columns)

    def cross_section(
        self, wavelength, temperature, neutral_helium_density, *, wavenumber_shift=None
    ):
        w, t, n = map(
            lambda a: np.asarray(a, dtype=float),
            (wavelength, temperature, neutral_helium_density),
        )
        if (
            w.ndim != 1
            or t.ndim != 1
            or n.shape != t.shape
            or np.any(~np.isfinite(w))
            or np.any(w <= 0)
        ):
            raise ValueError("invalid molecular opacity wavelength/layer arrays")
        helium_impact_hwhm(t, n)  # validate even if all wavelengths are outside
        shift = (
            np.zeros_like(t)
            if wavenumber_shift is None
            else np.asarray(wavenumber_shift, dtype=float)
        )
        if shift.shape != t.shape or np.any(~np.isfinite(shift)) or np.any(shift < 0):
            raise ValueError(
                "finite nonnegative per-layer Swan wavenumber shifts required"
            )
        if np.any(t < self.table.temperature_K[0]) or np.any(
            t > self.table.temperature_K[-1]
        ):
            raise ValueError(
                "C2 cross-section temperature outside supplied data; no endpoint clamping"
            )
        out = np.empty((len(w), len(t)))
        for j, (tj, nj) in enumerate(zip(t, n)):
            out[:, j] = np.interp(
                1e8 / w - shift[j], self.nu, self._column(tj, nj), left=0, right=0
            )
        return out
