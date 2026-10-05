"""Hydrogen Stark-profile tables and interpolation.

The bundled data are the Doppler-convolved Tremblay & Bergeron (2009) tables,
updated in 2015 and distributed under CC BY 4.0.  Their on-disk layout follows
Lemke (1997).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
import re

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StarkLine:
    lower_level: int
    upper_level: int
    log_alpha: FloatArray
    log_electron_density: FloatArray
    log_temperature: FloatArray
    log_profile: FloatArray
    goodness_flag: NDArray[np.int64]

    def _local_profile_state(
        self,
        temperature: float,
        electron_density: float,
    ) -> tuple[float, FloatArray]:
        """Return the local field scale and interpolated log-profile.

        Keeping this state construction separate lets the neutral-broadening
        convolution reuse it for every quadrature abscissa.  The interpolation
        is identical to :meth:`wavelength_profile`; only redundant work is
        removed. Below the density grid, keep the whole Doppler-convolved
        boundary profile on its tabulated physical wavelength scale.
        """

        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(electron_density) or electron_density <= 0.0:
            raise ValueError("electron_density must be finite and positive")

        log_ne = float(
            np.clip(
                np.log10(electron_density),
                self.log_electron_density[0],
                self.log_electron_density[-1],
            )
        )
        log_t = float(
            np.clip(
                np.log10(temperature),
                self.log_temperature[0],
                self.log_temperature[-1],
            )
        )
        ne_lower, ne_fraction = _bracket(self.log_electron_density, log_ne)
        t_lower, t_fraction = _bracket(self.log_temperature, log_t)
        local_log_profile = np.zeros_like(self.log_alpha)
        for ne_index, ne_weight in (
            (ne_lower, 1.0 - ne_fraction),
            (ne_lower + 1, ne_fraction),
        ):
            for t_index, t_weight in (
                (t_lower, 1.0 - t_fraction),
                (t_lower + 1, t_fraction),
            ):
                local_log_profile += (
                    ne_weight
                    * t_weight
                    * self.log_profile[ne_index, t_index]
                )
        # The tables already include thermal Doppler broadening. Scaling
        # their lowest-density shape with a smaller, out-of-table ne would
        # squeeze that thermal core toward zero width. Preserve the entire
        # physical edge profile below the grid; in-range and high-density
        # field scaling retain their previous floating-point operations.
        field_density = max(electron_density, 10.0 ** self.log_electron_density[0])
        field_strength = 1.25e-9 * field_density ** (2.0 / 3.0)
        return field_strength, local_log_profile

    def wavelength_profile(
        self,
        wavelength_angstrom: ArrayLike,
        line_center_angstrom: float,
        temperature: float,
        electron_density: float,
    ) -> FloatArray:
        r"""Return the normalized symmetric profile per Angstrom.

        Tables contain ``log10 S(alpha)`` with
        ``alpha = abs(delta_lambda) / F0`` and
        ``F0 = 1.25e-9 ne**(2/3)``.  Bilinear interpolation is performed in
        log electron density and log temperature, and linear interpolation in
        log alpha/log profile. Temperature queries use the boundary shape
        outside its grid. Below the density grid the complete physical
        wavelength profile is held at the density edge. Above the grid the
        boundary shape retains the existing ne**(2/3) field extrapolation.
        These edge policies do not provide new out-of-domain Stark data.
        """

        wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
        if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0):
            raise ValueError("wavelength must contain finite positive values")
        field_strength, local_log_profile = self._local_profile_state(
            temperature, electron_density
        )
        alpha = np.abs(wavelength - line_center_angstrom) / field_strength
        with np.errstate(divide="ignore"):
            query_log_alpha = np.log10(alpha)
        query_shape = query_log_alpha.shape
        flat_log_alpha = query_log_alpha.ravel()

        # Bilinear interpolation in log(ne), log(T) commutes with the linear
        # interpolation in log(alpha), because every corner uses the same
        # alpha grid.  Form the one local log-profile first instead of calling
        # np.interp four times for every query (particularly costly inside the
        # neutral-H Cauchy convolution).  This is algebraically the same
        # interpolation, apart from roundoff in the order of additions.
        log_values = np.interp(
            flat_log_alpha,
            self.log_alpha,
            local_log_profile,
            left=local_log_profile[0],
            right=local_log_profile[-1],
        )
        beyond_table = flat_log_alpha > self.log_alpha[-1]
        wing_slope = (
            (local_log_profile[-1] - local_log_profile[-2])
            / (self.log_alpha[-1] - self.log_alpha[-2])
        )
        log_values[beyond_table] = local_log_profile[-1] + wing_slope * (
            flat_log_alpha[beyond_table] - self.log_alpha[-1]
        )
        profile = np.power(10.0, log_values) / field_strength
        return profile.reshape(query_shape)


def _bracket(grid: FloatArray, value: float) -> tuple[int, float]:
    upper = int(np.searchsorted(grid, value, side="right"))
    lower = min(max(upper - 1, 0), grid.size - 2)
    fraction = (value - grid[lower]) / (grid[lower + 1] - grid[lower])
    return lower, float(np.clip(fraction, 0.0, 1.0))


@dataclass(frozen=True)
class HydrogenStarkTable:
    lines: dict[tuple[int, int], StarkLine]
    source_path: Path

    def __getitem__(self, transition: tuple[int, int]) -> StarkLine:
        return self.lines[transition]


def read_stark_table(path: str | Path) -> HydrogenStarkTable:
    """Read a Lemke-layout hydrogen Stark table."""

    path = Path(path)
    text_lines = path.read_text(encoding="ascii").splitlines()
    n_lines = int(text_lines[0])
    cursor = 1
    headers = []
    for _ in range(n_lines):
        tokens: list[str] = []
        while len(tokens) < 11:
            tokens.extend(text_lines[cursor].split())
            cursor += 1
        if len(tokens) != 11:
            raise ValueError(f"invalid Stark-table header in {path}")
        headers.append(tokens)

    profiles: dict[tuple[int, int], StarkLine] = {}
    for tokens in headers:
        lower, upper = int(tokens[0]), int(tokens[1])
        marker = re.fullmatch(r"nl=\s*(\d+)\s+nu=\s*(\d+)", text_lines[cursor].strip())
        cursor += 1
        if marker is None or (int(marker.group(1)), int(marker.group(2))) != (lower, upper):
            raise ValueError(f"inconsistent transition marker in {path}")

        log_alpha_min, log_ne_min, log_t_min = map(float, tokens[2:5])
        log_alpha_step, log_ne_step, log_t_step = map(float, tokens[5:8])
        n_alpha, n_ne, n_t = map(int, tokens[8:11])
        expected = n_ne * n_t * (n_alpha + 1)
        values: list[str] = []
        while len(values) < expected:
            values.extend(text_lines[cursor].split())
            cursor += 1
        if len(values) != expected:
            raise ValueError(f"invalid profile block for H {lower}->{upper}")
        block = np.asarray(values, dtype=np.float64).reshape(n_ne, n_t, n_alpha + 1)
        profiles[(lower, upper)] = StarkLine(
            lower_level=lower,
            upper_level=upper,
            log_alpha=log_alpha_min + log_alpha_step * np.arange(n_alpha),
            log_electron_density=log_ne_min + log_ne_step * np.arange(n_ne),
            log_temperature=log_t_min + log_t_step * np.arange(n_t),
            log_profile=block[:, :, 1:],
            goodness_flag=block[:, :, 0].astype(np.int64),
        )
    return HydrogenStarkTable(lines=profiles, source_path=path)


@lru_cache(maxsize=1)
def default_balmer_stark_table() -> HydrogenStarkTable:
    """Load the bundled Doppler-convolved Balmer table."""

    resource = files("wd_spectra").joinpath("data/stark/tremblay_balmer_conv")
    with resource.open("rb") as stream:
        # A real filesystem path is available for normal and editable installs.
        path = Path(stream.name)
    return read_stark_table(path)


@lru_cache(maxsize=1)
def default_lyman_stark_table() -> HydrogenStarkTable:
    """Load the bundled Doppler-convolved Lyman table."""

    resource = files("wd_spectra").joinpath("data/stark/tremblay_lyman_conv")
    with resource.open("rb") as stream:
        path = Path(stream.name)
    return read_stark_table(path)


@lru_cache(maxsize=1)
def default_paschen_stark_table() -> HydrogenStarkTable:
    """Load the bundled Doppler-convolved Paschen table."""

    resource = files("wd_spectra").joinpath(
        "data/stark/tremblay_paschen_conv"
    )
    with resource.open("rb") as stream:
        path = Path(stream.name)
    return read_stark_table(path)


@lru_cache(maxsize=1)
def default_brackett_stark_table() -> HydrogenStarkTable:
    """Load the bundled Doppler-convolved Brackett table."""

    resource = files("wd_spectra").joinpath(
        "data/stark/tremblay_brackett_conv"
    )
    with resource.open("rb") as stream:
        path = Path(stream.name)
    return read_stark_table(path)
