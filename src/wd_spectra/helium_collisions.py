"""Optional Storey--Hummer/Berrington--Kingston He I collision fits.

TLUSTY distributes a 929-coefficient fit in its official Fortran source, but
that archive does not include an explicit redistribution license.  This
module therefore contains no copied coefficient table.  It reads a local,
user-obtained TLUSTY source file and evaluates the published fit in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import BOLTZMANN


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_FINE_ENERGY_EV = np.asarray(
    [
        0.0,
        19.8198,
        20.6160,
        20.96432,
        21.2182,
        22.7187,
        22.9206,
        23.00731,
        23.0739,
        23.0743,
        23.0873,
        23.5942,
        23.6738,
        23.7081,
        23.7363,
        23.7366,
        23.7373,
        23.7373,
        23.7423,
    ],
    dtype=np.float64,
)
_FINE_STATISTICAL_WEIGHT = np.asarray(
    [1, 3, 1, 9, 3, 3, 1, 9, 15, 5, 3, 3, 1, 9, 15, 5, 21, 7, 3],
    dtype=np.float64,
)

# The first five TLUSTY-14 terms are explicit LS states.  Its n=3 and n=4
# singlet/triplet terms are statistical-weighted groups of the 19-state fit.
_TERM_FINE_INDEX = (
    (0,),
    (1,),
    (2,),
    (3,),
    (4,),
    (5, 7, 8),
    (6, 10, 9),
    (11, 13, 14, 16),
    (12, 18, 15, 17),
)


def _fortran_numbers(text: str) -> FloatArray:
    # Fixed-form continuation markers occupy column six.  Some NSTART lines
    # use ``.177`` without an intervening blank, which must not be parsed as
    # the decimal number 0.177.
    text = re.sub(r"(?m)^\s*[.*]\s*", "", text)
    pattern = re.compile(
        r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[dDeE][+-]?\d+)?"
    )
    return np.asarray(
        [float(value.replace("D", "E").replace("d", "e")) for value in pattern.findall(text)],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class TlustyHeliumCollisionData:
    """Storey--Hummer polynomial data extracted from official TLUSTY source."""

    fit_start_index: IntArray
    coefficient: FloatArray
    source_path: str
    ionization_scale: FloatArray | None = None

    def fine_structure_rate_matrix(
        self, temperature: ArrayLike
    ) -> FloatArray:
        """Return 19-state excitation/de-excitation coefficients in cm3/s.

        Outside the polynomial's Chebyshev coordinate interval (1,000 to
        50,000 K), hold its effective collision strength at the endpoint.
        The kinetic T**-1/2 and Boltzmann factors still use the actual T.
        This explicit extrapolation prevents negative polynomial rates in
        hot deep layers; it is not additional high-temperature atomic data.
        """

        temperature_array = np.asarray(temperature, dtype=np.float64)
        if (
            np.any(~np.isfinite(temperature_array))
            or np.any(temperature_array <= 0.0)
        ):
            raise ValueError("temperature must be finite and positive")
        flat_temperature = temperature_array.ravel()
        rate = np.zeros((flat_temperature.size, 19, 19), dtype=np.float64)
        reduced_temperature = (
            2.0 * (np.log10(np.clip(flat_temperature, 1.e3, 5.e4)) - 3.849485) / 0.849485002
        )
        temperature_factor = 1.0 / np.sqrt(flat_temperature)
        transition = 0
        for upper in range(1, 19):
            for lower in range(upper):
                start = int(self.fit_start_index[transition]) - 1
                stop = int(self.fit_start_index[transition + 1]) - 1
                fit = self.coefficient[start:stop]
                n_term = fit.size
                if n_term < 3:
                    raise ValueError("TLUSTY He I fit has fewer than three terms")
                work = np.zeros((n_term + 1, flat_temperature.size))
                work[n_term - 1] = fit[-1]
                work[n_term - 2] = (
                    reduced_temperature * work[n_term - 1] + fit[-2]
                )
                for index in range(n_term - 3, -1, -1):
                    work[index] = (
                        reduced_temperature * work[index + 1]
                        - work[index + 2]
                        + fit[index]
                    )
                downward = (work[0] - work[2]) * temperature_factor
                upward = (
                    downward
                    * _FINE_STATISTICAL_WEIGHT[upper]
                    / _FINE_STATISTICAL_WEIGHT[lower]
                    * np.exp(
                        (_FINE_ENERGY_EV[lower] - _FINE_ENERGY_EV[upper])
                        / (8.62e-5 * flat_temperature)
                    )
                )
                rate[:, upper, lower] = downward
                rate[:, lower, upper] = upward
                transition += 1
        return rate.reshape(temperature_array.shape + (19, 19))

    def term_rate_matrix(
        self,
        temperature: ArrayLike,
        term_groups: tuple[tuple[int, ...], ...] = _TERM_FINE_INDEX,
        n_terms: int = 14,
    ) -> FloatArray:
        """Return coefficients for the leading terms built from the 19 LS states.

        ``term_groups`` lists, for each leading model-atom term, its constituent
        COLLHE states; the default is the first nine TLUSTY-14 He I terms.  A
        24-term atom whose first 19 terms are the LS states uses singletons.

        A rate out of an averaged lower term is weighted by the constituent
        statistical populations; rates into an averaged upper term are
        summed.  This is the same averaging implemented by TLUSTY's CHEAV and
        CHEAVJ routines.
        """

        fine = self.fine_structure_rate_matrix(temperature)
        output = np.zeros(fine.shape[:-2] + (n_terms, n_terms), dtype=np.float64)
        for lower_term, lower_group in enumerate(term_groups):
            lower_weight = np.sum(_FINE_STATISTICAL_WEIGHT[list(lower_group)])
            for upper_term in range(lower_term + 1, len(term_groups)):
                upper_group = term_groups[upper_term]
                upward = np.zeros(fine.shape[:-2], dtype=np.float64)
                for lower in lower_group:
                    weight = _FINE_STATISTICAL_WEIGHT[lower] / lower_weight
                    upward += weight * np.sum(
                        fine[..., lower, list(upper_group)], axis=-1
                    )
                output[..., lower_term, upper_term] = upward
        return output

    def ionization_rate_coefficient(
        self, temperature: ArrayLike, binding_energy_erg: ArrayLike
    ) -> FloatArray:
        """Return TLUSTY/Mihalas He I ionization coefficients in cm3/s."""

        if self.ionization_scale is None:
            raise ValueError(
                "ionization coefficients require a TLUSTY He I atom file"
            )
        temperature_array = np.asarray(temperature, dtype=np.float64)
        binding = np.asarray(binding_energy_erg, dtype=np.float64)
        if (
            np.any(~np.isfinite(temperature_array))
            or np.any(temperature_array <= 0.0)
            or binding.shape != self.ionization_scale.shape
            or np.any(~np.isfinite(binding))
            or np.any(binding <= 0.0)
        ):
            raise ValueError("invalid temperature or He I binding energy")
        reduced = binding / (
            BOLTZMANN * temperature_array[..., np.newaxis]
        )
        shifted = reduced + 0.27
        auxiliary = (reduced + 3.43) / (reduced + 1.43) ** 3
        exponential_integral = _tlusty_exponential_integral_e1(reduced)
        shifted_integral = _tlusty_exponential_integral_e1(shifted)
        bracket = exponential_integral - reduced * (
            0.728 * shifted_integral / shifted
            + 0.189 * np.exp(-np.minimum(reduced, 745.0)) * auxiliary
        )
        return np.maximum(
            5.465e-11
            * np.sqrt(temperature_array)[..., np.newaxis]
            * self.ionization_scale
            * reduced
            * bracket,
            0.0,
        )


def _tlusty_exponential_integral_e1(argument: FloatArray) -> FloatArray:
    """Evaluate the rational E1 approximation used by TLUSTY 200."""

    value = np.asarray(argument, dtype=np.float64)
    low = (
        -np.log(value)
        - 0.57721566
        + value
        * (
            0.99999193
            + value
            * (
                -0.24991055
                + value
                * (0.05519968 + value * (-0.00976004 + value * 0.00107857))
            )
        )
    )
    numerator = 0.2677734343 + value * (
        8.6347608925
        + value * (18.059016973 + value * (8.5733287401 + value))
    )
    denominator = 3.9584969228 + value * (
        21.0996530827
        + value * (25.6329561486 + value * (9.5733223454 + value))
    )
    high = (
        np.exp(-np.minimum(value, 745.0))
        * numerator
        / denominator
        / value
    )
    return np.where(value <= 1.0, low, high)


def read_tlusty_helium_collision_data(
    source_path: str | Path,
    atom_path: str | Path | None = None,
) -> TlustyHeliumCollisionData:
    """Read COLLHE fit coefficients from an official TLUSTY Fortran source."""

    path = Path(source_path)
    text = path.read_text(errors="replace")
    match = re.search(
        r"^\s*SUBROUTINE\s+COLLHE\b(?P<body>.*?)^\s*END\s*$",
        text,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if match is None:
        raise ValueError("source does not contain TLUSTY's COLLHE subroutine")
    body = match.group("body")
    start_match = re.search(
        r"DATA\s+NSTART\s*/(?P<values>.*?)/",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if start_match is None:
        raise ValueError("COLLHE NSTART table was not found")
    start = _fortran_numbers(start_match.group("values")).astype(np.int64)
    coefficient_blocks = re.findall(
        r"DATA\s*\(A\(I\),I=\d+,\d+\)\s*/(?P<values>.*?)/",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    coefficient = np.concatenate(
        [_fortran_numbers(block) for block in coefficient_blocks]
    )
    if start.shape != (172,) or coefficient.shape != (929,):
        raise ValueError(
            "unexpected COLLHE table dimensions: "
            f"NSTART={start.size}, A={coefficient.size}"
        )
    if start[0] != 1 or start[-1] != coefficient.size + 1:
        raise ValueError("COLLHE coefficient offsets are inconsistent")
    ionization_scale = None
    if atom_path is not None:
        atom_text = Path(atom_path).read_text(errors="replace")
        section_match = re.search(
            r"\*+\s*Continuum transitions(?P<body>.*?)\*+\s*Line transitions",
            atom_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if section_match is None:
            raise ValueError("TLUSTY He I continuum-transition section was not found")
        values: list[float] = []
        for line in section_match.group("body").splitlines():
            numbers = _fortran_numbers(line)
            if numbers.size >= 9:
                values.append(float(numbers[7]))
        ionization_scale = np.asarray(values, dtype=np.float64)
        if ionization_scale.ndim != 1 or ionization_scale.size < 1 or np.any(ionization_scale <= 0.0):
            raise ValueError("unexpected TLUSTY He I ionization scale table")
    return TlustyHeliumCollisionData(
        fit_start_index=np.ascontiguousarray(start),
        coefficient=np.ascontiguousarray(coefficient),
        source_path=str(path.resolve()),
        ionization_scale=(
            None
            if ionization_scale is None
            else np.ascontiguousarray(ionization_scale)
        ),
    )
