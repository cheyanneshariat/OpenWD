"""Species-independent interfaces for non-LTE atmosphere calculations.

The rate equations remain species specific, but the atmosphere iteration only
needs their total true absorption, emissivity, and scattering coefficients.
Keeping that boundary small lets the same radiative-equilibrium driver host a
hydrogen atom now and He/metal model atoms later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .atmosphere import Atmosphere


FloatArray = NDArray[np.float64]


class NonphysicalPopulationError(RuntimeError):
    """An SE solution or conservation closure has inadmissible populations.

    Low-level atomic callers still fail. A nonlinear atmosphere adapter may
    translate this specific failure into rejection of a proposed trial; input,
    data and unrelated numerical/programming errors are not included.
    """


@dataclass(frozen=True)
class NLTETransferCoefficients:
    r"""Monochromatic coefficients needed by an NLTE formal solution.

    ``true_absorption`` and ``scattering`` are mass coefficients in
    cm2 g-1.  ``thermal_emissivity`` is the emissivity divided by mass density
    in the same wavelength-intensity convention as ``J_lambda`` times a mass
    coefficient.  Arrays have shape ``(wavelength, depth)``.
    """

    wavelength_angstrom: FloatArray
    true_absorption: FloatArray
    thermal_emissivity: FloatArray
    scattering: FloatArray
    metadata: dict[str, object]

    @property
    def total_extinction(self) -> FloatArray:
        """Return true absorption plus scattering."""

        return self.true_absorption + self.scattering


class NLTEAtmosphereModel(Protocol):
    """Boundary between the generic atmosphere loop and model atoms."""

    name: str

    def rebuild_atmosphere(
        self,
        template: Atmosphere,
        temperature: FloatArray,
        previous_state: Any | None,
    ) -> Atmosphere:
        """Update the EOS/charge state on the fixed column-mass grid."""

    def solve_populations(
        self, atmosphere: Atmosphere, previous_state: Any | None
    ) -> Any:
        """Solve statistical equilibrium on the current structure."""

    def transfer_coefficients(
        self,
        atmosphere: Atmosphere,
        wavelength_angstrom: FloatArray,
        state: Any,
    ) -> NLTETransferCoefficients:
        """Assemble the model atom's opacity and emissivity."""



class OpacitySpan:
    """Rows ``start:start+len(block)`` of an otherwise zero (wavelength, depth) array."""

    __slots__ = ("start", "block", "n_wavelength")

    def __init__(self, start, block, n_wavelength):
        self.start = int(start)
        self.block = block
        self.n_wavelength = int(n_wavelength)

    @property
    def rows(self):
        return slice(self.start, self.start + self.block.shape[0])


class _FixedTransferCache:
    """Private single-atmosphere, single-grid cache for population iteration."""
    def __init__(self, atmosphere, wavelength):
        self.atmosphere = atmosphere
        self.wavelength = np.array(wavelength, copy=True)
        self.values = {}

    def validate(self, atmosphere, wavelength):
        if atmosphere is not self.atmosphere or not np.array_equal(wavelength, self.wavelength):
            raise ValueError("atomic transfer cache belongs to a different atmosphere/grid")

    def get(self, key, build):
        if key not in self.values:
            self.values[key] = build()
        return self.values[key]


def _cached_transfer(cache, key, build):
    return build() if cache is None else cache.get(key, build)
