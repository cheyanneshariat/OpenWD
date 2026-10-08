"""Helium material adapter used by the PG 1159 He/C/O model."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray
from .atmosphere import Atmosphere
from .eos import hummer_mihalas_helium_lte
from .helium_nlte import (
    CoupledHeliumNLTEState,
    HydrogenicHeliumCollisionModel,
    coupled_helium_nlte_transfer_coefficients,
    remap_coupled_helium_state,
    solve_coupled_helium_nlte,
)
from .helium_collisions import TlustyHeliumCollisionData
from .multilevel_nlte import HydrogenElectronCollisionData
from .nlte_core import NLTETransferCoefficients, OpacitySpan, _FixedTransferCache

FloatArray = NDArray[np.float64]


_LINE_KEYS = ("he-i-line", "he-i-resonance", "he-ii-line")
# Upper bound on retained line-profile storage for one population solve.
_LINE_SPAN_CACHE_BYTES = 512 * 1024 * 1024


class PG1159HeliumTransferCache(_FixedTransferCache):
    """Cache fixed continuum kernels and compact He line opacities.

    A full-grid LTE line opacity is almost entirely zero away from the line,
    and dense copies of all He I/II lines would need gigabytes.  Keep only the
    contiguous wavelength rows that contain nonzero values, returned as an
    ``OpacitySpan``; its consumers add exactly the same terms to those rows
    (all other rows of ``build()`` are zero).  Once the byte budget is spent,
    remaining lines are recomputed each call.
    """

    def __init__(self, atmosphere, wavelength):
        super().__init__(atmosphere, wavelength)
        self.line_spans = {}
        self.line_span_bytes = 0

    def get(self, key, build):
        if key[0] in _LINE_KEYS:
            span = self.line_spans.get(key)
            if span is not None:
                return span
            value = build()
            if not (
                isinstance(value, np.ndarray)
                and value.dtype == np.float64
                and value.ndim == 2
            ):
                return value
            # Keep signed zeros too, so the retained rows are bit-exact.
            rows = np.flatnonzero(
                np.any((value != 0.0) | np.signbit(value), axis=1)
            )
            start = int(rows[0]) if rows.size else 0
            stop = int(rows[-1]) + 1 if rows.size else 0
            span = OpacitySpan(start, value[start:stop].copy(), value.shape[0])
            if self.line_span_bytes + span.block.nbytes <= _LINE_SPAN_CACHE_BYTES:
                self.line_spans[key] = span
                self.line_span_bytes += span.block.nbytes
            return span
        if key[0] not in ("he-i-continuum", "he-ii-continuum"):
            return build()
        return super().get(key, build)


@dataclass(frozen=True)
class CoupledHeliumNLTEModel:
    """Coupled He I/II/III adapter for the common atmosphere driver.

    The 14-term neutral atom, hydrogenic He II shells, and He III share one
    statistical-equilibrium matrix and one particle-conservation equation.
    The temperature-dependent HM EOS is rebuilt between outer iterations;
    NLTE charge feedback into that EOS is deliberately still disabled.
    """

    collision_data: HydrogenElectronCollisionData
    maximum_helium_ii_level: int = 8
    helium_i_stark_table: object | None = None
    helium_ii_stark_table: object | None = None
    include_helium_i_lines: bool = True
    include_helium_i_resonance_lines: bool = True
    include_helium_ii_lines: bool = True
    correlated_microfields: bool = True
    population_explicit_maximum_lower_level: int = 3
    population_n_angle: int = 3
    population_maximum_iterations: int = 120
    population_relative_tolerance: float = 5.0e-3
    population_relative_floor: float = 1.0e-12
    population_damping: float = 0.4
    population_acceleration: str = "anderson"
    neutral_collision_strength_scale: float = 1.0
    helium_i_collision_data: TlustyHeliumCollisionData | None = None
    hydrogenic_collision_model: HydrogenicHeliumCollisionModel = "ccc-scaled"
    hydrogenic_collision_rate_multiplier: float = 1.0
    hydrogenic_excitation_collision_rate_multiplier: float | None = None
    hydrogenic_ionization_collision_rate_multiplier: float | None = None

    @property
    def name(self) -> str:
        return "coupled-helium"

    def rebuild_atmosphere(
        self, template: Atmosphere, temperature: FloatArray, previous_state: Any | None
    ) -> Atmosphere:
        helium = template.helium_lte_state
        if helium is None:
            raise ValueError("CoupledHeliumNLTEModel requires a pure-helium atmosphere")
        eos = hummer_mihalas_helium_lte(
            temperature,
            template.gas_pressure,
            neutral_radius_scale=helium.neutral_radius_scale,
            correlated_microfields=self.correlated_microfields,
        )
        metadata = dict(template.metadata)
        metadata.update(
            {
                "composition": "pure-helium",
                "nlte_ion_stage_scope": "coupled He I/II/III",
                "nlte_charge_feedback": False,
            }
        )
        return Atmosphere(
            effective_temperature=template.effective_temperature,
            logg=template.logg,
            rosseland_optical_depth=template.rosseland_optical_depth,
            column_mass=template.column_mass,
            temperature=np.ascontiguousarray(temperature),
            gas_pressure=template.gas_pressure,
            mass_density=eos.mass_density,
            neutral_h_density=np.zeros(template.n_depth),
            proton_density=np.zeros(template.n_depth),
            electron_density=eos.electron_density,
            metadata=metadata,
            hydrogen_lte_state=None,
            helium_lte_state=eos,
        )

    def solve_populations(
        self, atmosphere: Atmosphere, previous_state: Any | None
    ) -> CoupledHeliumNLTEState:
        if self.helium_i_stark_table is None:
            raise ValueError("a He I Stark table is required for coupled line transfer")
        previous = (
            remap_coupled_helium_state(atmosphere, previous_state)
            if isinstance(previous_state, CoupledHeliumNLTEState)
            else None
        )
        return solve_coupled_helium_nlte(
            atmosphere,
            self.collision_data,
            helium_i_stark_table=self.helium_i_stark_table,
            helium_ii_stark_table=self.helium_ii_stark_table,
            maximum_helium_ii_level=self.maximum_helium_ii_level,
            helium_ii_explicit_maximum_lower_level=(
                self.population_explicit_maximum_lower_level
            ),
            n_angle=self.population_n_angle,
            maximum_iterations=self.population_maximum_iterations,
            relative_tolerance=self.population_relative_tolerance,
            relative_population_floor=self.population_relative_floor,
            population_damping=self.population_damping,
            acceleration=self.population_acceleration,
            neutral_collision_strength_scale=(self.neutral_collision_strength_scale),
            helium_i_collision_data=self.helium_i_collision_data,
            hydrogenic_collision_model=self.hydrogenic_collision_model,
            hydrogenic_collision_rate_multiplier=(
                self.hydrogenic_collision_rate_multiplier
            ),
            hydrogenic_excitation_collision_rate_multiplier=(
                self.hydrogenic_excitation_collision_rate_multiplier
            ),
            hydrogenic_ionization_collision_rate_multiplier=(
                self.hydrogenic_ionization_collision_rate_multiplier
            ),
            initial_state=previous,
        )

    def transfer_coefficients(
        self,
        atmosphere: Atmosphere,
        wavelength_angstrom: FloatArray,
        state: Any,
        *,
        _cache=None,
    ) -> NLTETransferCoefficients:
        if not isinstance(state, CoupledHeliumNLTEState):
            raise TypeError("CoupledHeliumNLTEModel requires a coupled helium state")
        if self.helium_i_stark_table is None:
            raise ValueError("a He I Stark table is required")
        return coupled_helium_nlte_transfer_coefficients(
            atmosphere,
            wavelength_angstrom,
            state,
            _cache=_cache,
            # He is one contribution to the He/C/O extinction. The full
            # mixture and radiation field are checked by the shared solver.
            _allow_signed_continuum=True,
            helium_i_stark_table=self.helium_i_stark_table,
            helium_ii_stark_table=self.helium_ii_stark_table,
            include_helium_i_lines=self.include_helium_i_lines,
            include_helium_i_resonance_lines=(self.include_helium_i_resonance_lines),
            include_helium_ii_lines=self.include_helium_ii_lines,
        )

    def perturbed_transfer_coefficients(
        self,
        atmosphere: Atmosphere,
        state: Any,
        wavelength_angstrom: FloatArray,
        fractional_temperature_step: float,
    ) -> NLTETransferCoefficients:
        if not isinstance(state, CoupledHeliumNLTEState):
            raise TypeError("CoupledHeliumNLTEModel requires a coupled helium state")
        hotter = self.rebuild_atmosphere(
            atmosphere,
            (1.0 + fractional_temperature_step) * atmosphere.temperature,
            state,
        )
        mapped_state = remap_coupled_helium_state(hotter, state)
        return self.transfer_coefficients(hotter, wavelength_angstrom, mapped_state)
