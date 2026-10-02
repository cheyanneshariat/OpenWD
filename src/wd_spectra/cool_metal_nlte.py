"""Approximate non-LTE source functions for cool-star resonance lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from .atmosphere import Atmosphere
from .chianti import read_chianti_scaled_collision_components
from .constants import BOLTZMANN, LIGHT_SPEED, PLANCK
from .metals import AtomicDatabase


FloatArray = NDArray[np.float64]
_EFFECTIVE_COLLISION_RATE_CONSTANT = 8.629e-6


@dataclass(frozen=True)
class ResonanceScatteringProbability:
    """Depth-dependent fate of an H/K photon absorbed in one line.

    ``probability`` is resonant re-emission in the same line.
    ``fine_structure_transfer`` is collisional transfer to the other 4p level,
    which re-emits in the partner line rather than destroying the photon; the
    remainder, ``destruction_probability``, thermalizes it.
    """

    probability: FloatArray
    resonant_einstein_a: float
    total_radiative_rate: float
    source: str
    fine_structure_partner: tuple[int, int] | None = None
    fine_structure_transfer: FloatArray | None = None

    @property
    def destruction_probability(self) -> FloatArray:
        transfer = (
            0.0 if self.fine_structure_transfer is None
            else self.fine_structure_transfer
        )
        return np.clip(1.0 - self.probability - transfer, 0.0, 1.0)


def ca_ii_resonance_scattering_probabilities(
    atmosphere: Atmosphere,
    atomic_database: AtomicDatabase,
    collision_strength_path: str | Path,
) -> Mapping[tuple[int, int], ResonanceScatteringProbability]:
    r"""Return physical H/K scattering probabilities at every depth.

    The upper ``4p ^2P^o`` level can resonantly decay to the ground state,
    radiatively branch into the infrared triplet, or be collisionally
    transferred to another Ca II level.  In this reduced-atom treatment the
    first channel is coherent line scattering and the latter channels destroy
    an H/K photon.  Electron rates use the R-matrix effective collision
    strengths supplied in the CHIANTI Ca II ``.scups`` file,

    ``q_ij = 8.629e-6 Upsilon/(g_i sqrt(T)) exp(-Delta E/kT)``

    for upward transitions, with the exponential omitted for downward ones.
    Ca II populations and extinction remain LTE, so this is a fixed-structure
    source-function treatment rather than a full multilevel NLTE solve.
    """

    ion = atomic_database.ions.get(("Ca", 1))
    if ion is None:
        raise ValueError("atomic database does not contain Ca II")
    levels = {level.index: level for level in ion.levels}
    collisions = read_chianti_scaled_collision_components(
        collision_strength_path
    )
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    electron_density = np.asarray(atmosphere.electron_density, dtype=np.float64)
    if np.any(temperature <= 0.0) or np.any(electron_density < 0.0):
        raise ValueError("atmosphere temperatures and electron densities are invalid")

    radiative_rate_by_upper: dict[int, float] = {}
    for transition in ion.transitions:
        radiative_rate_by_upper[transition.upper_index] = (
            radiative_rate_by_upper.get(transition.upper_index, 0.0)
            + transition.einstein_a
        )

    resonance = tuple(
        transition
        for transition in ion.transitions
        if transition.lower_index == 1
        and 3920.0 < transition.wavelength_vacuum_angstrom < 3990.0
    )
    if len(resonance) != 2:
        raise ValueError("Ca II atomic data must contain both H and K transitions")

    result: dict[tuple[int, int], ResonanceScatteringProbability] = {}
    for transition in resonance:
        upper = levels[transition.upper_index]
        partner = next(
            other for other in resonance if other.upper_index != transition.upper_index
        )
        collisional_exit_coefficient = np.zeros_like(temperature)
        fine_structure_coefficient = np.zeros_like(temperature)
        for (lower_index, upper_index), fit in collisions.items():
            if transition.upper_index not in (lower_index, upper_index):
                continue
            other_index = (
                upper_index
                if lower_index == transition.upper_index
                else lower_index
            )
            other = levels.get(other_index)
            if other is None:
                continue
            upsilon = np.asarray(
                [fit.effective_collision_strength(value) for value in temperature],
                dtype=np.float64,
            )
            coefficient = (
                _EFFECTIVE_COLLISION_RATE_CONSTANT
                * upsilon
                / (upper.statistical_weight * np.sqrt(temperature))
            )
            energy_difference = (
                other.energy_wavenumber - upper.energy_wavenumber
            ) * PLANCK * LIGHT_SPEED
            if energy_difference > 0.0:
                coefficient *= np.exp(
                    -energy_difference / (BOLTZMANN * temperature)
                )
            collisional_exit_coefficient += coefficient
            if other_index == partner.upper_index:
                fine_structure_coefficient += coefficient
        collisional_rate = electron_density * collisional_exit_coefficient
        total_radiative_rate = radiative_rate_by_upper[transition.upper_index]
        total_rate = total_radiative_rate + collisional_rate
        probability = transition.einstein_a / total_rate
        result[(transition.lower_index, transition.upper_index)] = (
            ResonanceScatteringProbability(
                probability=np.clip(probability, 0.0, 1.0),
                resonant_einstein_a=transition.einstein_a,
                total_radiative_rate=total_radiative_rate,
                source=(
                    "CHIANTI Ca II electron collision strengths; radiative "
                    "branching from the supplied atomic database"
                ),
                fine_structure_partner=(partner.lower_index, partner.upper_index),
                fine_structure_transfer=np.clip(
                    electron_density * fine_structure_coefficient / total_rate,
                    0.0,
                    1.0,
                ),
            )
        )
    return MappingProxyType(result)


def _planck_frequency(wavelength_angstrom: FloatArray, temperature: FloatArray) -> FloatArray:
    wavelength_cm = np.asarray(wavelength_angstrom, dtype=np.float64)[:, np.newaxis] * 1.0e-8
    frequency = LIGHT_SPEED / wavelength_cm
    return (
        2.0 * PLANCK * frequency**3 / LIGHT_SPEED**2
        / np.expm1(PLANCK * frequency / (BOLTZMANN * temperature[np.newaxis, :]))
    )


def ca_ii_crd_resonance_source_functions(
    atmosphere: Atmosphere,
    wavelength_angstrom: FloatArray,
    background_absorption: FloatArray,
    background_scattering: FloatArray,
    line_extinction: Callable[[FloatArray], Mapping[tuple[int, int], FloatArray]],
    probabilities: Mapping[tuple[int, int], ResonanceScatteringProbability],
    *,
    n_angle: int,
    half_width_angstrom: float = 800.0,
    points_per_side: int = 160,
) -> dict[tuple[int, int], FloatArray]:
    r"""Solve the Ca II H and K line source functions in complete redistribution.

    H/K damping in helium is dominated by elastic He collisions, which
    redistribute the photon frequency over the whole profile.  Each line
    therefore has one frequency-independent source function,

    ``S_j = p_j Jbar_j + eps_j B_j + c_j (B_j/B_i) S_i``,

    with ``Jbar_j = int phi_j J_nu dnu``.  ``p_j`` is resonant re-emission,
    ``eps_j`` thermal destruction (infrared branching and electron exit), and
    ``c_j`` collisional transfer to the partner 4p level, which re-emits in
    the other line.  This follows from the statistical equilibrium of the
    two 4p levels with detailed balance and the Einstein relations, neglecting
    stimulated emission as the reduced atom already does.  Background
    electron and Rayleigh scattering stay coherent.

    The equations are solved exactly as one linear system in ``S_H, S_K``
    using the exact linear-formal Lambda operator of every frequency.  The
    frequency integral uses a dedicated grid that resolves each Doppler core
    and extends ``half_width_angstrom`` into the damping wings; the line
    extinction is evaluated exactly there, while the smooth background is
    interpolated from the caller's opacity (held at the nearest edge outside
    its range).  Returned source functions are per unit frequency, one
    ``(n_depth,)`` array per line key.
    """

    from ._linear_scattering import linear_lambda_operator
    from .opacity import optical_depth_from_mass_opacity

    keys = tuple(probabilities)
    if len(keys) != 2:
        raise ValueError("complete redistribution requires both Ca II H and K")
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    n_depth = temperature.size

    user_lines = line_extinction(wavelength)
    centers = {
        key: float(wavelength[int(np.argmax(np.max(user_lines[key], axis=1)))])
        for key in keys
    }
    # Grid: symmetric log-spaced offsets about each centre plus the caller's
    # own points inside the window.
    offsets = np.geomspace(1.0e-3, half_width_angstrom, points_per_side)
    nodes = [wavelength[
        (wavelength > min(centers.values()) - half_width_angstrom)
        & (wavelength < max(centers.values()) + half_width_angstrom)
    ]]
    for center in centers.values():
        nodes.extend((center - offsets, center + offsets, np.asarray([center])))
    grid = np.unique(np.concatenate(nodes))
    grid = grid[grid > 0.0]

    lines = line_extinction(grid)
    line_sum_user = sum(user_lines[key] for key in keys)
    other_absorption = np.maximum(
        background_absorption - line_sum_user, np.finfo(np.float64).tiny
    )

    def interpolate(values: FloatArray) -> FloatArray:
        values = np.broadcast_to(values, (wavelength.size, n_depth))
        logs = np.log(np.maximum(values, np.finfo(np.float64).tiny))
        return np.exp(np.stack(
            [np.interp(grid, wavelength, logs[:, depth]) for depth in range(n_depth)],
            axis=1,
        ))

    absorption = interpolate(other_absorption)
    scattering = interpolate(background_scattering) if np.any(background_scattering > 0.0) else np.zeros_like(absorption)
    line_total = sum(lines[key] for key in keys)
    total = absorption + scattering + line_total
    tau = optical_depth_from_mass_opacity(atmosphere.column_mass, total)
    planck = _planck_frequency(grid, temperature)

    frequency = LIGHT_SPEED / (grid * 1.0e-8)
    order = np.argsort(frequency)
    weight = np.zeros_like(frequency)
    sorted_frequency = frequency[order]
    step = np.diff(sorted_frequency)
    weight[order[:-1]] += 0.5 * step
    weight[order[1:]] += 0.5 * step
    profile = {}
    for key in keys:
        normalization = np.sum(lines[key] * weight[:, np.newaxis], axis=0)
        profile[key] = lines[key] * weight[:, np.newaxis] / np.maximum(
            normalization, np.finfo(np.float64).tiny
        )[np.newaxis, :]

    identity = np.eye(n_depth)
    # Jbar_i = gbar_i + sum_j Gbar_ij S_j
    gbar = {key: np.zeros(n_depth) for key in keys}
    Gbar = {(i, j): np.zeros((n_depth, n_depth)) for i in keys for j in keys}
    chunk = 64
    for begin in range(0, grid.size, chunk):
        index = np.arange(begin, min(begin + chunk, grid.size))
        lam = linear_lambda_operator(tau[index], n_angle)
        for local, k in enumerate(index):
            coherent = scattering[k] / total[k]
            response = lam[local] @ np.linalg.inv(identity - coherent[:, np.newaxis] * lam[local])
            thermal = response @ (absorption[k] * planck[k] / total[k])
            for i in keys:
                weight_i = profile[i][k]
                if not np.any(weight_i > 0.0):
                    continue
                gbar[i] += weight_i * thermal
                for j in keys:
                    Gbar[(i, j)] += weight_i[:, np.newaxis] * (
                        response * (lines[j][k] / total[k])[np.newaxis, :]
                    )

    center_planck = {
        key: _planck_frequency(np.asarray([centers[key]]), temperature)[0]
        for key in keys
    }
    matrix = np.zeros((2 * n_depth, 2 * n_depth))
    rhs = np.zeros(2 * n_depth)
    for row, i in enumerate(keys):
        record = probabilities[i]
        p = record.probability
        c = (
            np.zeros(n_depth) if record.fine_structure_transfer is None
            else record.fine_structure_transfer
        )
        epsilon = record.destruction_probability
        partner = record.fine_structure_partner
        block = slice(row * n_depth, (row + 1) * n_depth)
        matrix[block, block] += identity
        for column, j in enumerate(keys):
            other = slice(column * n_depth, (column + 1) * n_depth)
            matrix[block, other] -= p[:, np.newaxis] * Gbar[(i, j)]
            if partner == j:
                matrix[block, other] -= np.diag(
                    c * center_planck[i] / center_planck[j]
                )
        rhs[block] = p * gbar[i] + epsilon * center_planck[i]
    solution = np.linalg.solve(matrix, rhs)
    return {
        key: solution[row * n_depth:(row + 1) * n_depth]
        for row, key in enumerate(keys)
    }
