"""Scoped recovery of the pre-cool-DZ-release DZ physics for immutable controls.

This is a research/test helper, never a production configuration option.  Inside
previous_cool_dz_release_physics() and with previous_cool_dz_release_config,
a DZ spectrum reproduces the controls approved before the cool-DZ release
(independent_kurucz_2026_10_14) exactly, so the older approval chain stays provable:

- the seven DZConfig defaults the release changed;
- the HM88 neutral hard sphere on every bound level (now Rydberg levels only);
- the TLUSTY RAP reader that added the ground-term centroid to RAP's absolute
  excitations and cut the absolute photon table at the mean I-E binding;
- the Ca II CRD source quadrature centred on the sampled extinction maxima and
  including the caller's wavelength points.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import replace
import sys
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

import wd_spectra.cool_metal_nlte as cool_metal_nlte
import wd_spectra.d6 as d6
import wd_spectra.metals as metals
from wd_spectra.constants import LIGHT_SPEED
from wd_spectra.cool_metal_nlte import _planck_frequency

FloatArray = NDArray[np.float64]

PREVIOUS_DZ_DEFAULTS = dict(
    dense_helium_eos="ideal", dense_helium_free_free="off", helium_minus_free_free="automatic",
    mg_i_b_he_unified_profile=False, mg_ii_he_unified_profile=False,
    ca_ii_he_unified_profile=False, na_i_he_unified_profile=False,
)


def previous_cool_dz_release_config(config):
    """The same configuration with the defaults that preceded the release."""
    return replace(config, **PREVIOUS_DZ_DEFAULTS)


def _previous_neutral_hard_sphere(level_radius_cm, neutral_perturber_number_density, neutral_perturber_radius_cm):
    level_radius = np.asarray(level_radius_cm, dtype=np.float64)
    densities = {k: np.asarray(v, dtype=np.float64) for k, v in neutral_perturber_number_density.items()}
    if not densities:
        return np.ones_like(level_radius)
    broadcast = np.broadcast_arrays(level_radius, *densities.values())
    exponent = np.zeros_like(broadcast[0])
    for (element, _), density in zip(densities.items(), broadcast[1:]):
        exponent += density * (broadcast[0] + float(neutral_perturber_radius_cm[element])) ** 3
    return np.exp(-np.minimum((4.0 * np.pi / 3.0) * exponent, 745.0))


def _previous_rap_reader(reader):
    def read(entries, atomic_database, *args, **kwargs):
        database = reader(entries, atomic_database, *args, **kwargs)
        sections = []
        for section in database.sections:
            if not section.absolute_photon_grid:
                sections.append(section); continue
            ion = atomic_database.ions[(section.element, section.charge)]
            excitation = d6.stout_ground_term_centroid_ev(ion) + section.excitation_energy_ev
            edge = float(ion.ionization_energy_ev - excitation)
            if excitation < 0.0 or edge <= 0.0:
                continue
            sections.append(replace(section, excitation_energy_ev=excitation, threshold_energy_ev=edge,
                                    source_threshold_energy_ev=edge, absolute_photon_grid=False))
        return replace(database, sections=tuple(sections))
    return read


@contextmanager
def previous_cool_dz_release_physics():
    """Patch every wd_spectra binding of the three changed functions."""
    swaps = {
        metals.metal_neutral_hard_sphere_occupation_probability: _previous_neutral_hard_sphere,
        d6.read_tlusty_rap_lte_photoionization: _previous_rap_reader(d6.read_tlusty_rap_lte_photoionization),
        cool_metal_nlte.ca_ii_crd_resonance_source_functions: _previous_ca_ii_crd_resonance_source_functions,
    }
    with ExitStack() as stack:
        for module in tuple(sys.modules.values()):
            if module is None or not getattr(module, "__name__", "").startswith("wd_spectra"):
                continue
            for name, value in tuple(vars(module).items()):
                for current, previous in swaps.items():
                    if value is current:
                        stack.enter_context(patch.object(module, name, previous))
        yield


def _previous_ca_ii_crd_resonance_source_functions(
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

    from wd_spectra._linear_scattering import linear_lambda_operator
    from wd_spectra.opacity import optical_depth_from_mass_opacity

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
