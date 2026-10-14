"""Photospheric H2 Lyman (B-X) and Werner (C-X) line opacity with fluorescent source functions.

Cool hydrogen white dwarfs (Teff below about 14,000 K) show H2 Lyman- and Werner-band lines from vibrationally
excited ground-state levels in the far ultraviolet (Xu et al. 2013, ApJ 766, L18).  This module supplies their
opacity and line source functions for the formal solution of a hydrogen atmosphere.

Line data are Robert L. Kurucz's ``h2bx.asc`` and ``h2cx.asc`` lists.  Only transitions whose two levels have
measured (not calculated) energies contribute opacity: Kurucz flags calculated levels with negative energies, and
their wavelengths are not reliable at COS resolution.  Every listed transition (measured or not) enters the
fluorescent branch sums of its upper level, because the level decays through all of them.  Kurucz's gf includes
the nuclear-spin weight divided by four, ``gf = g_ns (2J''+1) f / 4``, with g_ns set by the lower level (even J''
para, odd J'' ortho).  Summed over each upper level, the Kurucz rates reproduce the bound emission rates of
Abgrall, Roueff & Drira (2000) to about 1 per cent away from perturbed levels.  For every level they tabulate,
the bound and dissociative decay rates are theirs, with the Kurucz branches rescaled to that bound total.

Lower (X-state) levels are in LTE: collisional dissociation and three-body formation by H atoms (Glover & Abel
2008) are far faster than photodissociation wherever the lines form.  Level populations follow from the EOS H2
density and the same Barklem & Collet partition function the EOS uses, so ``n_l f = n(H2) 4 gf exp(-E''/kT) / Q``
with ``Q`` = 4 x the nuclear-spin-normalized table.

The upper B and C levels decay radiatively in about 1 ns, much faster than collisions in the line-forming layers,
so the lines do not have LTE source functions.  Statistical equilibrium of each upper level ``u`` with LTE lower
levels gives, for every line ``l -> u``,

    S_lu / B_lu(T) = n_u / n_u* = [sum_branches A_ul' (J/B)_l' + A_c + C_u] / [A_u + A_c + C_u],

where the sum runs over all bound decay branches of ``u`` (Kurucz A values), ``A_c`` is the spontaneous
radiative dissociation rate into the X continuum (Abgrall, Roueff & Drira 2000), ``C_u`` the van Regemorter (1962)
electron de-excitation rate summed over branches, and ``(J/B)`` the ratio of mean intensity to the local Planck
function at each branch wavelength.  The branches are treated as optically thin (``J/B`` from the continuum and
hydrogen lines alone), an upper bound on the infilling of strong lines.  No collisional quenching by H atoms is
included: no rates are available.
"""

from __future__ import annotations

import hashlib
import lzma
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np
from scipy.special import exp1, wofz

from .molecules import molecular_hydrogen_internal_partition_function

H2_LINE_RESOURCE = "data/molecular/h2_lyman_werner_kurucz.asc.xz"
H2_LINE_SHA256 = "7c94475dd574c006005607b795f4d78703c160ac3c456a0717846ab2b4c515de"
H2_ABGRALL_RESOURCES = {
    "B": ("data/molecular/h2_abgrall2000_table1.dat", "f4afc90fcfc408a15f9ba1a1a4e6d05dae359cb0021f2cd261b0c00144b98fff"),
    "C+": ("data/molecular/h2_abgrall2000_table2.dat", "04054067c6d7748d0347dd2f0e5772bf367ac4ef141286c4dc8629e82829a575"),
    "C-": ("data/molecular/h2_abgrall2000_table3.dat", "c4d876db564a0ba7718bc2b1c2dcf07ff681e4d43e494a86aedcf047074d1201"),
}
# Coarse grid on which the mean-intensity ratio J/B is solved for the branch wavelengths.
H2_RADIATION_GRID_ANGSTROM = np.arange(845.0, 1850.0, 1.0)

_LIGHT_SPEED = 2.99792458e10
_BOLTZMANN = 1.380649e-16
_SECOND_RADIATION_CONSTANT = 1.4387769  # hc/k in cm K
_INTEGRATED_CROSS_SECTION = 0.026540081  # pi e^2 / (m_e c), cm^2 s^-1
_EINSTEIN_A_FACTOR = 0.6670251  # 8 pi^2 e^2 / (m_e c), cm^2 s^-1 (A = this x (1/lambda)^2 x g_l f / g_u)
_ATOMIC_MASS_UNIT = 1.66053907e-24
_ELECTRON_MASS = 9.1093837e-28
_H2_MASS = 2.01588 * _ATOMIC_MASS_UNIT


@dataclass(frozen=True)
class H2LymanWernerData:
    """H2 Lyman and Werner lines grouped by upper level.

    The ``wavelength_angstrom``, ``gf``, ``lower_energy_wavenumber`` and
    ``upper_index`` arrays hold the measured-level lines that contribute
    opacity.  The ``branch_*`` arrays hold every listed line, used for the
    radiative decay and fluorescence of the upper levels.
    """

    wavelength_angstrom: np.ndarray       # vacuum, measured-level lines
    gf: np.ndarray                        # Kurucz convention, g_ns (2J''+1) f / 4
    lower_energy_wavenumber: np.ndarray
    upper_index: np.ndarray               # index into the upper-level arrays
    branch_wavelength_angstrom: np.ndarray  # every listed line
    branch_upper_index: np.ndarray
    branch_einstein_a: np.ndarray           # A_ul, s^-1
    upper_bound_decay_rate: np.ndarray    # bound A per upper level (Abgrall et al. 2000 where tabulated)
    upper_dissociation_rate: np.ndarray   # A_c per upper level (Abgrall et al. 2000)
    source: str

    @property
    def n_lines(self) -> int:
        return int(self.wavelength_angstrom.size)


def _verified(resource: str, sha256: str) -> bytes:
    content = files("wd_spectra").joinpath(resource).read_bytes()
    if resource.endswith(".xz"):
        content = lzma.decompress(content)
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"checksum mismatch: {resource}")
    return content


def _kurucz_einstein_a(wavelength_angstrom, gf, lower_j, upper_j) -> np.ndarray:
    """Einstein A from Kurucz's gf = g_ns (2J''+1) f / 4.

    Ortho/para is conserved, so the nuclear-spin weight follows the lower X level: even J'' para
    (g_ns = 1), odd J'' ortho (g_ns = 3). J' parity gives the same answer only for P and R lines.
    """
    nuclear_spin = np.where(np.round(np.asarray(lower_j)) % 2 == 0, 1.0, 3.0)
    return (
        _EINSTEIN_A_FACTOR * (1.0e8 / np.asarray(wavelength_angstrom)) ** 2 * 4.0 * np.asarray(gf)
        / (nuclear_spin * (2.0 * np.asarray(upper_j) + 1.0))
    )


def _abgrall_rates() -> dict[tuple[str, int, int], tuple[float, float]]:
    """Abgrall, Roueff & Drira (2000) per (state, v, J): total decay rate A_t and its dissociative part A_c.

    A_t includes radiative dissociation (A_c <= A_t, and A_c -> A_t near the dissociation limit), so
    the bound emission rate into X is A_t - A_c.
    """
    rates: dict[tuple[str, int, int], tuple[float, float]] = {}
    for state, (resource, sha256) in H2_ABGRALL_RESOURCES.items():
        for line in _verified(resource, sha256).decode("ascii").splitlines():
            if not line.strip():
                continue
            j, v = int(line[1:3]), int(line[5:7])
            if state == "C-":
                total, continuum = float(line[46:55]), float(line[58:67])
            else:
                total, continuum = float(line[70:79]), float(line[82:91])
            rates[(state, v, j)] = (total, continuum)
    return rates


@lru_cache(maxsize=1)
def read_h2_lyman_werner_data() -> H2LymanWernerData:
    """Load the bundled line list and the Abgrall et al. (2000) level rates (checksums verified).

    Kurucz's lines give the wavelengths, oscillator strengths and branching ratios. For every upper
    level tabulated by Abgrall et al. (J <= 10), the bound decay rate is their A_t - A_c, with the
    Kurucz branches rescaled to that total, and the dissociation rate is their A_c. Where rounding of
    the tabulated values exceeds 1 per cent of A_t - A_c (levels that almost always dissociate), the
    Kurucz bound sum is kept. Untabulated levels keep the Kurucz bound sum, with the dissociation
    fraction of the highest tabulated J of the same (state, v).
    """

    pattern = re.compile(r"101X\s*(\d+)\s+([BC])\s*(\d+)([CD]?)")
    wavelength, gf, lower_energy, lower_j, upper_j, upper_energy, state, upper_v, measured = (
        [], [], [], [], [], [], [], [], []
    )
    for line in _verified(H2_LINE_RESOURCE, H2_LINE_SHA256).decode("latin-1").splitlines():
        if not line.strip():
            continue
        match = pattern.search(line)
        wavelength.append(float(line[0:10]) * 10.0)
        gf.append(10.0 ** float(line[10:17]))
        lower = float(line[22:32])
        upper = float(line[37:48])
        measured.append(lower >= 0.0 and upper >= 0.0)
        lower_energy.append(abs(lower))
        lower_j.append(float(line[17:22]))
        upper_j.append(float(line[32:37]))
        upper_energy.append(abs(upper))
        state.append("B" if match.group(2) == "B" else ("C-" if match.group(4) == "D" else "C+"))
        upper_v.append(int(match.group(3)))
    wavelength = np.asarray(wavelength)
    gf = np.asarray(gf)
    upper_j = np.asarray(upper_j)
    einstein_a = _kurucz_einstein_a(wavelength, gf, np.asarray(lower_j), upper_j)
    # Integer key: energy in 0.01 cm^-1 and J' in separate digits, so distinct levels cannot collide.
    keys = np.round(np.asarray(upper_energy) * 100.0).astype(np.int64) * 1000 + np.round(upper_j).astype(np.int64)
    _, upper_index = np.unique(keys, return_inverse=True)
    n_upper = int(upper_index.max()) + 1
    label: dict[int, tuple[str, int, int]] = {}
    for i, (s, v, j) in enumerate(zip(state, upper_v, np.round(upper_j).astype(int))):
        label[int(upper_index[i])] = (s, v, int(j))
    kurucz_bound = np.bincount(upper_index, weights=einstein_a, minlength=n_upper)

    rates = _abgrall_rates()
    bound = kurucz_bound.copy()
    dissociation = np.zeros(n_upper)
    for u, (s, v, j) in label.items():
        if (s, v, j) in rates:
            total, continuum = rates[(s, v, j)]
            emission = total - continuum
            rounding = 0.005 * (10.0 ** np.floor(np.log10(total)) + 10.0 ** np.floor(np.log10(max(continuum, 1e-300))))
            if emission > 0.0 and rounding < 0.01 * emission:
                bound[u] = emission
            dissociation[u] = continuum
        else:
            fraction = 0.0
            for jj in range(min(j, 10), -1, -1):  # untabulated J: hold the highest tabulated J
                if (s, v, jj) in rates:
                    total, continuum = rates[(s, v, jj)]
                    fraction = min(continuum / total, 0.99)
                    break
            dissociation[u] = kurucz_bound[u] * fraction / (1.0 - fraction)
    einstein_a = einstein_a * (bound / kurucz_bound)[upper_index]
    measured = np.asarray(measured)
    return H2LymanWernerData(
        wavelength_angstrom=wavelength[measured],
        gf=gf[measured],
        lower_energy_wavenumber=np.asarray(lower_energy)[measured],
        upper_index=upper_index[measured],
        branch_wavelength_angstrom=wavelength,
        branch_upper_index=upper_index,
        branch_einstein_a=einstein_a,
        upper_bound_decay_rate=bound,
        upper_dissociation_rate=dissociation,
        source=(
            "Kurucz h2bx/h2cx H2 Lyman and Werner lines, opacity from transitions between measured "
            f"levels (SHA256 {H2_LINE_SHA256}); level decay and radiative dissociation rates from "
            "Abgrall, Roueff & Drira (2000)"
        ),
    )


def h2_partition_function(temperature) -> np.ndarray:
    """Full nuclear-spin-weighted H2 partition function, consistent with the EOS H2 density."""

    return 4.0 * molecular_hydrogen_internal_partition_function(np.atleast_1d(temperature))


def h2_source_function_ratio(
    data: H2LymanWernerData,
    radiation_grid_angstrom: np.ndarray,
    mean_intensity_over_planck: np.ndarray,
    temperature: np.ndarray,
    electron_density: np.ndarray,
) -> np.ndarray:
    """Return S/B for every opacity (measured-level) line at every depth, shape (n_lines, n_depth).

    ``mean_intensity_over_planck`` has shape (radiation grid, depth).
    """

    temperature = np.asarray(temperature, dtype=np.float64)
    electron_density = np.asarray(electron_density, dtype=np.float64)
    ratio = np.asarray(mean_intensity_over_planck, dtype=np.float64)
    n_depth = temperature.size
    if ratio.shape != (radiation_grid_angstrom.size, n_depth):
        raise ValueError("mean_intensity_over_planck must have shape (radiation grid, depth)")
    branch_wavelength = data.branch_wavelength_angstrom
    branch_ratio = np.empty((branch_wavelength.size, n_depth))
    for j in range(n_depth):
        branch_ratio[:, j] = np.interp(
            branch_wavelength, radiation_grid_angstrom, ratio[:, j],
            left=ratio[0, j], right=ratio[-1, j],
        )
    n_upper = data.upper_bound_decay_rate.size
    pumped = np.zeros((n_upper, n_depth))
    np.add.at(pumped, data.branch_upper_index, data.branch_einstein_a[:, None] * branch_ratio)
    # van Regemorter (1962): C_ul = 20.6 lambda^3[cm] n_e T^-1/2 A_ul P(y), neutral P(y) = 0.276 e^y E1(y).
    y = (_SECOND_RADIATION_CONSTANT * 1.0e8 / branch_wavelength)[:, None] / temperature[None, :]
    gaunt = 0.276 * np.exp(y) * exp1(y)
    collision_lines = (
        20.6 * (branch_wavelength * 1.0e-8)[:, None] ** 3
        * electron_density[None, :] / np.sqrt(temperature)[None, :]
        * data.branch_einstein_a[:, None] * gaunt
    )
    collisions = np.zeros((n_upper, n_depth))
    np.add.at(collisions, data.branch_upper_index, collision_lines)
    bound = data.upper_bound_decay_rate[:, None]
    dissociation = data.upper_dissociation_rate[:, None]
    upper_ratio = (pumped + dissociation + collisions) / (bound + dissociation + collisions)
    return upper_ratio[data.upper_index]


def h2_line_mass_absorption(
    data: H2LymanWernerData,
    atmosphere,
    wavelength_angstrom,
    source_ratio: np.ndarray | None = None,
    *,
    window_angstrom: float = 1.5,
    relative_strength_cut: float = 1.0e-7,
):
    """Return (kappa, kappa x S/B) summed over H2 lines, each (n_wave, n_depth) in cm^2 g^-1.

    Voigt profiles with the thermal Doppler width and the radiative damping of the upper level.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    kappa = np.zeros((wavelength.size, atmosphere.n_depth))
    weighted = np.zeros_like(kappa)
    state = atmosphere.hydrogen_lte_state
    if state is None or state.molecular_hydrogen_density is None or wavelength.size == 0:
        return kappa, weighted
    h2_density = np.asarray(state.molecular_hydrogen_density, dtype=np.float64)
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    mass_density = np.asarray(atmosphere.mass_density, dtype=np.float64)
    partition = h2_partition_function(temperature)
    select = np.flatnonzero(
        (data.wavelength_angstrom > wavelength.min() - window_angstrom)
        & (data.wavelength_angstrom < wavelength.max() + window_angstrom)
    )
    if select.size == 0:
        return kappa, weighted
    strength = (
        h2_density[None, :] * 4.0 * data.gf[select, None]
        * np.exp(-_SECOND_RADIATION_CONSTANT * data.lower_energy_wavenumber[select, None] / temperature[None, :])
        / partition[None, :]
    )
    keep = strength.max(axis=1) > relative_strength_cut * strength.max()
    damping = (data.upper_bound_decay_rate + data.upper_dissociation_rate)[data.upper_index[select]]
    thermal_speed = np.sqrt(2.0 * _BOLTZMANN * temperature / _H2_MASS)
    order = np.argsort(wavelength)
    sorted_wavelength = wavelength[order]
    for local in np.flatnonzero(keep):
        line = select[local]
        center = data.wavelength_angstrom[line]
        lo, hi = np.searchsorted(sorted_wavelength, [center - window_angstrom, center + window_angstrom])
        if hi <= lo:
            continue
        nu0 = _LIGHT_SPEED / (center * 1.0e-8)
        nu = _LIGHT_SPEED / (sorted_wavelength[lo:hi] * 1.0e-8)
        doppler = nu0 * thermal_speed / _LIGHT_SPEED
        a = damping[local] / (4.0 * np.pi * doppler)
        x = (nu[:, None] - nu0) / doppler[None, :]
        profile = wofz(x + 1j * a[None, :]).real / (np.sqrt(np.pi) * doppler[None, :])
        stimulated = -np.expm1(
            -_SECOND_RADIATION_CONSTANT * (1.0e8 / sorted_wavelength[lo:hi])[:, None] / temperature[None, :]
        )
        contribution = (
            _INTEGRATED_CROSS_SECTION * strength[local][None, :] * profile * stimulated / mass_density[None, :]
        )
        kappa[order[lo:hi]] += contribution
        if source_ratio is not None:
            weighted[order[lo:hi]] += contribution * source_ratio[line][None, :]
    if source_ratio is None:
        weighted = kappa.copy()
    return kappa, weighted
