"""LTE C/O-dominated atmospheres for warm D6 supernova survivors.

The first validation target is SDSS J163712.21+363155.9 from Hollands et al.
(2025).  Unlike a DZ atmosphere, carbon and oxygen are not trace species in a
hydrogen or helium pressure reservoir.  This module therefore solves the
particle equation, multi-element Saha balance, charge neutrality, and mass
density simultaneously for a genuinely hydrogen/helium-free mixture.

The present implementation is a plane-parallel, homogeneous, LTE reference
model. It includes Stout bound-bound opacity, level-resolved TOPbase
photoionization for C/O and several trace species with Verner fallbacks,
ionic free-free opacity, Thomson scattering, non-gray radiative equilibrium,
and optional ML2 convection.
Molecules and negative ions are deliberately omitted at the 15,680 K
milestone: Hollands et al. found their contributions negligible.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d

from ._compat import trapezoid
from ._rosseland import rosseland_mean_from_opacity_grid
from ._spectrum_source import solve_spectrum_source
from .atmosphere import Atmosphere
from .constants import (
    BOLTZMANN,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_ESU,
    LIGHT_SPEED,
    PI,
    PLANCK,
)
from .eos import charged_particle_hydrogen_occupation_probability
from .light_metal_nlte import (
    light_metal_free_free_mass_absorption_coefficient,
    read_tlusty_photoionization_threshold_data,
)
from .metals import (
    ATOMIC_MASS_U,
    ATOMIC_NUMBER,
    AtomicDatabase,
    AtomicIon,
    AtomicTransition,
    EV_TO_WAVENUMBER,
    METAL_LINE_MINIMUM_HALF_WINDOW_ANGSTROM,
    MetalLTEState,
    VernerPhotoionizationDatabase,
    _canonical_element,
    _ion_fractions,
    cool_metal_negative_ion_mass_absorption_coefficient,
    metal_ionic_microfield_perturber_density,
    metal_bound_free_mass_absorption_coefficient,
    metal_line_mass_absorption_coefficient,
    oxygen_i_dissolved_series_mass_absorption_coefficient,
    selected_metal_lines,
    WAVENUMBER_TO_ERG,
)
from .opacity import electron_scattering_mass_coefficient, optical_depth_from_mass_opacity
from .radiative_transfer import Backend, emergent_flux
from .spectrum import Spectrum, planck_lambda_angstrom



FloatArray = NDArray[np.float64]
EV_TO_ERG = 1.602_176_634e-12
ATOMIC_MASS_UNIT = 1.660_539_068_92e-24
RYDBERG_ENERGY_EV = 13.605_693_122_994

_LS_TERM_PATTERN = re.compile(
    r"\((?P<multiplicity>\d+)(?P<letter>[SPDFGHIKLMNOQRTUVWX])"
    r"(?P<odd>o?)<[^>]+>\)"
)
_LS_ANGULAR_MOMENTUM = MappingProxyType({
    letter: angular_momentum
    for angular_momentum, letter in enumerate("SPDFGHIKLMNOQRTUVWX")
})


# Hollands et al. (2025), their Table 2.  Values are logarithmic number ratios
# relative to carbon; upper limits on H, He, and Ni are intentionally excluded.
SDSS_J1637_LOG_NUMBER_ABUNDANCE = MappingProxyType(
    {
        "C": 0.0,
        "O": +0.27,
        "Ne": -1.86,
        "Mg": -2.33,
        "Al": -4.00,
        "Si": -1.76,
        "S": -2.84,
        "Ca": -2.85,
        "Fe": -3.00,
    }
)
SDSS_J1637_EFFECTIVE_TEMPERATURE = 15_680.0
SDSS_J1637_LOGG = 6.3

# The optical opacity is dominated by stages I--III, but the J1637 atmosphere
# reaches roughly 1e5 K at its deepest boundary.  Truncating the Saha ladder at
# charge +2 then traps the dominant C/O/Ne nuclei in the last loaded stage and
# underestimates their electron donation and free-free opacity.  Retain the
# complete Stout ladders for the three bulk electron donors.  The remaining
# trace elements contribute only a few per cent of the nuclei and keep the
# smaller three-stage atoms until their deep-structure effect is demonstrated.
D6_STOUT_MAXIMUM_CHARGE = MappingProxyType(
    {
        element: (
            ATOMIC_NUMBER[element] if element in ("C", "O", "Ne") else 2
        )
        for element in SDSS_J1637_LOG_NUMBER_ABUNDANCE
    }
)


def retain_d6_structure_line_stages(
    atomic_database: AtomicDatabase,
    *,
    maximum_line_charge: int = 2,
) -> AtomicDatabase:
    """Keep complete C/O/Ne EOS levels but only validated low-ion lines.

    Hollands et al. included line opacity for the first low ionization stages
    in J1637.  Higher Stout stages are loaded here to close the deep Saha
    ladder, not to let unobservable hot-ion transitions consume the finite
    line-blanketing budget selected for a 15,680-K atmosphere.
    """

    if maximum_line_charge < 0:
        raise ValueError("maximum_line_charge must be non-negative")
    ions = {
        key: (
            replace(ion, transitions=())
            if ion.charge > maximum_line_charge else ion
        )
        for key, ion in atomic_database.ions.items()
    }
    return AtomicDatabase(
        MappingProxyType(ions),
        source=(
            atomic_database.source
            + f"; lines restricted to charge <= {maximum_line_charge}"
        ),
    )

_SIROCCO_COMMIT = "e3a8c4db3229fc90934d890f882c7edcea68f0f5"
_SIROCCO_RAW_ROOT = (
    "https://raw.githubusercontent.com/sirocco-rt/sirocco/"
    f"{_SIROCCO_COMMIT}/xdata/atomic_macro2"
)
D6_TOPBASE_FILES = MappingProxyType(
    {
        "c_1_levels.dat": (
            f"{_SIROCCO_RAW_ROOT}/c_1_levels.dat",
            "52b18057b11ea89d6c8b178763aae6c0c71ed8dcd2d0219c083cf49bc23284d7",
        ),
        "c_1_phot.dat": (
            f"{_SIROCCO_RAW_ROOT}/c_1_phot.dat",
            "86aa033431484ecc6e9600820892ec2ccbaaf8257ab9810692d9465ff243c8e5",
        ),
        "c_2_levels.dat": (
            f"{_SIROCCO_RAW_ROOT}/c_2_levels.dat",
            "ed297aa3397b01217db485dd9dab0634111e09a9d1a8ddf63307c67370fe2ecf",
        ),
        "c_2_phot.dat": (
            f"{_SIROCCO_RAW_ROOT}/c_2_phot.dat",
            "eb04f7f5b4cd57bb0da4a94d81210590af6f53f6367653a13470c23f1531e554",
        ),
        "o_2_levels.dat": (
            f"{_SIROCCO_RAW_ROOT}/o_2_levels.dat",
            "678fca7d7500f64e1dadb8693dbc9a08ee4cca3dfcc97b96168b7aa29e742db2",
        ),
        "o_2_phot.dat": (
            f"{_SIROCCO_RAW_ROOT}/o_2_phot.dat",
            "10225d051973b1e038c736283033a99615429d5c1f760056ec4b8c76532e7d21",
        ),
    }
)

_TLUSTY_ATOM_ROOT = "https://tlusty.oca.eu/tlusty/Tlusty2002/database/atom"
D6_TLUSTY_TOPBASE_FILES = MappingProxyType(
    {
        "c1_28+12lev.dat": (
            f"{_TLUSTY_ATOM_ROOT}/c1_28%2B12lev.dat",
            "2d6d069cfb1e429d7e6a38e228729f156628b3033ec37b41932ba6585c126eaa",
            "C",
            0,
        ),
        "o1_23+10lev.dat": (
            f"{_TLUSTY_ATOM_ROOT}/o1_23%2B10lev.dat",
            "55c31ec8550a16268d978349fb9aee1eee1a0fce7a1ceb93da6e1b4b10176ba3",
            "O",
            0,
        ),
        "al2_20+9lev.dat": (
            f"{_TLUSTY_ATOM_ROOT}/al2_20%2B9lev.dat",
            "62054ccc4e827a868437e013c00585f24ca3ebc9e271129eac8dad02e045db8e",
            "Al",
            1,
        ),
        "ne1_23+12lev.dat": (
            f"{_TLUSTY_ATOM_ROOT}/ne1_23%2B12lev.dat",
            "1b8463eab829793ffab8d1c15cd34230b77c75d5b3e4f5f4d15a03ebb78ce32e",
            "Ne",
            0,
        ),
        "mg1_12+6lev.dat": (
            f"{_TLUSTY_ATOM_ROOT}/mg1_12%2B6lev.dat",
            "8bef9a8fde605065aaa4380f9e9a8ace549f8edc292dd0cac71e94b6282d5307",
            "Mg",
            0,
        ),
        "mg2_21+4lev.dat": (
            f"{_TLUSTY_ATOM_ROOT}/mg2_21%2B4lev.dat",
            "02d86f5702e1b7be51d87ef26939cd027b09afce7804a84bf4e7cb5352312c87",
            "Mg",
            1,
        ),
        "si1_16+6lev.dat": (
            f"{_TLUSTY_ATOM_ROOT}/si1_16%2B6lev.dat",
            "e36dc88e8d268b072e1e098e0e33d3661a2e4b56689b333d1c3e262e731054a8",
            "Si",
            0,
        ),
        "si2_36+4lev.dat": (
            f"{_TLUSTY_ATOM_ROOT}/si2_36%2B4lev.dat",
            "a194209c8c876ed7b3f80cbf9d24b1299516977a071aa53dc58f8ec9abdbaf8e",
            "Si",
            1,
        ),
        "s2_23+10lev.dat": (
            f"{_TLUSTY_ATOM_ROOT}/s2_23%2B10lev.dat",
            "79d19fca41737511f56a9545aab6dff2ee3bef102525c8b3eede4eadd2786d9d",
            "S",
            1,
        ),
    }
)

D6_TLUSTY_RAP_FILES = MappingProxyType(
    {
        "fe2p_14+11lev.rap": (
            f"{_TLUSTY_ATOM_ROOT}/fe2p_14%2B11lev.rap",
            "54d314b75e6826b84ca80d6705a436f698f66d43b3a3bb022e6d84ca5816f8c3",
            "Fe",
            1,
        ),
    }
)


@dataclass(frozen=True)
class TOPbaseLevelPhotoionization:
    """One SIROCCO/TOPbase level-resolved photoionization section."""

    element: str
    charge: int
    level_index: int
    excitation_energy_ev: float
    statistical_weight: float
    threshold_energy_ev: float
    source_threshold_energy_ev: float
    photon_energy_ev: FloatArray
    cross_section_cm2: FloatArray

    def cross_section(self, photon_energy_ev: ArrayLike) -> FloatArray:
        energy = np.asarray(photon_energy_ev, dtype=np.float64)
        result = np.zeros_like(energy)
        valid = energy >= self.threshold_energy_ev
        if not np.any(valid):
            return result
        # Preserve TOPbase's resonance coordinate relative to threshold while
        # shifting the absolute edge to the experimental/Stout ionization
        # energy.  This is the standard correction needed because raw OP term
        # energies are less accurate than the cross-section shapes.
        # Subtract the two nearby target-coordinate energies before adding
        # the source threshold.  The algebraically equivalent left-to-right
        # expression ``source_threshold + energy - target_threshold`` can
        # round an exact rebased threshold infinitesimally below the first
        # tabulated source point.  That made ``cross_section(target_edge)``
        # spuriously zero and, in turn, silently disabled the dissolved-series
        # pseudocontinuum that samples the threshold value.
        source_energy = self.source_threshold_energy_ev + (
            energy[valid] - self.threshold_energy_ev
        )
        positive = np.flatnonzero(self.cross_section_cm2 > 0.0)
        if positive.size == 0:
            return result
        first = int(positive[0])
        # The first tabulated energy is the first channel covered by this
        # cross-section, even when the source omits explicit leading zeros.
        # In particular, SIROCCO/TOPbase can give a nominal spin-forbidden
        # threshold in the header but start its positive table only at a much
        # higher allowed final-state threshold.  Never extrapolate through
        # that gap.  The comparison is deliberately one-sided so samples just
        # below a delayed opening remain exactly transparent.
        covered = source_energy >= self.photon_energy_ev[first]
        if np.isclose(
            self.photon_energy_ev[first],
            self.source_threshold_energy_ev,
            rtol=1.0e-4,
            atol=0.0,
        ):
            # Some tables round a genuine threshold sample more coarsely than
            # the level header.  Only bridge that small header/table mismatch;
            # it is many orders smaller than a delayed allowed-channel gap.
            covered |= source_energy >= self.source_threshold_energy_ev
        if not np.any(covered):
            return result
        source_covered = source_energy[covered]
        x = np.log(np.maximum(self.photon_energy_ev[first:], 1.0e-300))
        y = np.log(np.maximum(self.cross_section_cm2[first:], 1.0e-300))
        sampled = np.interp(np.log(source_covered), x, y)
        above = source_covered > self.photon_energy_ev[-1]
        if np.any(above):
            sampled[above] = y[-1] - 3.0 * np.log(
                source_covered[above] / self.photon_energy_ev[-1]
            )
        local = np.zeros_like(source_energy)
        local[covered] = np.exp(sampled)
        result[valid] = local
        return result


@dataclass(frozen=True)
class TOPbasePhotoionizationDatabase:
    """Level-resolved TOPbase sections used by a bulk-metal LTE mixture."""

    sections: tuple[TOPbaseLevelPhotoionization, ...]
    source: str

    @property
    def elements(self) -> frozenset[str]:
        return frozenset(section.element for section in self.sections)

    @property
    def ion_stages(self) -> frozenset[tuple[str, int]]:
        """Element/charge pairs covered by at least one level-resolved section."""

        return frozenset(
            (section.element, section.charge) for section in self.sections
        )


def resonance_average_topbase_photoionization(
    database: TOPbasePhotoionizationDatabase,
    fractional_width: float,
) -> TOPbasePhotoionizationDatabase:
    """Average uncertain resonances without opening forbidden channels.

    A constant Gaussian width in log photon energy approximates the resonance-
    averaged photoionization prescription of Bautista, Romano & Pradhan
    (1998).  Leading zero samples are retained exactly: in TOPbase they can
    mark a continuum channel whose first allowed final state opens above the
    lower level's nominal, spin-forbidden threshold.
    """

    if not 0.0 < fractional_width < 0.25:
        raise ValueError("resonance-average fraction must lie between 0 and 0.25")
    averaged = []
    for section in database.sections:
        energy = np.asarray(section.photon_energy_ev, dtype=float)
        cross_section = np.asarray(section.cross_section_cm2, dtype=float)
        positive = np.flatnonzero(cross_section > 0.0)
        if positive.size == 0:
            averaged.append(section)
            continue
        first = int(positive[0])
        active_energy = energy[first:]
        active_cross_section = cross_section[first:]
        if (
            active_energy.size < 4
            or active_energy[-1] <= active_energy[0] * (1.0 + 1.0e-8)
        ):
            averaged.append(section)
            continue
        log_energy = np.log(active_energy)
        minimum_count = int(
            np.ceil((log_energy[-1] - log_energy[0]) / (fractional_width / 10.0))
        ) + 1
        point_count = min(50_000, max(256, active_energy.size, minimum_count))
        uniform_log_energy = np.linspace(log_energy[0], log_energy[-1], point_count)
        uniform_energy = np.exp(uniform_log_energy)
        sampled = np.interp(
            uniform_log_energy,
            log_energy,
            active_cross_section,
        )
        sigma_pixels = fractional_width / (
            uniform_log_energy[1] - uniform_log_energy[0]
        )
        smoothed = gaussian_filter1d(
            sampled,
            sigma=sigma_pixels,
            mode="nearest",
            truncate=4.0,
        )
        original_area = float(trapezoid(active_cross_section, active_energy))
        smoothed_area = float(trapezoid(smoothed, uniform_energy))
        if original_area > 0.0 and smoothed_area > 0.0:
            smoothed *= original_area / smoothed_area
        if first > 0:
            # Keeping an explicit zero immediately below the active interval
            # is essential: TOPbaseLevelPhotoionization.cross_section uses it
            # to distinguish a delayed channel from a rounded edge sample.
            averaged_energy = np.concatenate((energy[:first], uniform_energy))
            averaged_cross_section = np.concatenate(
                (np.zeros(first, dtype=float), np.maximum(smoothed, 0.0))
            )
        else:
            averaged_energy = uniform_energy
            averaged_cross_section = np.maximum(smoothed, 0.0)
        averaged.append(
            replace(
                section,
                photon_energy_ev=averaged_energy,
                cross_section_cm2=averaged_cross_section,
            )
        )
    return TOPbasePhotoionizationDatabase(
        tuple(averaged),
        database.source
        + f"; Gaussian resonance average sigma(E)/E={fractional_width:.3f}",
    )


@dataclass(frozen=True)
class BulkMetalThermodynamics:
    """Thermodynamic derivatives of an ideal, ionizing bulk-metal mixture."""

    specific_heat_constant_pressure: FloatArray
    density_temperature_derivative: FloatArray
    adiabatic_temperature_gradient: FloatArray


# TOPbase level energies are theoretical; their channel thresholds differ
# from the experimental values by up to about 0.3 eV (O II ground: 0.31 eV).
SIROCCO_CHANNEL_ENERGY_TOLERANCE_EV = 0.4
SIROCCO_PARENT_MAXIMUM_EXCITATION_EV = 15.0


def read_sirocco_topbase_lte_photoionization(
    datasets: Iterable[tuple[str | Path, str | Path, str, int]],
    atomic_database: AtomicDatabase,
) -> TOPbasePhotoionizationDatabase:
    """Read SIROCCO's level-resolved TOPbase files for an LTE atmosphere.

    Each dataset is ``(levels_path, photoionization_path, element, charge)``.
    SIROCCO uses the astronomical ion stage in its files, while this API uses
    zero-based charge.  Both its macro-atom ``LevMacro/PhotMac`` files and
    older simple-atom ``LevTop/PhotTop`` files are supported.  Fine-structure
    levels are retained individually and populated with their own statistical
    weights.
    """

    sections: list[TOPbaseLevelPhotoionization] = []
    sources: list[str] = []
    for levels_path, photo_path, element, charge in datasets:
        symbol = _canonical_element(element)
        ion = atomic_database.ions.get((symbol, int(charge)))
        if ion is None or ion.ionization_energy_ev is None:
            raise ValueError(f"missing population ion for {symbol} {charge:+d}")
        levels_source = Path(levels_path)
        photo_source = Path(photo_path)
        # SIROCCO's ``ex`` column uses a common neutral-atom zero point for
        # every ion stage (for example, the C II ground state is listed at
        # the 11.260-eV C I ionization energy).  Convert it to excitation
        # above the requested ion's own ground before applying Boltzmann
        # factors or comparing with that ion's ionization energy.
        raw_levels: dict[tuple[str, int, int], tuple[float, float, int | None]] = {}
        for line in levels_source.read_text(encoding="ascii").splitlines():
            fields = line.split()
            if len(fields) < 8 or fields[0] not in ("LevMacro", "LevTop"):
                continue
            if int(fields[1]) != ATOMIC_NUMBER[symbol]:
                continue
            ion_stage = int(fields[2])
            if ion_stage != charge + 1:
                continue
            if fields[0] == "LevMacro":
                level_index = int(fields[3])
                level_key = ("macro", level_index, 0)
                excitation = float(fields[5])
                weight = float(fields[6])
                output_level_index: int | None = level_index
            else:
                # The older SIROCCO simple-atom format identifies a term by
                # the pair (iSLP, iLV) and supplies its excitation directly
                # in the Te(eV) column.
                level_key = ("top", int(fields[3]), int(fields[4]))
                excitation = float(fields[6])
                weight = float(fields[7])
                output_level_index = None
            if weight <= 0.0:
                continue
            raw_levels[level_key] = (excitation, weight, output_level_index)
        if not raw_levels:
            continue
        stage_ground_energy = min(value[0] for value in raw_levels.values())
        levels = {
            level_key: (
                excitation - stage_ground_energy,
                weight,
                level_index if level_index is not None else ordinal,
            )
            for ordinal, (
                level_key,
                (excitation, weight, level_index),
            ) in enumerate(raw_levels.items(), start=1)
            if 0.0 <= excitation - stage_ground_energy < ion.ionization_energy_ev
        }

        stage_sections: list[tuple[float, float, int, float, FloatArray, FloatArray]] = []
        lines = photo_source.read_text(encoding="ascii").splitlines()
        line_index = 0
        while line_index < len(lines):
            fields = lines[line_index].split()
            line_index += 1
            if not fields or fields[0] not in ("PhotMacS", "PhotTopS"):
                continue
            if len(fields) != 7:
                raise ValueError(f"malformed TOPbase header in {photo_source}")
            atomic_number = int(fields[1])
            ion_stage = int(fields[2])
            if fields[0] == "PhotMacS":
                source_level = ("macro", int(fields[3]), 0)
                point_keyword = "PhotMac"
            else:
                source_level = ("top", int(fields[3]), int(fields[4]))
                point_keyword = "PhotTop"
            source_threshold = float(fields[5])
            point_count = int(fields[6])
            energy = np.empty(point_count, dtype=np.float64)
            cross_section = np.empty(point_count, dtype=np.float64)
            for point_index in range(point_count):
                point = lines[line_index].split()
                line_index += 1
                if len(point) != 3 or point[0] != point_keyword:
                    raise ValueError(f"malformed TOPbase point in {photo_source}")
                energy[point_index] = float(point[1])
                cross_section[point_index] = float(point[2])
            if (
                atomic_number != ATOMIC_NUMBER[symbol]
                or ion_stage != charge + 1
                or source_level not in levels
            ):
                continue
            excitation, weight, output_level_index = levels[source_level]
            if np.any(np.diff(energy) <= 0.0):
                unique_energy, inverse = np.unique(energy, return_inverse=True)
                summed = np.zeros_like(unique_energy)
                count = np.zeros_like(unique_energy)
                np.add.at(summed, inverse, cross_section)
                np.add.at(count, inverse, 1.0)
                energy = unique_energy
                cross_section = summed / count
            if (
                source_threshold <= 0.0
                or np.any(energy <= 0.0)
                or np.any(cross_section < 0.0)
                or not np.any(cross_section > 0.0)
                or np.any(np.diff(energy) <= 0.0)
            ):
                raise ValueError(f"invalid TOPbase section in {photo_source}")
            stage_sections.append((
                excitation, weight, output_level_index, source_threshold,
                energy, cross_section,
            ))
        # A header threshold is the section's own channel opening.  Many
        # channels leave the next ion in an excited parent state (e.g.
        # C II 2s2p2 4P -> C III 2s2p 3P, 6.49 eV above the ground-parent
        # limit), and some tables are assigned to the wrong level.  Rebasing
        # every header onto ``IP - excitation`` drags those channels deep into
        # the optical.  Identify each channel instead: its header must match
        # the experimental binding plus one low parent-level energy of the
        # next ion within the theoretical TOPbase energy error; the edge is
        # then placed at that experimental channel energy.  Sections that
        # match no parent are mis-assigned and are not used.
        parent = atomic_database.ions.get((symbol, int(charge) + 1))
        parent_energies = np.unique(np.concatenate((
            [0.0],
            [
                level.energy_wavenumber / EV_TO_WAVENUMBER
                for level in (parent.levels if parent is not None else ())
                if np.isfinite(level.energy_wavenumber)
                and 0.0 < level.energy_wavenumber / EV_TO_WAVENUMBER
                < SIROCCO_PARENT_MAXIMUM_EXCITATION_EV
            ],
        )))
        rejected = 0
        channels = []
        for section in stage_sections:
            binding = float(ion.ionization_energy_ev - section[0])
            offset = np.abs(section[3] - (binding + parent_energies))
            best = int(np.argmin(offset))
            if offset[best] > SIROCCO_CHANNEL_ENERGY_TOLERANCE_EV:
                rejected += 1
                continue
            channels.append((binding + float(parent_energies[best]), section))
        for threshold, (
            excitation, weight, output_level_index, source_threshold,
            energy, cross_section,
        ) in channels:
            sections.append(
                TOPbaseLevelPhotoionization(
                    symbol,
                    int(charge),
                    output_level_index,
                    excitation,
                    weight,
                    threshold,
                    source_threshold,
                    energy,
                    cross_section,
                )
            )
        sources.append(
            photo_source.name
            + (f" ({rejected} sections without a parent channel omitted)" if rejected else "")
        )
    if not sections:
        raise ValueError("no level-resolved TOPbase sections were loaded")
    return TOPbasePhotoionizationDatabase(
        tuple(sections),
        "SIROCCO/TOPbase level-resolved photoionization: " + ", ".join(sources),
    )


def stout_ground_term_centroid_ev(ion: AtomicIon) -> float:
    """Return the weighted fine-structure centroid of an ion's ground term.

    Stout energies and ionization potentials refer to the lowest ``J`` level.
    LS-coupled model atoms (TLUSTY, Opacity Project) store term-averaged
    energies, so their ground term lies this far above the Stout zero point.
    Levels are grouped by their full label with the ``<J>`` value removed.
    An unparseable ground label returns zero (no fine-structure information).
    """

    levels = [
        level for level in ion.levels
        if np.isfinite(level.energy_wavenumber) and level.statistical_weight > 0.0
    ]
    if not levels:
        return 0.0
    ground = min(levels, key=lambda level: level.energy_wavenumber)
    if re.search(r"<[^>]+>\)", ground.label) is None:
        return 0.0

    def term(label: str) -> str:
        return re.sub(r"<[^>]+>\)", ")", label)

    members = [level for level in levels if term(level.label) == term(ground.label)]
    weights = np.asarray([level.statistical_weight for level in members])
    energies = np.asarray([level.energy_wavenumber for level in members])
    return float(np.sum(weights * energies) / np.sum(weights) / EV_TO_WAVENUMBER)


def read_tlusty_topbase_lte_photoionization(
    datasets: Iterable[tuple[str | Path, str, int]],
    atomic_database: AtomicDatabase,
    *,
    excitation_energy_overrides_ev: Mapping[
        tuple[str, int, int], float
    ] | None = None,
) -> TOPbasePhotoionizationDatabase:
    """Convert public TLUSTY/Opacity-Project model atoms to LTE sections.

    Each input is ``(atom_path, element, zero_based_charge)``. TLUSTY stores
    continuum thresholds as frequencies and detailed TOPbase fits relative to
    those thresholds. The cross-section shapes are sampled densely here,
    while one common ion-stage shift puts the absolute edges on the
    experimental ionization energy in the population database. Model-atom
    energies are term averages, so excitations are measured from the Stout
    ground-term centroid rather than from its lowest ``J`` level (for O I
    this moves every optical edge by about 0.010 eV). Resonance positions
    relative to each edge are preserved.

    The model-atom term energies are theoretical and can be inaccurate enough
    to move an optical edge by many Angstroms.  Callers with an observed term
    assignment may supply ``excitation_energy_overrides_ev`` keyed by
    ``(element, zero_based_charge, one_based_model_level_index)``.  Only the
    absolute threshold is rebased; the tabulated cross-section coordinate
    relative to that threshold is unchanged.
    """

    sections: list[TOPbaseLevelPhotoionization] = []
    sources: list[str] = []
    for atom_path, element, charge in datasets:
        symbol = _canonical_element(element)
        ion = atomic_database.ions.get((symbol, int(charge)))
        if ion is None or ion.ionization_energy_ev is None:
            raise ValueError(f"missing population ion for {symbol} {charge:+d}")
        data = read_tlusty_photoionization_threshold_data(atom_path)
        source_threshold = PLANCK * data.threshold_frequency_hz / EV_TO_ERG
        positive = source_threshold > 0.0
        if not np.any(positive):
            continue
        source_ground_threshold = float(np.max(source_threshold[positive]))
        # Model-atom energies are term averages measured from the ground
        # term; the population database measures from the lowest J level.
        ground_term = stout_ground_term_centroid_ev(ion)
        for level_index, source_edge in enumerate(source_threshold):
            if source_edge <= 0.0:
                continue
            key = (symbol, int(charge), level_index + 1)
            excitation = float(
                (excitation_energy_overrides_ev or {}).get(
                    key,
                    ground_term + source_ground_threshold - float(source_edge),
                )
            )
            target_edge = float(ion.ionization_energy_ev - excitation)
            if excitation < 0.0 or target_edge <= 0.0:
                continue
            x_fit = data.log_frequency_ratio[level_index]
            fit_ratio = (
                np.empty(0)
                if x_fit is None
                else 10.0
                ** np.asarray(x_fit, dtype=np.float64)[
                    np.asarray(x_fit, dtype=np.float64) >= 0.0
                ]
            )
            ratio = np.unique(np.concatenate((
                np.asarray([1.0]),
                fit_ratio,
                np.geomspace(1.0, 1.0e3, 384),
            )))
            source_frequency = data.threshold_frequency_hz[level_index] * ratio
            cross_section = data.cross_section(
                source_frequency,
                data.threshold_frequency_hz[level_index],
                relative_tolerance=1.0e-10,
                level_label=data.level_label[level_index],
            )
            if cross_section is None or not np.any(cross_section > 0.0):
                continue
            sections.append(TOPbaseLevelPhotoionization(
                element=symbol,
                charge=int(charge),
                level_index=level_index + 1,
                excitation_energy_ev=excitation,
                statistical_weight=float(data.statistical_weight[level_index]),
                threshold_energy_ev=target_edge,
                source_threshold_energy_ev=float(source_edge),
                photon_energy_ev=source_edge * ratio,
                cross_section_cm2=np.asarray(cross_section),
            ))
        sources.append(Path(atom_path).name)
    if not sections:
        raise ValueError("no usable TLUSTY/TOPbase photoionization sections")
    return TOPbasePhotoionizationDatabase(
        tuple(sections),
        "TLUSTY/Opacity Project level-resolved photoionization: "
        + ", ".join(sources),
    )


def read_tlusty_rap_lte_photoionization(
    datasets: Iterable[tuple[str | Path, str, int]],
    atomic_database: AtomicDatabase,
) -> TOPbasePhotoionizationDatabase:
    """Read TLUSTY's separate ``.rap`` Opacity Project cross sections.

    Iron-peak model atoms store their level-resolved photoionization data in
    a compact companion file rather than embedding it in the atom file. Each
    section supplies the lower-level excitation energy, statistical weight,
    frequency samples, and cross sections in units of 1e-18 cm2.
    """

    sections: list[TOPbaseLevelPhotoionization] = []
    sources: list[str] = []
    for rap_path, element, charge in datasets:
        symbol = _canonical_element(element)
        ion = atomic_database.ions.get((symbol, int(charge)))
        if ion is None or ion.ionization_energy_ev is None:
            raise ValueError(f"missing population ion for {symbol} {charge:+d}")
        source = Path(rap_path)
        lines = [
            line.strip()
            for line in source.read_text(encoding="ascii").splitlines()
            if line.strip() and not line.lstrip().startswith(("*", "!"))
        ]
        if not lines:
            continue
        header = lines[0].split()
        if len(header) < 3:
            raise ValueError(f"malformed TLUSTY RAP header in {source}")
        atomic_number, ion_stage, section_count = map(int, header[:3])
        if atomic_number != ATOMIC_NUMBER[symbol] or ion_stage != charge + 1:
            raise ValueError(
                f"TLUSTY RAP header in {source} does not describe "
                f"{symbol} {charge:+d}"
            )
        cursor = 1
        loaded = 0
        ground_term = stout_ground_term_centroid_ev(ion)
        for _ in range(section_count):
            if cursor >= len(lines):
                raise ValueError(f"truncated TLUSTY RAP file {source}")
            fields = lines[cursor].split()
            cursor += 1
            if len(fields) < 4:
                raise ValueError(f"malformed TLUSTY RAP section in {source}")
            level_index = int(fields[0])
            excitation_wavenumber = float(fields[1])
            statistical_weight = float(fields[2])
            point_count = int(fields[3])
            if point_count < 2 or cursor + point_count > len(lines):
                raise ValueError(f"invalid TLUSTY RAP point count in {source}")
            frequency = np.empty(point_count, dtype=np.float64)
            cross_section = np.empty(point_count, dtype=np.float64)
            for point in range(point_count):
                values = lines[cursor + point].split()
                if len(values) < 2:
                    raise ValueError(f"malformed TLUSTY RAP data in {source}")
                frequency[point] = float(values[0])
                cross_section[point] = float(values[1]) * 1.0e-18
            cursor += point_count
            if (
                statistical_weight <= 0.0
                or np.any(~np.isfinite(frequency))
                or np.any(np.diff(frequency) <= 0.0)
                or np.any(~np.isfinite(cross_section))
                or np.any(cross_section < 0.0)
            ):
                raise ValueError(f"non-physical TLUSTY RAP section in {source}")
            positive = np.flatnonzero(cross_section > 0.0)
            if positive.size == 0:
                continue
            first_positive = int(positive[0])
            start = max(0, first_positive - 1)
            photon_energy = PLANCK * frequency[start:] / EV_TO_ERG
            # RAP sections share one absolute frequency grid, so the first
            # positive sample is not a threshold.  Use the level's physical
            # edge; energies are LS term averages measured from the ground
            # term, which lies at the Stout ground-term centroid.
            excitation = (
                ground_term + excitation_wavenumber / EV_TO_WAVENUMBER
            )
            target_edge = float(ion.ionization_energy_ev - excitation)
            if excitation < 0.0 or target_edge <= 0.0:
                continue
            source_edge = target_edge
            sections.append(TOPbaseLevelPhotoionization(
                element=symbol,
                charge=int(charge),
                level_index=level_index,
                excitation_energy_ev=excitation,
                statistical_weight=statistical_weight,
                threshold_energy_ev=target_edge,
                source_threshold_energy_ev=source_edge,
                photon_energy_ev=photon_energy,
                cross_section_cm2=cross_section[start:],
            ))
            loaded += 1
        if loaded:
            sources.append(source.name)
    if not sections:
        raise ValueError("no usable TLUSTY RAP photoionization sections")
    return TOPbasePhotoionizationDatabase(
        tuple(sections),
        "TLUSTY/Opacity Project RAP level-resolved photoionization: "
        + ", ".join(sources),
    )


def read_norad_ls_photoionization(
    path: str | Path,
    element: str,
    charge: int,
    atomic_database: AtomicDatabase,
    *,
    maximum_photon_energy_ev: float | None = None,
    excitation_energy_overrides_ev: Mapping[
        tuple[int, int, int, int], float
    ] | None = None,
    only_excitation_overrides: bool = False,
) -> TOPbasePhotoionizationDatabase:
    """Read a NORAD/Iron-Project LS total-photoionization file.

    NORAD ``*.px.txt`` files contain one R-matrix cross-section block for
    every bound LS term.  Energies and binding energies are in Rydbergs and
    cross sections are in megabarns.  The theoretical term binding energies
    determine excitation energies; the entire bundle is shifted onto the
    experimental ionization energy in ``atomic_database`` in the same manner
    as the TOPbase and TLUSTY readers above.  For files whose calculated term
    ordering is unreliable, ``excitation_energy_overrides_ev`` maps the NORAD
    symmetry tuple ``(2S+1, L, parity, index)`` onto an observed excitation
    energy.  ``only_excitation_overrides`` restricts the result to those terms.

    ``maximum_photon_energy_ev`` can trim very large files for a bounded
    synthesis interval.  The first point above the limit is retained so the
    interpolation remains well defined.
    """

    symbol = _canonical_element(element)
    stage = int(charge)
    ion = atomic_database.ions.get((symbol, stage))
    if ion is None or ion.ionization_energy_ev is None:
        raise ValueError(f"missing population ion for {symbol} {stage:+d}")
    if maximum_photon_energy_ev is not None and (
        not np.isfinite(maximum_photon_energy_ev)
        or maximum_photon_energy_ev <= 0.0
    ):
        raise ValueError("maximum_photon_energy_ev must be positive")

    source = Path(path)
    raw_sections: list[
        tuple[int, int, int, int, float, FloatArray, FloatArray]
    ] = []
    excitation_overrides = dict(excitation_energy_overrides_ev or {})
    if only_excitation_overrides and not excitation_overrides:
        raise ValueError(
            "only_excitation_overrides requires excitation-energy overrides"
        )
    expected_atomic_number = ATOMIC_NUMBER[symbol]
    expected_residual_electrons = expected_atomic_number - stage - 1
    with source.open(encoding="ascii") as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            fields = line.split()
            if len(fields) != 3:
                continue
            if fields[2].upper() == "P":
                try:
                    atomic_number = int(fields[0])
                    residual_electrons = int(fields[1])
                except ValueError as error:
                    raise ValueError(f"malformed NORAD header in {source}") from error
                if (
                    atomic_number != expected_atomic_number
                    or residual_electrons != expected_residual_electrons
                ):
                    raise ValueError(
                        f"NORAD header in {source} does not describe "
                        f"{symbol} {stage:+d}"
                    )
                # Older Iron-Project files use one global ``Z Ne P`` header
                # followed directly by LS blocks.  Their block header is
                # symmetry; (internal index, point count); (binding, accuracy).
                while True:
                    symmetry_line = stream.readline()
                    if not symmetry_line:
                        break
                    if not symmetry_line.strip():
                        continue
                    symmetry_fields = symmetry_line.split()
                    if len(symmetry_fields) >= 4:
                        try:
                            symmetry_values = tuple(
                                map(int, symmetry_fields[:4])
                            )
                        except ValueError:
                            symmetry_values = ()
                        if symmetry_values == (0, 0, 0, 0):
                            break
                    count_line = stream.readline()
                    binding_line = stream.readline()
                    if not count_line or not binding_line:
                        raise ValueError(f"truncated NORAD section in {source}")
                    try:
                        multiplicity, angular_momentum, parity, symmetry_index = map(
                            int, symmetry_fields[:4]
                        )
                        _internal_index, point_count = map(int, count_line.split()[:2])
                        binding_fields = binding_line.split()
                        binding_rydberg = abs(float(binding_fields[0]))
                        float(binding_fields[1])
                    except (ValueError, IndexError) as error:
                        raise ValueError(
                            f"malformed NORAD section header in {source}"
                        ) from error
                    if (
                        multiplicity <= 0
                        or angular_momentum < 0
                        or parity not in (0, 1)
                        or symmetry_index <= 0
                        or binding_rydberg <= 0.0
                        or point_count < 2
                    ):
                        raise ValueError(
                            f"non-physical NORAD section header in {source}"
                        )
                    energy_rydberg = np.empty(point_count, dtype=np.float64)
                    cross_section_megabar = np.empty(point_count, dtype=np.float64)
                    for point in range(point_count):
                        values = stream.readline().split()
                        if len(values) < 2:
                            raise ValueError(
                                f"truncated NORAD cross section in {source}"
                            )
                        energy_rydberg[point] = float(values[0])
                        cross_section_megabar[point] = float(values[-1])
                    if (
                        np.any(~np.isfinite(energy_rydberg))
                        or np.any(energy_rydberg <= 0.0)
                        or np.any(~np.isfinite(cross_section_megabar))
                        or np.any(cross_section_megabar < 0.0)
                    ):
                        raise ValueError(
                            f"non-physical NORAD cross section in {source}"
                        )
                    raw_sections.append((
                        multiplicity,
                        angular_momentum,
                        parity,
                        symmetry_index,
                        binding_rydberg * RYDBERG_ENERGY_EV,
                        energy_rydberg * RYDBERG_ENERGY_EV,
                        cross_section_megabar * 1.0e-18,
                    ))
                break
            try:
                atomic_number, residual_electrons, target_count = map(int, fields)
            except ValueError:
                continue
            if target_count <= 0:
                continue
            if (
                atomic_number != expected_atomic_number
                or residual_electrons != expected_residual_electrons
            ):
                raise ValueError(
                    f"NORAD header in {source} does not describe {symbol} {stage:+d}"
                )

            target_energy_count = 0
            while target_energy_count < target_count:
                target_line = stream.readline()
                if not target_line:
                    raise ValueError(f"truncated NORAD target energies in {source}")
                try:
                    target_energy_count += len([
                        float(value) for value in target_line.split()
                    ])
                except ValueError as error:
                    raise ValueError(
                        f"malformed NORAD target energies in {source}"
                    ) from error
            if target_energy_count != target_count:
                raise ValueError(f"wrong NORAD target-energy count in {source}")

            symmetry_line = stream.readline()
            binding_line = stream.readline()
            accuracy_line = stream.readline()
            if not symmetry_line or not binding_line or not accuracy_line:
                raise ValueError(f"truncated NORAD section header in {source}")
            try:
                multiplicity, angular_momentum, parity, symmetry_index = map(
                    int, symmetry_line.split()
                )
                binding_rydberg, point_count_text = binding_line.split()[:2]
                binding_rydberg = abs(float(binding_rydberg))
                point_count = int(point_count_text)
                float(accuracy_line)
            except (ValueError, IndexError) as error:
                raise ValueError(f"malformed NORAD section header in {source}") from error
            if (
                multiplicity <= 0
                or angular_momentum < 0
                or parity not in (0, 1)
                or symmetry_index <= 0
                or binding_rydberg <= 0.0
                or point_count < 2
            ):
                raise ValueError(f"non-physical NORAD section header in {source}")

            energy_rydberg = np.empty(point_count, dtype=np.float64)
            cross_section_megabar = np.empty(point_count, dtype=np.float64)
            for point in range(point_count):
                values = stream.readline().split()
                if len(values) < 2:
                    raise ValueError(f"truncated NORAD cross section in {source}")
                try:
                    energy_rydberg[point] = float(values[0])
                    cross_section_megabar[point] = float(values[-1])
                except ValueError as error:
                    raise ValueError(
                        f"malformed NORAD cross section in {source}"
                    ) from error
            if (
                np.any(~np.isfinite(energy_rydberg))
                or np.any(energy_rydberg <= 0.0)
                or np.any(~np.isfinite(cross_section_megabar))
                or np.any(cross_section_megabar < 0.0)
            ):
                raise ValueError(f"non-physical NORAD cross section in {source}")
            photon_energy = energy_rydberg * RYDBERG_ENERGY_EV
            cross_section = cross_section_megabar * 1.0e-18
            if np.any(np.diff(photon_energy) <= 0.0):
                unique_energy, inverse = np.unique(photon_energy, return_inverse=True)
                summed = np.zeros_like(unique_energy)
                count = np.zeros_like(unique_energy)
                np.add.at(summed, inverse, cross_section)
                np.add.at(count, inverse, 1.0)
                photon_energy = unique_energy
                cross_section = summed / count
            raw_sections.append((
                multiplicity,
                angular_momentum,
                parity,
                symmetry_index,
                binding_rydberg * RYDBERG_ENERGY_EV,
                photon_energy,
                cross_section,
            ))

    if not raw_sections:
        raise ValueError(f"no NORAD photoionization sections in {source}")
    source_ground_threshold = max(section[4] for section in raw_sections)
    sections: list[TOPbaseLevelPhotoionization] = []
    for level_index, (
        multiplicity,
        angular_momentum,
        parity,
        symmetry_index,
        source_threshold,
        photon_energy,
        cross_section,
    ) in enumerate(raw_sections, start=1):
        symmetry = (
            multiplicity,
            angular_momentum,
            parity,
            symmetry_index,
        )
        if only_excitation_overrides and symmetry not in excitation_overrides:
            continue
        excitation = excitation_overrides.get(
            symmetry,
            source_ground_threshold - source_threshold,
        )
        target_threshold = float(ion.ionization_energy_ev - excitation)
        if excitation < -1.0e-8 or target_threshold <= 0.0:
            continue
        if photon_energy[0] > source_threshold * (1.0 + 1.0e-10):
            photon_energy = np.concatenate(([source_threshold], photon_energy))
            cross_section = np.concatenate(([0.0], cross_section))
        if maximum_photon_energy_ev is not None:
            stop = int(np.searchsorted(
                photon_energy, maximum_photon_energy_ev, side="right"
            ))
            stop = min(photon_energy.size, max(2, stop + 1))
            photon_energy = photon_energy[:stop]
            cross_section = cross_section[:stop]
            if photon_energy[0] > maximum_photon_energy_ev:
                continue
        if not np.any(cross_section > 0.0):
            continue
        sections.append(TOPbaseLevelPhotoionization(
            element=symbol,
            charge=stage,
            level_index=level_index,
            excitation_energy_ev=max(0.0, float(excitation)),
            statistical_weight=float(multiplicity * (2 * angular_momentum + 1)),
            threshold_energy_ev=target_threshold,
            source_threshold_energy_ev=float(source_threshold),
            photon_energy_ev=photon_energy,
            cross_section_cm2=cross_section,
        ))
    if not sections:
        raise ValueError(f"no usable NORAD photoionization sections in {source}")
    return TOPbasePhotoionizationDatabase(
        tuple(sections),
        f"NORAD/Iron Project LS photoionization: {source.name}",
    )


def stout_ls_term_excitation_overrides(
    atomic_database: AtomicDatabase,
    element: str,
    charge: int,
    *,
    maximum_excitation_energy_ev: float | None = None,
) -> dict[tuple[int, int, int, int], float]:
    """Map observed Stout fine-structure energies onto NORAD LS terms.

    Stout level labels retain the parent LS term and fine-structure ``J``.
    This routine groups the fine-structure components by configuration and
    term, computes their statistical-weighted centroid, and numbers repeated
    terms of the same ``(2S+1, L, parity)`` symmetry in increasing observed
    excitation energy.  The resulting keys can be passed directly as
    ``excitation_energy_overrides_ev`` to
    :func:`read_norad_ls_photoionization`.

    Levels whose Stout labels do not expose an unambiguous LS term are skipped.
    This is intentional: callers can combine the returned mapping with
    ``only_excitation_overrides=True`` to obtain a conservative bundle based
    only on experimentally identified terms.
    """

    symbol = _canonical_element(element)
    stage = int(charge)
    ion = atomic_database.ions.get((symbol, stage))
    if ion is None:
        raise ValueError(f"missing population ion for {symbol} {stage:+d}")
    if maximum_excitation_energy_ev is not None and (
        not np.isfinite(maximum_excitation_energy_ev)
        or maximum_excitation_energy_ev < 0.0
    ):
        raise ValueError("maximum_excitation_energy_ev must be non-negative")

    grouped: dict[
        tuple[int, int, int, str], list[tuple[float, float]]
    ] = {}
    for level in ion.levels:
        matches = tuple(_LS_TERM_PATTERN.finditer(level.label))
        if not matches:
            continue
        match = matches[-1]
        multiplicity = int(match.group("multiplicity"))
        angular_momentum = _LS_ANGULAR_MOMENTUM[match.group("letter")]
        parity = int(bool(match.group("odd")))
        excitation = float(level.energy_wavenumber / EV_TO_WAVENUMBER)
        weight = float(level.statistical_weight)
        if (
            not np.isfinite(excitation)
            or excitation < 0.0
            or not np.isfinite(weight)
            or weight <= 0.0
        ):
            continue
        # Retain the configuration text so distinct terms with the same LS
        # symmetry are combined only across their fine-structure components.
        identity = (
            level.label[: match.start()]
            + f"({multiplicity}{match.group('letter')}"
            + ("o" if parity else "")
            + ")"
            + level.label[match.end() :]
        )
        grouped.setdefault(
            (multiplicity, angular_momentum, parity, identity), []
        ).append((excitation, weight))

    by_symmetry: dict[tuple[int, int, int], list[tuple[float, str]]] = {}
    for (multiplicity, angular_momentum, parity, identity), components in grouped.items():
        weights = np.asarray([item[1] for item in components], dtype=float)
        excitation = float(np.average(
            [item[0] for item in components], weights=weights
        ))
        if (
            maximum_excitation_energy_ev is not None
            and excitation > maximum_excitation_energy_ev
        ):
            continue
        by_symmetry.setdefault(
            (multiplicity, angular_momentum, parity), []
        ).append((excitation, identity))

    result: dict[tuple[int, int, int, int], float] = {}
    for symmetry, terms in by_symmetry.items():
        for symmetry_index, (excitation, _identity) in enumerate(
            sorted(terms), start=1
        ):
            result[(*symmetry, symmetry_index)] = excitation
    if not result:
        raise ValueError(f"no observed LS terms found for {symbol} {stage:+d}")
    return result


def d6_tlusty_excitation_energy_overrides(
    atomic_database: AtomicDatabase,
    *,
    elements: Iterable[str] | None = None,
) -> dict[tuple[str, int, int], float]:
    """Return securely identified observed energies for the D6 TLUSTY atoms.

    The public TLUSTY model atoms use theoretical term energies.  O I model
    level 6 is the observed ``3p 5P`` term, which carries the optical series
    limit near 4309 A.  Measured from the Stout ground-term centroid its
    theoretical energy already places that limit within 0.5 A; the observed
    term energy removes the remainder.  (The 15-A discrepancy once attributed
    to the theoretical energy was the J-level versus term-centroid zero
    point.)  Keep this model-level assignment in one shared helper so every
    D6 entry point applies the same correction.

    Only assignments independently identified against the Stout/NIST level
    set belong here.  Unmatched theoretical terms are deliberately left
    unchanged rather than being paired by nearest energy alone.
    """

    selected = (
        set(atomic_database.elements)
        if elements is None
        else {_canonical_element(element) for element in elements}
    )
    result: dict[tuple[str, int, int], float] = {}
    if "O" in selected and ("O", 0) in atomic_database.ions:
        oxygen_terms = stout_ls_term_excitation_overrides(
            atomic_database,
            "O",
            0,
            maximum_excitation_energy_ev=12.0,
        )
        oxygen_3p_5p = oxygen_terms.get((5, 1, 0, 1))
        if oxygen_3p_5p is not None:
            result[("O", 0, 6)] = oxygen_3p_5p
    return result


def merge_topbase_photoionization_databases(
    *databases: TOPbasePhotoionizationDatabase,
) -> TOPbasePhotoionizationDatabase:
    """Combine disjoint level-resolved photoionization bundles."""

    if not databases:
        raise ValueError("at least one photoionization database is required")
    sections: list[TOPbaseLevelPhotoionization] = []
    occupied: set[tuple[str, int]] = set()
    for database in databases:
        stages = {(section.element, section.charge) for section in database.sections}
        overlap = occupied & stages
        if overlap:
            raise ValueError(f"duplicate photoionization ion stages: {sorted(overlap)}")
        occupied |= stages
        sections.extend(database.sections)
    return TOPbasePhotoionizationDatabase(
        tuple(sections),
        " + ".join(database.source for database in databases),
    )


def topbase_bound_free_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    metal_state: MetalLTEState,
    database: TOPbasePhotoionizationDatabase,
) -> FloatArray:
    """Return level-resolved TOPbase bound-free opacity in cm2 g-1."""

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    photon_energy = PLANCK * LIGHT_SPEED / (wavelength * 1.0e-8) / EV_TO_ERG
    stimulated = -np.expm1(
        -PLANCK * LIGHT_SPEED
        / (
            wavelength[:, np.newaxis] * 1.0e-8
            * BOLTZMANN * atmosphere.temperature[np.newaxis, :]
        )
    )
    result = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    for section in database.sections:
        populations = metal_state.ion_number_density[section.element]
        if section.charge >= populations.shape[0]:
            continue
        partition = metal_state.partition_function[(section.element, section.charge)]
        lower_population = (
            populations[section.charge]
            * section.statistical_weight
            * np.exp(
                -section.excitation_energy_ev * EV_TO_ERG
                / (BOLTZMANN * atmosphere.temperature)
            )
            / partition
        )
        result += (
            section.cross_section(photon_energy)[:, np.newaxis]
            * lower_population[np.newaxis, :]
            * stimulated
            / atmosphere.mass_density[np.newaxis, :]
        )
    return result


def topbase_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    metal_state: MetalLTEState,
    database: TOPbasePhotoionizationDatabase,
    *,
    elements: Iterable[str] = ("O", "Mg"),
    correlated_microfields: bool = True,
    upper_level_offset: float = 3.0,
    continuum_cutoff_probability: float | None = None,
) -> FloatArray:
    """Return a DAM/HM continuation of level-resolved metal continua.

    A photon just redward of a bound-free edge is mapped to the fictitious
    Rydberg level whose binding energy is the difference between the edge and
    photon energies.  The dissolved fraction ``1 - w_upper / w_lower`` then
    continues the threshold cross section into the high-series interval.  As
    in the hydrogen implementation, the continuation stops when the
    fictitious upper effective principal quantum number falls below three
    shells above the lower state.  This prevents an unphysical optical/IR tail
    from a raw DAM continuation.

    The lower-state population and threshold cross section are the same
    level-resolved Opacity Project quantities used by the ordinary continuum,
    and the charged-particle survival is the package's Q-MHD/Holtsmark
    closure. The default includes oxygen and magnesium, for which the package
    uses level-resolved TOPbase continua and whose strong optical series limits
    otherwise remain unphysically abrupt in cool metal-dominated atmospheres.
    Other elements can be selected explicitly as their level assignments are
    validated.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be a finite positive 1D array")
    if not np.isfinite(upper_level_offset) or upper_level_offset <= 0.0:
        raise ValueError("upper_level_offset must be finite and positive")
    if (
        continuum_cutoff_probability is not None
        and (
            not np.isfinite(continuum_cutoff_probability)
            or not 0.0 < continuum_cutoff_probability < 1.0
        )
    ):
        raise ValueError("the continuum-cutoff probability must lie in (0, 1)")
    selected_elements = {_canonical_element(element) for element in elements}
    result = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    photon_energy = (
        PLANCK * LIGHT_SPEED / (wavelength * 1.0e-8) / EV_TO_ERG
    )
    charged_perturber_density = metal_ionic_microfield_perturber_density(
        metal_state
    )
    for section in database.sections:
        if section.element not in selected_elements:
            continue
        populations = metal_state.ion_number_density.get(section.element)
        if populations is None or section.charge >= populations.shape[0]:
            continue
        threshold = float(section.threshold_energy_ev)
        core_charge = float(section.charge + 1)
        lower_effective_n = max(
            1.0,
            float(np.sqrt(
                RYDBERG_ENERGY_EV * core_charge**2 / threshold
            )),
        )
        minimum_upper_n = lower_effective_n + upper_level_offset
        maximum_upper_binding = (
            RYDBERG_ENERGY_EV * core_charge**2 / minimum_upper_n**2
        )
        minimum_photon_energy = threshold - maximum_upper_binding
        if minimum_photon_energy <= 0.0:
            continue
        valid = (
            (photon_energy < threshold)
            & (photon_energy > minimum_photon_energy)
        )
        if not np.any(valid):
            continue
        selected_photon_energy = photon_energy[valid]
        # A structure-grid point can agree with the shifted series limit to
        # machine precision while still compare infinitesimally redward.  Do
        # not let that harmless roundoff turn n* into infinity; n*=10^6 is
        # already completely dissolved for any atmosphere represented here.
        minimum_binding = (
            RYDBERG_ENERGY_EV * core_charge**2 / 1.0e12
        )
        upper_binding = np.maximum(
            threshold - selected_photon_energy,
            minimum_binding,
        )
        upper_effective_n = np.sqrt(
            RYDBERG_ENERGY_EV * core_charge**2 / upper_binding
        )
        temperature = (
            atmosphere.temperature[np.newaxis, :]
            if correlated_microfields else None
        )
        upper_survival = charged_particle_hydrogen_occupation_probability(
            charged_perturber_density[np.newaxis, :],
            upper_effective_n[:, np.newaxis],
            temperature,
            ionic_charge=core_charge,
        )
        lower_survival = charged_particle_hydrogen_occupation_probability(
            charged_perturber_density,
            lower_effective_n,
            atmosphere.temperature if correlated_microfields else None,
            ionic_charge=core_charge,
        )
        bound_survival = np.clip(
            upper_survival
            / np.maximum(lower_survival[np.newaxis, :], np.finfo(np.float64).tiny),
            0.0,
            1.0,
        )
        if continuum_cutoff_probability is not None:
            bound_survival = (
                bound_survival >= continuum_cutoff_probability
            ).astype(np.float64)
        dissolved_fraction = 1.0 - bound_survival
        threshold_cross_section = float(
            section.cross_section(np.asarray([threshold]))[0]
        )
        if threshold_cross_section <= 0.0:
            continue
        series_limit = (
            PLANCK * LIGHT_SPEED / (threshold * EV_TO_ERG) * 1.0e8
        )
        selected_wavelength = wavelength[valid]
        extrapolated_cross_section = (
            threshold_cross_section * (selected_wavelength / series_limit) ** 3
        )
        partition = metal_state.partition_function[
            (section.element, section.charge)
        ]
        lower_population = (
            populations[section.charge]
            * section.statistical_weight
            * np.exp(
                -section.excitation_energy_ev * EV_TO_ERG
                / (BOLTZMANN * atmosphere.temperature)
            )
            / partition
        )
        stimulated = -np.expm1(
            -selected_photon_energy[:, np.newaxis]
            * EV_TO_ERG
            / (BOLTZMANN * atmosphere.temperature[np.newaxis, :])
        )
        result[valid] += (
            lower_population[np.newaxis, :]
            / atmosphere.mass_density[np.newaxis, :]
            * extrapolated_cross_section[:, np.newaxis]
            * dissolved_fraction
            * stimulated
        )
    return result


def _validate_bulk_abundances(
    log_number_abundance: Mapping[str, float],
    reference_element: str,
) -> tuple[str, dict[str, float], dict[str, float]]:
    reference = _canonical_element(reference_element)
    abundances: dict[str, float] = {}
    for element, abundance in log_number_abundance.items():
        symbol = _canonical_element(element)
        value = float(abundance)
        if not np.isfinite(value):
            raise ValueError("log abundances must be finite")
        abundances[symbol] = value
    if reference not in abundances:
        abundances[reference] = 0.0
    if not np.isclose(abundances[reference], 0.0, atol=1.0e-12):
        raise ValueError("the reference element must have log abundance zero")
    if not abundances:
        raise ValueError("at least one bulk element is required")
    ratios = {element: 10.0**value for element, value in abundances.items()}
    return reference, abundances, ratios


def bulk_metal_lte_state(
    atmosphere: Atmosphere,
    atomic_database: AtomicDatabase,
    log_number_abundance: Mapping[str, float],
    *,
    reference_element: str = "C",
    include_nonideal_partitions: bool = False,
) -> MetalLTEState:
    """Solve a homogeneous multi-element LTE mixture at fixed pressure.

    Every supplied element contributes nuclei, ions, electrons, and mass.  No
    hidden H/He host is present. Number abundances are logarithmic ratios to
    ``reference_element`` and the reference itself must be zero dex. The
    optional non-ideal mode uses Q-MHD level survival in both the partition
    functions and the bound-bound lower-level populations.
    """

    _, abundances, number_ratio = _validate_bulk_abundances(
        log_number_abundance, reference_element
    )
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    pressure = np.asarray(atmosphere.gas_pressure, dtype=np.float64)
    if temperature.shape != pressure.shape or temperature.ndim != 1:
        raise ValueError("atmosphere temperature and pressure grids must match")
    particle_density = pressure / (BOLTZMANN * temperature)
    translational_log = 1.5 * np.log(
        2.0 * PI * ELECTRON_MASS * BOLTZMANN * temperature / PLANCK**2
    )

    ion_stages = {}
    ideal_partitions: dict[tuple[str, int], FloatArray] = {}
    for element in abundances:
        stages = atomic_database.ion_stages(element)
        if len(stages) < 2:
            raise ValueError(f"at least two ion stages are required for {element}")
        ion_stages[element] = stages
        for ion in stages:
            ideal_partitions[(element, ion.charge)] = ion.partition_function(temperature)

    nuclei_per_reference = sum(number_ratio.values())
    partitions = dict(ideal_partitions)

    def fractions_and_charge(
        electron_density: FloatArray,
    ) -> tuple[dict[str, FloatArray], FloatArray]:
        fractions: dict[str, FloatArray] = {}
        charge_per_reference = np.zeros_like(temperature)
        for element, stages in ion_stages.items():
            log_ratios = []
            for lower, upper in zip(stages[:-1], stages[1:]):
                if lower.ionization_energy_ev is None:
                    raise ValueError(
                        f"missing ionization energy for {element} {lower.charge:+d}"
                    )
                log_ratios.append(
                    np.log(2.0)
                    + translational_log
                    + np.log(partitions[(element, upper.charge)])
                    - np.log(partitions[(element, lower.charge)])
                    - lower.ionization_energy_ev * EV_TO_ERG
                    / (BOLTZMANN * temperature)
                    - np.log(electron_density)
                )
            fraction = _ion_fractions(np.stack(log_ratios))
            fractions[element] = fraction
            charge_per_reference += number_ratio[element] * np.sum(
                np.arange(fraction.shape[0])[:, np.newaxis] * fraction,
                axis=0,
            )
        return fractions, charge_per_reference

    def solve_charge_neutrality() -> FloatArray:
        lower = np.full_like(temperature, np.log(np.finfo(np.float64).tiny))
        upper = np.log(np.maximum(particle_density * (1.0 - 1.0e-12), 1.0))
        for _ in range(64):
            middle = 0.5 * (lower + upper)
            trial_electron_density = np.exp(middle)
            _, trial_charge = fractions_and_charge(trial_electron_density)
            reference_density = particle_density / (
                nuclei_per_reference + trial_charge
            )
            required_electrons = reference_density * trial_charge
            positive = trial_electron_density > required_electrons
            upper = np.where(positive, middle, upper)
            lower = np.where(positive, lower, middle)
        return np.exp(0.5 * (lower + upper))

    electron_density = solve_charge_neutrality()
    if include_nonideal_partitions:
        # Partition functions depend on the ionic microfield density moment.
        # A small outer fixed-point iteration avoids recomputing every level
        # sum in all 64 scalar charge-neutrality bisection steps.
        for _ in range(8):
            trial_fractions, trial_charge = fractions_and_charge(
                electron_density
            )
            trial_reference_density = particle_density / (
                nuclei_per_reference + trial_charge
            )
            charged_perturber_density = np.zeros_like(electron_density)
            for element, fraction in trial_fractions.items():
                charged_perturber_density += (
                    trial_reference_density
                    * number_ratio[element]
                    * np.sum(
                        np.arange(fraction.shape[0])[:, np.newaxis] ** 1.5
                        * fraction,
                        axis=0,
                    )
                )
            updated = {
                (element, ion.charge): ion.occupation_weighted_partition_function(
                    temperature, charged_perturber_density
                )
                for element, stages in ion_stages.items()
                for ion in stages
            }
            maximum_change = max(
                float(np.max(np.abs(np.log(
                    updated[key] / np.maximum(
                        partitions[key], np.finfo(np.float64).tiny
                    )
                ))))
                for key in updated
            )
            partitions = updated
            electron_density = solve_charge_neutrality()
            if maximum_change < 1.0e-10:
                break
    fractions, charge_per_reference = fractions_and_charge(electron_density)
    reference_density = particle_density / (
        nuclei_per_reference + charge_per_reference
    )
    element_density = {
        element: np.asarray(reference_density * ratio)
        for element, ratio in number_ratio.items()
    }
    populations = {
        element: element_density[element][np.newaxis, :] * fractions[element]
        for element in abundances
    }
    mass_density = np.zeros_like(temperature)
    for element, density in element_density.items():
        mass_density += ATOMIC_MASS_U[element] * ATOMIC_MASS_UNIT * density

    total_mass = sum(number_ratio[e] * ATOMIC_MASS_U[e] for e in abundances)
    mass_fraction = {
        element: number_ratio[element] * ATOMIC_MASS_U[element] / total_mass
        for element in abundances
    }
    return MetalLTEState(
        reference_species="metal",
        log_number_abundance=MappingProxyType(dict(abundances)),
        element_number_density=MappingProxyType(element_density),
        ion_number_density=MappingProxyType(populations),
        partition_function=MappingProxyType(partitions),
        electron_density=np.asarray(electron_density),
        metal_electron_density=np.asarray(electron_density),
        nonideal_ionization=bool(include_nonideal_partitions),
        metal_level_dissolution=bool(include_nonideal_partitions),
        composition_mode="bulk",
        mass_fraction=MappingProxyType(mass_fraction),
        total_mass_density=np.asarray(mass_density),
    )


def atmosphere_with_bulk_metal_state(
    atmosphere: Atmosphere,
    state: MetalLTEState,
) -> Atmosphere:
    """Attach the pressure-consistent pure-metal density and electron state."""

    if state.reference_species != "metal" or state.total_mass_density is None:
        raise ValueError("a bulk pure-metal LTE state is required")
    zeros = np.zeros_like(atmosphere.temperature)
    metadata = dict(atmosphere.metadata)
    metadata.update(
        {
            "composition": "hydrogen-helium-free-bulk-metals",
            "metal_abundances": dict(state.log_number_abundance),
            "mass_fractions": dict(state.mass_fraction or {}),
            "metal_electron_feedback": "fixed-pressure particle and charge closure",
            "metal_composition_mode": "bulk",
        }
    )
    return replace(
        atmosphere,
        mass_density=np.asarray(state.total_mass_density),
        neutral_h_density=zeros,
        proton_density=zeros,
        electron_density=np.asarray(state.electron_density),
        hydrogen_lte_state=None,
        helium_lte_state=None,
        metadata=metadata,
    )


def _bulk_metal_specific_enthalpy(
    atmosphere: Atmosphere,
    state: MetalLTEState,
    atomic_database: AtomicDatabase,
) -> FloatArray:
    """Return LTE specific enthalpy including excitation and ionization."""

    if state.total_mass_density is None:
        raise ValueError("bulk-metal mass density is required")
    temperature = atmosphere.temperature
    internal_energy_density = np.zeros_like(temperature)
    for element, populations in state.ion_number_density.items():
        cumulative_ionization_ev = 0.0
        for charge, ion in enumerate(atomic_database.ion_stages(element)):
            if charge >= populations.shape[0]:
                break
            energy = np.asarray(
                [level.energy_wavenumber for level in ion.levels],
                dtype=np.float64,
            )
            weight = np.asarray(
                [level.statistical_weight for level in ion.levels],
                dtype=np.float64,
            )
            selected = np.isfinite(energy) & (energy >= 0.0) & (weight > 0.0)
            if ion.ionization_energy_ev is not None:
                cutoff_ev = max(ion.ionization_energy_ev - 0.1, 0.0)
                selected &= energy <= cutoff_ev * EV_TO_WAVENUMBER
            exponent = (
                -WAVENUMBER_TO_ERG * energy[selected, np.newaxis]
                / (BOLTZMANN * temperature[np.newaxis, :])
            )
            boltzmann_weight = weight[selected, np.newaxis] * np.exp(exponent)
            mean_excitation = (
                np.sum(
                    boltzmann_weight
                    * energy[selected, np.newaxis]
                    * WAVENUMBER_TO_ERG,
                    axis=0,
                )
                / np.sum(boltzmann_weight, axis=0)
            )
            internal_energy_density += populations[charge] * (
                cumulative_ionization_ev * EV_TO_ERG + mean_excitation
            )
            if ion.ionization_energy_ev is not None:
                cumulative_ionization_ev += ion.ionization_energy_ev
    density = np.asarray(state.total_mass_density)
    return (
        2.5 * atmosphere.gas_pressure + internal_energy_density
    ) / density


def bulk_metal_thermodynamics(
    atmosphere: Atmosphere,
    atomic_database: AtomicDatabase,
    log_number_abundance: Mapping[str, float],
    *,
    reference_element: str = "C",
    relative_temperature_step: float = 1.0e-3,
) -> BulkMetalThermodynamics:
    """Return fixed-pressure thermodynamic derivatives for a metal mixture.

    Central finite differences include the latent heat and density response of
    every Saha ionization stage, as well as excitation of the same bound levels
    used by the partition functions.  This is the composition-independent EOS
    information required by the local ML2 closure.
    """

    if not 0.0 < relative_temperature_step < 0.1:
        raise ValueError("relative_temperature_step must lie between 0 and 0.1")
    step = float(relative_temperature_step)

    def state_at(temperature: FloatArray) -> tuple[Atmosphere, MetalLTEState]:
        provisional = replace(atmosphere, temperature=np.asarray(temperature))
        state = bulk_metal_lte_state(
            provisional,
            atomic_database,
            log_number_abundance,
            reference_element=reference_element,
        )
        return atmosphere_with_bulk_metal_state(provisional, state), state

    lower_atmosphere, lower_state = state_at(atmosphere.temperature * (1.0 - step))
    upper_atmosphere, upper_state = state_at(atmosphere.temperature * (1.0 + step))
    lower_enthalpy = _bulk_metal_specific_enthalpy(
        lower_atmosphere, lower_state, atomic_database
    )
    upper_enthalpy = _bulk_metal_specific_enthalpy(
        upper_atmosphere, upper_state, atomic_database
    )
    specific_heat = (
        (upper_enthalpy - lower_enthalpy)
        / (2.0 * step * atmosphere.temperature)
    )
    expansion = -(
        np.log(np.asarray(upper_state.total_mass_density))
        - np.log(np.asarray(lower_state.total_mass_density))
    ) / np.log((1.0 + step) / (1.0 - step))
    adiabatic_gradient = (
        atmosphere.gas_pressure * expansion
        / (
            atmosphere.mass_density
            * atmosphere.temperature
            * specific_heat
        )
    )
    return BulkMetalThermodynamics(
        np.asarray(specific_heat),
        np.asarray(expansion),
        np.asarray(adiabatic_gradient),
    )


def gray_d6_atmosphere(
    effective_temperature: float,
    logg: float,
    atomic_database: AtomicDatabase,
    log_number_abundance: Mapping[str, float],
    *,
    reference_element: str = "C",
    n_depth: int = 60,
    tau_min: float = 1.0e-8,
    tau_max: float = 1.0e3,
    seed_rosseland_opacity: float = 0.1,
) -> Atmosphere:
    """Construct an Eddington-gray seed with a self-consistent pure-metal EOS."""

    if not np.isfinite(effective_temperature) or effective_temperature <= 0.0:
        raise ValueError("effective_temperature must be finite and positive")
    if not np.isfinite(logg):
        raise ValueError("logg must be finite")
    if n_depth < 3 or not 0.0 < tau_min < tau_max:
        raise ValueError("require n_depth >= 3 and 0 < tau_min < tau_max")
    if not np.isfinite(seed_rosseland_opacity) or seed_rosseland_opacity <= 0.0:
        raise ValueError("seed_rosseland_opacity must be finite and positive")

    tau = np.geomspace(tau_min, tau_max, n_depth)
    temperature = effective_temperature * (0.75 * (tau + 2.0 / 3.0)) ** 0.25
    column_mass = tau / seed_rosseland_opacity
    pressure = 10.0**logg * column_mass
    zeros = np.zeros_like(tau)
    provisional = Atmosphere(
        float(effective_temperature),
        float(logg),
        tau,
        column_mass,
        temperature,
        pressure,
        np.ones_like(tau),
        zeros,
        zeros,
        np.ones_like(tau),
        {
            "model": "eddington-gray-seed",
            "composition": "hydrogen-helium-free-bulk-metals",
            "seed_rosseland_opacity_cm2_g": float(seed_rosseland_opacity),
        },
    )
    state = bulk_metal_lte_state(
        provisional,
        atomic_database,
        log_number_abundance,
        reference_element=reference_element,
    )
    return atmosphere_with_bulk_metal_state(provisional, state)


def _metal_opacity(
    atmosphere: Atmosphere,
    wavelength: FloatArray,
    atomic_database: AtomicDatabase,
    state: MetalLTEState,
    photoionization_database: VernerPhotoionizationDatabase,
    *,
    minimum_oscillator_strength: float,
    maximum_lines: int | None,
    include_lines: bool,
    line_transition_keys: Iterable[tuple[str, int, int, int]] | None = None,
    microturbulent_velocity_kms: float = 0.0,
    include_classical_electron_stark: bool = True,
    classical_electron_stark_scale: float = 1.0,
    classical_electron_stark_scale_by_ion: Mapping[tuple[str, int], float] | None = None,
    include_oxygen_i_series_stark: bool = False,
    oxygen_i_series_stark_minimum_effective_n: float | None = None,
    include_oxygen_i_quasistatic_microfields: bool = False,
    include_linear_stark_quasistatic: bool = False,
    linear_stark_profile: str = "manifold",
    include_rydberg_dissolution: bool = False,
    rydberg_dissolution_cutoff_probability: float | None = None,
    include_metal_series_pseudocontinuum: bool = False,
    metal_series_pseudocontinuum_method: str = "topbase",
    metal_series_pseudocontinuum_elements: Iterable[str] = ("O", "Mg"),
    rydberg_correlated_microfields: bool = True,
    rydberg_neutral_perturbers: bool = False,
    include_negative_ion_continuum: bool = False,
    topbase_photoionization_database: TOPbasePhotoionizationDatabase | None = None,
    profile_edge_optical_depth: float | None = None,
    profile_edge_optical_depth_elements: Iterable[str] | None = None,
    profile_edge_optical_depth_ions: Iterable[tuple[str, int]] | None = None,
    profile_support_maximum_rosseland_optical_depth: float = 2.0,
    profile_support_maximum_half_window_angstrom: float = 100.0,
) -> tuple[FloatArray, FloatArray]:
    absorption = metal_bound_free_mass_absorption_coefficient(
        atmosphere,
        wavelength,
        atomic_database,
        state,
        photoionization_database,
        excluded_ions=(
            () if topbase_photoionization_database is None
            else topbase_photoionization_database.ion_stages
        ),
    )
    if topbase_photoionization_database is not None:
        absorption += topbase_bound_free_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            state,
            topbase_photoionization_database,
        )
        if (
            include_metal_series_pseudocontinuum
            and metal_series_pseudocontinuum_method == "topbase"
        ):
            absorption += (
                topbase_dissolved_level_pseudocontinuum_mass_absorption_coefficient(
                    atmosphere,
                    wavelength,
                    state,
                    topbase_photoionization_database,
                    elements=metal_series_pseudocontinuum_elements,
                    correlated_microfields=rydberg_correlated_microfields,
                    continuum_cutoff_probability=(
                        rydberg_dissolution_cutoff_probability
                    ),
                )
            )
        elif (
            include_metal_series_pseudocontinuum
            and metal_series_pseudocontinuum_method == "line-strength"
        ):
            absorption += oxygen_i_dissolved_series_mass_absorption_coefficient(
                atmosphere,
                wavelength,
                atomic_database,
                state,
                correlated_microfields=rydberg_correlated_microfields,
                continuum_cutoff_probability=(
                    rydberg_dissolution_cutoff_probability
                ),
            )
        elif include_metal_series_pseudocontinuum:
            raise ValueError(
                "metal_series_pseudocontinuum_method must be 'topbase' "
                "or 'line-strength'"
            )
    absorption += light_metal_free_free_mass_absorption_coefficient(
        atmosphere, wavelength, state
    )
    if include_negative_ion_continuum:
        absorption += cool_metal_negative_ion_mass_absorption_coefficient(
            atmosphere, wavelength, state
        )
    if include_lines:
        absorption += metal_line_mass_absorption_coefficient(
            atmosphere,
            wavelength,
            atomic_database,
            state,
            minimum_oscillator_strength=minimum_oscillator_strength,
            maximum_lines=maximum_lines,
            microturbulent_velocity_kms=microturbulent_velocity_kms,
            include_classical_electron_stark=include_classical_electron_stark,
            classical_electron_stark_scale=classical_electron_stark_scale,
            classical_electron_stark_scale_by_ion=(
                classical_electron_stark_scale_by_ion
            ),
            include_oxygen_i_series_stark=include_oxygen_i_series_stark,
            oxygen_i_series_stark_minimum_effective_n=(
                oxygen_i_series_stark_minimum_effective_n
            ),
            include_oxygen_i_quasistatic_microfields=(
                include_oxygen_i_quasistatic_microfields
            ),
            include_linear_stark_quasistatic=include_linear_stark_quasistatic,
            linear_stark_profile=linear_stark_profile,
            include_rydberg_dissolution=include_rydberg_dissolution,
            rydberg_dissolution_cutoff_probability=(
                rydberg_dissolution_cutoff_probability
            ),
            # Dissolve lines only where the dissolved strength is returned
            # as the level-resolved pseudo-continuum below.
            rydberg_dissolution_elements=tuple(metal_series_pseudocontinuum_elements),
            rydberg_correlated_microfields=rydberg_correlated_microfields,
            rydberg_neutral_perturbers=rydberg_neutral_perturbers,
            profile_edge_optical_depth=profile_edge_optical_depth,
            profile_edge_optical_depth_elements=(
                profile_edge_optical_depth_elements
            ),
            profile_edge_optical_depth_ions=(
                profile_edge_optical_depth_ions
            ),
            profile_support_maximum_rosseland_optical_depth=(
                profile_support_maximum_rosseland_optical_depth
            ),
            profile_support_maximum_half_window_angstrom=(
                profile_support_maximum_half_window_angstrom
            ),
            transition_keys=line_transition_keys,
        )
    scattering = electron_scattering_mass_coefficient(atmosphere)[np.newaxis, :]
    return np.maximum(absorption, np.finfo(np.float64).tiny), scattering



def _structure_metal_lines(
    seed: Atmosphere,
    atomic_database: AtomicDatabase,
    abundances: Mapping[str, float],
    minimum_oscillator_strength: float,
    maximum_lines: int | None,
    *,
    reference_element: str = "C",
    flux_weighted_line_ranking: bool = True,
) -> list[tuple[AtomicIon, AtomicTransition]]:
    """Select the fixed, flux-ranked line set used by a D6 structure run."""

    ion_stage_weight = None
    if flux_weighted_line_ranking:
        selection_state = bulk_metal_lte_state(
            seed,
            atomic_database,
            abundances,
            reference_element=reference_element,
        )
        line_forming_depth = (
            (seed.rosseland_optical_depth >= 1.0e-4)
            & (seed.rosseland_optical_depth <= 30.0)
        )
        if not np.any(line_forming_depth):
            line_forming_depth = np.ones(seed.n_depth, dtype=bool)
        ion_stage_weight = {
            (element, charge): float(np.max(
                populations[charge, line_forming_depth]
                / np.maximum(
                    selection_state.element_number_density[element][
                        line_forming_depth
                    ],
                    np.finfo(np.float64).tiny,
                )
            ))
            for element, populations in selection_state.ion_number_density.items()
            for charge in range(populations.shape[0])
        }
    return selected_metal_lines(
        atomic_database,
        abundances,
        100.0,
        100_000.0,
        minimum_oscillator_strength,
        maximum_lines,
        seed.effective_temperature,
        ion_stage_weight,
        flux_weighted=flux_weighted_line_ranking,
    )


def _structure_wavelength_grid(
    seed: Atmosphere,
    atomic_database: AtomicDatabase,
    abundances: Mapping[str, float],
    n_continuum_wavelength: int,
    minimum_oscillator_strength: float,
    maximum_lines: int | None,
    *,
    reference_element: str = "C",
    flux_weighted_line_ranking: bool = True,
    microturbulent_velocity_kms: float = 0.0,
    selected_lines: Iterable[tuple[AtomicIon, AtomicTransition]] | None = None,
) -> FloatArray:
    continuum = np.geomspace(100.0, 100_000.0, n_continuum_wavelength)
    # Resolve each sampled line on both its thermal-width scale and the full
    # minimum support used by metal_line_mass_absorption_coefficient.  The
    # latter evaluates even a very narrow ordinary profile out to +/-0.25 A.
    # Sampling only +/-4 Doppler sigmas leaves the nonzero Lorentz wing at the
    # edge of a potentially broad trapezoid interval.  In dense Fe/Ni forests
    # that over-weights line opacity and biases the radiative-equilibrium
    # temperature upward.  The half-support point resolves the wing and the
    # point just outside the support explicitly restores the continuum.
    lines = list(selected_lines) if selected_lines is not None else (
        _structure_metal_lines(
            seed,
            atomic_database,
            abundances,
            minimum_oscillator_strength,
            maximum_lines,
            reference_element=reference_element,
            flux_weighted_line_ranking=flux_weighted_line_ranking,
        )
    )
    if not lines:
        return continuum
    centers = np.asarray([line.wavelength_vacuum_angstrom for _, line in lines])
    masses = np.asarray([ion.atomic_mass_u for ion, _ in lines])
    sigma = centers * np.sqrt(
        BOLTZMANN * seed.effective_temperature
        / (masses * ATOMIC_MASS_UNIT * LIGHT_SPEED**2)
        + (microturbulent_velocity_kms * 1.0e5 / LIGHT_SPEED) ** 2
    )
    thermal = (
        centers[:, np.newaxis]
        + sigma[:, np.newaxis] * np.asarray([-4.0, -2.0, 0.0, 2.0, 4.0])
    ).ravel()
    half_window = METAL_LINE_MINIMUM_HALF_WINDOW_ANGSTROM
    support = (
        centers[:, np.newaxis]
        + half_window
        * np.asarray([-1.004, -1.0, -0.5, 0.5, 1.0, 1.004])
    ).ravel()
    local = np.concatenate((thermal, support))
    return np.unique(np.concatenate((continuum, local[local > 0.0])))




def _bulk_atmosphere(
    effective_temperature: float,
    logg: float,
    rosseland_optical_depth: FloatArray,
    column_mass: FloatArray,
    temperature: FloatArray,
    atomic_database: AtomicDatabase,
    log_number_abundance: Mapping[str, float],
    *,
    reference_element: str,
    metadata: Mapping[str, object],
) -> tuple[Atmosphere, MetalLTEState]:
    """Return a hydrostatic bulk-metal atmosphere and its LTE state."""

    pressure = 10.0**logg * np.asarray(column_mass, dtype=np.float64)
    zeros = np.zeros_like(pressure)
    provisional = Atmosphere(
        float(effective_temperature),
        float(logg),
        np.asarray(rosseland_optical_depth, dtype=np.float64),
        np.asarray(column_mass, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
        pressure,
        np.ones_like(pressure),
        zeros,
        zeros,
        np.ones_like(pressure),
        dict(metadata),
    )
    state = bulk_metal_lte_state(
        provisional,
        atomic_database,
        log_number_abundance,
        reference_element=reference_element,
    )
    return atmosphere_with_bulk_metal_state(provisional, state), state


def continuum_d6_atmosphere(
    effective_temperature: float,
    logg: float,
    atomic_database: AtomicDatabase,
    photoionization_database: VernerPhotoionizationDatabase,
    log_number_abundance: Mapping[str, float],
    *,
    reference_element: str = "C",
    n_depth: int = 40,
    tau_min: float = 1.0e-8,
    tau_max: float = 1.0e2,
    hopf_constant: float = 2.0 / 3.0,
    n_wavelength: int = 900,
    max_iterations: int = 200,
    pressure_tolerance: float = 1.0e-8,
    topbase_photoionization_database: TOPbasePhotoionizationDatabase | None = None,
) -> Atmosphere:
    """Gray-temperature pure-metal seed on its physical Rosseland scale.

    This is the D6 counterpart of the continuum seeds used by the other LTE
    classes: the depth grid is uniform in continuum Rosseland optical depth
    between ``tau_min`` and ``tau_max`` and the pressure satisfies
    ``dP/dtau = g / kappa_R(T, P)`` for the bulk-metal EOS and the complete
    D6 continuum (level-resolved and Verner bound-free, free-free, electron
    scattering).  The hydrostatic equation is closed by a damped fixed-point
    iteration on the whole column; no temperature is clipped.  Bound-bound
    blanketing is left to the non-gray structure solution.
    """

    if not np.isfinite(effective_temperature) or effective_temperature <= 0.0:
        raise ValueError("effective_temperature must be finite and positive")
    if not np.isfinite(logg):
        raise ValueError("logg must be finite")
    if n_depth < 3 or not 0.0 < tau_min < tau_max:
        raise ValueError("require n_depth >= 3 and 0 < tau_min < tau_max")
    if n_wavelength < 100 or max_iterations < 1 or pressure_tolerance <= 0.0:
        raise ValueError("continuum-seed grids, iterations and tolerance must be positive")
    tau = np.geomspace(tau_min, tau_max, n_depth)
    temperature = effective_temperature * (0.75 * (tau + hopf_constant)) ** 0.25
    gravity = 10.0**logg
    log_tau = np.log(tau)
    wavelength = np.geomspace(100.0, 100_000.0, n_wavelength)
    pressure = gravity * tau / 0.1
    maximum_change = np.inf
    for iteration in range(1, max_iterations + 1):
        atmosphere, state = _bulk_atmosphere(
            effective_temperature, logg, tau, pressure / gravity, temperature,
            atomic_database, log_number_abundance,
            reference_element=reference_element, metadata={},
        )
        absorption, scattering = _metal_opacity(
            atmosphere,
            wavelength,
            atomic_database,
            state,
            photoionization_database,
            minimum_oscillator_strength=1.0,
            maximum_lines=0,
            include_lines=False,
            topbase_photoionization_database=topbase_photoionization_database,
        )
        rosseland = rosseland_mean_from_opacity_grid(
            wavelength, absorption + scattering, temperature
        )
        # P(tau_0) = g tau_0 / kappa_0 treats the unresolved layers above the
        # first point as having its opacity; the column below is integrated
        # in ln(tau), where the integrand g tau / kappa is smooth.
        integrand = gravity * tau / rosseland
        hydrostatic = np.empty_like(pressure)
        hydrostatic[0] = integrand[0]
        hydrostatic[1:] = hydrostatic[0] + np.cumsum(
            0.5 * (integrand[1:] + integrand[:-1]) * np.diff(log_tau)
        )
        maximum_change = float(np.max(np.abs(np.log(hydrostatic / pressure))))
        # Geometric damping converges for any local kappa ~ P^a with
        # -1 < a < 3, which covers bound-free and free-free absorption.
        pressure = np.sqrt(pressure * hydrostatic)
        if maximum_change < pressure_tolerance:
            break
    atmosphere, _ = _bulk_atmosphere(
        effective_temperature, logg, tau, pressure / gravity, temperature,
        atomic_database, log_number_abundance,
        reference_element=reference_element,
        metadata={
            "model": "eddington-gray-temperature/d6-continuum-hydrostatic",
            "composition": "hydrogen-helium-free-bulk-metals",
            "eos": "ideal-multi-element-saha-charge-pressure-closure",
            "hopf_constant": float(hopf_constant),
            "continuum_seed_iterations": int(iteration),
            "continuum_seed_converged": bool(maximum_change < pressure_tolerance),
            "continuum_seed_maximum_log_pressure_change": maximum_change,
            "continuum_seed_wavelength_points": int(n_wavelength),
        },
    )
    return atmosphere


def d6_structure_opacity_function(
    wavelength: FloatArray,
    atomic_database: AtomicDatabase,
    photoionization_database: VernerPhotoionizationDatabase,
    *,
    structure_line_transition_keys: tuple[tuple[str, int, int, int], ...],
    topbase_photoionization_database: TOPbasePhotoionizationDatabase | None,
    minimum_metal_oscillator_strength: float,
    microturbulent_velocity_kms: float = 0.0,
    include_rydberg_dissolution: bool = True,
    include_metal_series_pseudocontinuum: bool = True,
    metal_series_pseudocontinuum_elements: Iterable[str] = ("O", "Mg"),
    rydberg_correlated_microfields: bool = True,
    include_linear_stark_quasistatic: bool = False,
    linear_stark_profile: str = "manifold",
    include_oxygen_i_series_stark: bool = False,
    profile_edge_optical_depth: float | None = None,
) -> Callable[[Atmosphere, MetalLTEState], tuple[FloatArray, FloatArray]]:
    """Return the D6 structural absorption/scattering closure."""

    elements = tuple(metal_series_pseudocontinuum_elements)

    def opacity(
        atmosphere: Atmosphere, state: MetalLTEState
    ) -> tuple[FloatArray, FloatArray]:
        return _metal_opacity(
            atmosphere,
            wavelength,
            atomic_database,
            state,
            photoionization_database,
            minimum_oscillator_strength=minimum_metal_oscillator_strength,
            maximum_lines=len(structure_line_transition_keys),
            include_lines=bool(structure_line_transition_keys),
            line_transition_keys=structure_line_transition_keys,
            microturbulent_velocity_kms=microturbulent_velocity_kms,
            include_rydberg_dissolution=include_rydberg_dissolution,
            include_metal_series_pseudocontinuum=include_metal_series_pseudocontinuum,
            metal_series_pseudocontinuum_elements=elements,
            rydberg_correlated_microfields=rydberg_correlated_microfields,
            topbase_photoionization_database=topbase_photoionization_database,
            include_linear_stark_quasistatic=include_linear_stark_quasistatic,
            linear_stark_profile=linear_stark_profile,
            include_oxygen_i_series_stark=include_oxygen_i_series_stark,
            profile_edge_optical_depth=profile_edge_optical_depth,
        )

    return opacity


def radiative_equilibrium_d6_atmosphere(
    effective_temperature: float,
    logg: float,
    atomic_database: AtomicDatabase,
    photoionization_database: VernerPhotoionizationDatabase,
    log_number_abundance: Mapping[str, float],
    *,
    reference_element: str = "C",
    n_depth: int = 40,
    tau_min: float = 1.0e-8,
    tau_max: float = 1.0e2,
    max_iterations: int = 120,
    temperature_tolerance: float = 3.0e-4,
    flux_tolerance: float = 3.0e-3,
    enforce_local_energy_balance: bool = True,
    n_continuum_wavelength: int = 450,
    minimum_metal_oscillator_strength: float = 1.0e-4,
    maximum_metal_lines: int | None = 25_000,
    microturbulent_velocity_kms: float = 0.0,
    flux_weighted_structure_line_ranking: bool = True,
    include_rydberg_dissolution: bool = True,
    include_metal_series_pseudocontinuum: bool = True,
    metal_series_pseudocontinuum_elements: Iterable[str] = ("O", "Mg"),
    rydberg_correlated_microfields: bool = True,
    topbase_photoionization_database: TOPbasePhotoionizationDatabase | None = None,
    include_linear_stark_quasistatic: bool = False,
    linear_stark_profile: str = "manifold",
    include_oxygen_i_series_stark: bool = False,
    profile_edge_optical_depth: float | None = None,
    n_angle: int = 3,
    mixing_length_alpha: float | None = 1.25,
    initial_temperature: ArrayLike | None = None,
    initial_column_mass: ArrayLike | None = None,
    initial_rosseland_optical_depth: ArrayLike | None = None,
    resume_supplied_structure_in_formal_flux_phase: bool = False,
    iteration_callback: Callable[
        [int, Atmosphere, Mapping[str, object]], None
    ] | None = None,
) -> Atmosphere:
    """Solve a line-blanketed, hydrogen/helium-free LTE D6 atmosphere.

    D6 supplies only material closures -- the bulk-metal EOS and ML2
    thermodynamics and the D6 continuum plus fixed structural line opacity --
    to :func:`wd_spectra.adaptive_structure.solve_adaptive_lte_structure`, the
    same trust-region Newton/ML2 solver and equilibrium certification used by
    the DA, DB, DAB and DZ classes.  All selected lines enter the structure as
    true LTE absorption on a wavelength grid that resolves their profile
    support, following Hollands et al. (2025): most of a warm D6 star's flux
    emerges in the line-crowded ultraviolet.

    A supplied ``initial_column_mass`` and ``initial_temperature`` define a
    complete warm start on that grid.  ``initial_rosseland_optical_depth`` is
    only a label for depth-dependent diagnostics.  Without them the solver
    starts from :func:`continuum_d6_atmosphere`.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if temperature_tolerance <= 0.0 or flux_tolerance <= 0.0:
        raise ValueError("convergence tolerances must be positive")
    if n_continuum_wavelength < 80 or n_angle < 1:
        raise ValueError("wavelength and angle grids are too small")
    if not np.isfinite(microturbulent_velocity_kms) or microturbulent_velocity_kms < 0.0:
        raise ValueError("microturbulent_velocity_kms must be finite and non-negative")
    if mixing_length_alpha is not None and (
        not np.isfinite(mixing_length_alpha) or mixing_length_alpha <= 0.0
    ):
        raise ValueError("mixing_length_alpha must be positive or None")
    if (initial_column_mass is None) != (initial_temperature is None) and (
        initial_column_mass is not None
    ):
        raise ValueError("initial_column_mass requires initial_temperature")

    if initial_column_mass is not None:
        column_mass = np.asarray(initial_column_mass, dtype=np.float64)
        temperature = np.asarray(initial_temperature, dtype=np.float64)
        depth = (
            np.asarray(initial_rosseland_optical_depth, dtype=np.float64)
            if initial_rosseland_optical_depth is not None
            else np.geomspace(tau_min, tau_max, column_mass.size)
        )
        if (
            column_mass.ndim != 1
            or column_mass.shape != temperature.shape
            or column_mass.shape != depth.shape
            or column_mass.size < 3
            or np.any(~np.isfinite(column_mass))
            or np.any(np.diff(column_mass) <= 0.0)
            or np.any(column_mass <= 0.0)
            or np.any(~np.isfinite(temperature))
            or np.any(temperature <= 0.0)
        ):
            raise ValueError(
                "initial structure must be finite, positive, and increase inward"
            )
        seed, _ = _bulk_atmosphere(
            effective_temperature, logg, depth, column_mass, temperature,
            atomic_database, log_number_abundance,
            reference_element=reference_element,
            metadata={
                "model": "supplied-d6-structure",
                "composition": "hydrogen-helium-free-bulk-metals",
                "eos": "ideal-multi-element-saha-charge-pressure-closure",
            },
        )
    else:
        seed = continuum_d6_atmosphere(
            effective_temperature,
            logg,
            atomic_database,
            photoionization_database,
            log_number_abundance,
            reference_element=reference_element,
            n_depth=n_depth,
            tau_min=tau_min,
            tau_max=tau_max,
            topbase_photoionization_database=topbase_photoionization_database,
        )
        if initial_temperature is not None:
            supplied = np.asarray(initial_temperature, dtype=np.float64)
            if supplied.shape != seed.temperature.shape or np.any(supplied <= 0.0):
                raise ValueError("initial_temperature must be positive and match n_depth")
            seed, _ = _bulk_atmosphere(
                effective_temperature, logg, seed.rosseland_optical_depth,
                seed.column_mass, supplied, atomic_database,
                log_number_abundance, reference_element=reference_element,
                metadata=seed.metadata,
            )

    structure_lines = _structure_metal_lines(
        seed,
        atomic_database,
        log_number_abundance,
        minimum_metal_oscillator_strength,
        maximum_metal_lines,
        reference_element=reference_element,
        flux_weighted_line_ranking=flux_weighted_structure_line_ranking,
    )
    structure_line_transition_keys = tuple(
        (ion.element, ion.charge, line.lower_index, line.upper_index)
        for ion, line in structure_lines
    )
    wavelength = _structure_wavelength_grid(
        seed,
        atomic_database,
        log_number_abundance,
        n_continuum_wavelength,
        minimum_metal_oscillator_strength,
        maximum_metal_lines,
        reference_element=reference_element,
        flux_weighted_line_ranking=flux_weighted_structure_line_ranking,
        microturbulent_velocity_kms=microturbulent_velocity_kms,
        selected_lines=structure_lines,
    )
    structure_opacity = d6_structure_opacity_function(
        wavelength,
        atomic_database,
        photoionization_database,
        structure_line_transition_keys=structure_line_transition_keys,
        topbase_photoionization_database=topbase_photoionization_database,
        minimum_metal_oscillator_strength=minimum_metal_oscillator_strength,
        microturbulent_velocity_kms=microturbulent_velocity_kms,
        include_rydberg_dissolution=include_rydberg_dissolution,
        include_metal_series_pseudocontinuum=include_metal_series_pseudocontinuum,
        metal_series_pseudocontinuum_elements=metal_series_pseudocontinuum_elements,
        rydberg_correlated_microfields=rydberg_correlated_microfields,
        include_linear_stark_quasistatic=include_linear_stark_quasistatic,
        linear_stark_profile=linear_stark_profile,
        include_oxygen_i_series_stark=include_oxygen_i_series_stark,
        profile_edge_optical_depth=profile_edge_optical_depth,
    )

    # One opacity evaluation per distinct temperature.  The nonlinear driver
    # routinely asks for absorption, scattering and the Rosseland mean of the
    # same trial atmosphere; the metal-line forest dominates the cost.
    cache: dict[str, object] = {}

    def with_temperature(values: FloatArray) -> Atmosphere:
        current, state = _bulk_atmosphere(
            seed.effective_temperature, seed.logg, seed.rosseland_optical_depth,
            seed.column_mass, np.asarray(values, dtype=np.float64),
            atomic_database, log_number_abundance,
            reference_element=reference_element, metadata=seed.metadata,
        )
        cache.clear()
        cache.update(atmosphere=current, state=state)
        return current

    def opacity(current: Atmosphere) -> tuple[FloatArray, FloatArray]:
        if cache.get("atmosphere") is not current:
            state = bulk_metal_lte_state(
                current, atomic_database, log_number_abundance,
                reference_element=reference_element,
            )
            cache.clear()
            cache.update(atmosphere=current, state=state)
        if "absorption" not in cache:
            absorption, scattering = structure_opacity(current, cache["state"])
            cache.update(
                absorption=absorption,
                scattering=np.broadcast_to(scattering, absorption.shape),
            )
        return cache["absorption"], cache["scattering"]

    def true_absorption(current: Atmosphere) -> FloatArray:
        return opacity(current)[0]

    def scattering_opacity(current: Atmosphere) -> FloatArray:
        return opacity(current)[1]

    def rosseland_opacity(current: Atmosphere) -> FloatArray:
        absorption, scattering = opacity(current)
        return rosseland_mean_from_opacity_grid(
            wavelength, absorption + scattering, current.temperature
        )

    def thermodynamics(current: Atmosphere) -> BulkMetalThermodynamics:
        return bulk_metal_thermodynamics(
            current, atomic_database, log_number_abundance,
            reference_element=reference_element,
        )

    from .adaptive_structure import solve_adaptive_lte_structure

    return solve_adaptive_lte_structure(
        seed,
        wavelength,
        with_temperature=with_temperature,
        true_absorption=true_absorption,
        scattering_opacity=scattering_opacity,
        rosseland_opacity=rosseland_opacity,
        thermodynamics=thermodynamics,
        mixing_length_alpha=mixing_length_alpha,
        max_iterations=max_iterations,
        temperature_tolerance=temperature_tolerance,
        flux_tolerance=flux_tolerance,
        n_angle=n_angle,
        enforce_local_energy_balance=enforce_local_energy_balance,
        initial_temperature_was_supplied=initial_temperature is not None,
        resume_supplied_structure_in_formal_flux_phase=(
            initial_temperature is not None
            and resume_supplied_structure_in_formal_flux_phase
        ),
        iteration_callback=iteration_callback,
        metadata={
            **{
                key: value
                for key, value in seed.metadata.items()
                if key not in ("model",)
            },
            "composition": "hydrogen-helium-free-bulk-metals",
            "eos": "ideal-multi-element-saha-charge-pressure-closure",
            "reference_element": _canonical_element(reference_element),
            "metal_abundances": dict(log_number_abundance),
            "radiative_equilibrium_maximum_metal_lines": maximum_metal_lines,
            "radiative_equilibrium_selected_metal_lines": int(
                len(structure_line_transition_keys)
            ),
            "radiative_equilibrium_minimum_metal_oscillator_strength": float(
                minimum_metal_oscillator_strength
            ),
            "radiative_equilibrium_wavelength_points": int(wavelength.size),
            "microturbulent_velocity_kms": float(microturbulent_velocity_kms),
            "structure_line_ranking": (
                "abundance-ion-population-lower-level-Planck-flux"
                if flux_weighted_structure_line_ranking
                else "legacy-abundance-lower-level"
            ),
            "metal_rydberg_dissolution": bool(include_rydberg_dissolution),
            "metal_series_pseudocontinuum": bool(
                include_metal_series_pseudocontinuum
            ),
            "linear_stark_quasistatic_in_structure": bool(
                include_linear_stark_quasistatic
            ),
            "oxygen_i_series_stark_in_structure": bool(include_oxygen_i_series_stark),
            "line_profile_edge_optical_depth": profile_edge_optical_depth,
            "level_resolved_metal_bound_free": (
                topbase_photoionization_database.source
                if topbase_photoionization_database is not None
                else "disabled"
            ),
        },
    )


def synthesize_d6_spectrum(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    atomic_database: AtomicDatabase,
    photoionization_database: VernerPhotoionizationDatabase,
    log_number_abundance: Mapping[str, float],
    *,
    reference_element: str = "C",
    minimum_metal_oscillator_strength: float = 1.0e-6,
    maximum_metal_lines: int | None = 25_000,
    line_transition_keys: Iterable[tuple[str, int, int, int]] | None = None,
    microturbulent_velocity_kms: float = 0.0,
    include_lines: bool = True,
    n_angle: int = 4,
    backend: Backend = "auto",
    topbase_photoionization_database: TOPbasePhotoionizationDatabase | None = None,
    include_rydberg_dissolution: bool = True,
    include_metal_series_pseudocontinuum: bool = True,
    metal_series_pseudocontinuum_elements: Iterable[str] = ("O", "Mg"),
    rydberg_correlated_microfields: bool = True,
    include_nonideal_partitions: bool = False,
    **line_options: object,
) -> Spectrum:
    """Synthesize a hydrogen/helium-free LTE D6 spectrum.

    The coherent electron-scattering source is solved exactly with the
    shared checked source solver used by the other LTE classes, and the
    emergent flux uses the same piecewise-linear formal solution as DZ.
    ``line_options`` forwards explicitly named line-physics ablations (for
    example the diagnostic O I series-Stark modes) to the opacity routine.
    ``line_transition_keys`` freezes the transition list across wavelength
    windows and numerical refinements; it supersedes ``maximum_metal_lines``.
    The optional nonideal partitions are a fixed-structure EOS diagnostic,
    not a claim of a reconverged nonideal atmosphere.
    """

    wavelength = np.ascontiguousarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be finite, positive, and increasing")
    if not np.isfinite(microturbulent_velocity_kms) or microturbulent_velocity_kms < 0.0:
        raise ValueError("microturbulent_velocity_kms must be finite and non-negative")
    elements = tuple(metal_series_pseudocontinuum_elements)
    keys = None if line_transition_keys is None else tuple(sorted({
        (_canonical_element(element), int(charge), int(lower), int(upper))
        for element, charge, lower, upper in line_transition_keys
    }))
    state = bulk_metal_lte_state(
        atmosphere,
        atomic_database,
        log_number_abundance,
        reference_element=reference_element,
        include_nonideal_partitions=include_nonideal_partitions,
    )
    atmosphere = atmosphere_with_bulk_metal_state(atmosphere, state)
    absorption, scattering = _metal_opacity(
        atmosphere,
        wavelength,
        atomic_database,
        state,
        photoionization_database,
        minimum_oscillator_strength=minimum_metal_oscillator_strength,
        maximum_lines=maximum_metal_lines,
        line_transition_keys=keys,
        include_lines=include_lines,
        microturbulent_velocity_kms=microturbulent_velocity_kms,
        include_rydberg_dissolution=include_rydberg_dissolution,
        include_metal_series_pseudocontinuum=include_metal_series_pseudocontinuum,
        metal_series_pseudocontinuum_elements=elements,
        rydberg_correlated_microfields=rydberg_correlated_microfields,
        topbase_photoionization_database=topbase_photoionization_database,
        **line_options,
    )
    scattering = np.ascontiguousarray(np.broadcast_to(scattering, absorption.shape))
    optical_depth = optical_depth_from_mass_opacity(
        atmosphere.column_mass, absorption + scattering
    )
    planck = np.ascontiguousarray(
        planck_lambda_angstrom(
            wavelength[:, np.newaxis], atmosphere.temperature[np.newaxis, :]
        )
    )
    source, _, transfer_metadata = solve_spectrum_source(
        optical_depth, planck, absorption, scattering,
        wavelength=wavelength, n_angle=n_angle, discretization="formal-linear",
    )
    flux = emergent_flux(
        optical_depth, np.ascontiguousarray(source), n_angle=n_angle, backend=backend
    )
    return Spectrum(
        wavelength,
        flux,
        {
            **transfer_metadata,
            "transfer_discretization": "formal-linear",
            "wavelength_medium": "vacuum",
            "flux_convention": "surface F_lambda",
            "flux_unit": "erg s^-1 cm^-2 Angstrom^-1",
            "composition": "hydrogen-helium-free-bulk-metals",
            "reference_element": _canonical_element(reference_element),
            "metal_abundances": dict(state.log_number_abundance),
            "mass_fractions": dict(state.mass_fraction or {}),
            "metal_lines": atomic_database.source if include_lines else "disabled",
            "metal_rydberg_dissolution": (
                (
                    "Q-MHD correlated charged-microfield upper/lower survival ratio"
                    if rydberg_correlated_microfields
                    else "Holtsmark charged-microfield upper/lower survival ratio"
                )
                if include_lines and include_rydberg_dissolution else "disabled"
            ),
            "metal_series_pseudocontinuum": (
                "DAM level-resolved TOPbase continuation ("
                + ", ".join(_canonical_element(element) for element in elements)
                + ")"
                if include_metal_series_pseudocontinuum else "disabled"
            ),
            "metal_bound_free": (
                photoionization_database.source
                if topbase_photoionization_database is None
                else topbase_photoionization_database.source
                + "; Verner ground-state fallback for remaining ion stages"
            ),
            "metal_free_free": "charge-weighted ionic bremsstrahlung",
            "metal_electron_feedback": "fixed-pressure particle and charge closure",
            "transfer": "LTE absorption plus coherent-isotropic electron scattering",
            "minimum_metal_oscillator_strength": float(minimum_metal_oscillator_strength),
            "maximum_metal_lines": maximum_metal_lines,
            "explicit_line_transition_count": None if keys is None else len(keys),
            "explicit_line_transition_sha256": (
                None if keys is None
                else hashlib.sha256(json.dumps(keys).encode("ascii")).hexdigest()
            ),
            "nonideal_partition_functions": bool(include_nonideal_partitions),
            "microturbulent_velocity_kms": float(microturbulent_velocity_kms),
            "line_physics_options": {
                str(key): value for key, value in line_options.items()
            },
        },
    )
