"""C2 data and coupled He/C ionization/dissociation for classical DQs.

This module is opt-in. No existing composition or solver imports it unless
carbon molecules are requested. Cross sections are per molecule, not per
carbon nucleus. The ideal C I + C I <-> C2 mass action law is solved *inside*
charge neutrality, so atomic lines cannot double-count molecular carbon.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
import numpy as np

from .constants import (
    BOLTZMANN,
    ELECTRON_MASS,
    HELIUM_MASS,
    HYDROGEN_MASS,
    LIGHT_SPEED,
    PLANCK,
)
from .metals import (
    MetalLTEState,
    _ion_fractions,
    dense_helium_ionization_potential_shift_ev,
)

EV = 1.602176634e-12
# Ground X(v=0,J=0) -> two C(3P0) atoms, consistent with the zero of
# the atomic/molecular partition sums. Borsovszky et al. (2021), PNAS
# 118, e2113315118, measure 50390.5 +/- 2.4 cm^-1. This replaces the
# older 6.297-eV estimate; it is laboratory input, not a fitted DQ knob.
C2_DISSOCIATION_WAVENUMBER_CM = 50390.5
C2_DISSOCIATION_ENERGY_EV = C2_DISSOCIATION_WAVENUMBER_CM * PLANCK * LIGHT_SPEED / EV
C2_DISSOCIATION_REFERENCE = "https://doi.org/10.1073/pnas.2113315118"
C2_EXOMOL_REFERENCE = (
    "Yurchenko et al. 2018, MNRAS 480, 3397; McKemmish et al. 2020, MNRAS 497, 1081"
)
C2_EXOMOL_ROOT_URL = "https://www.exomol.com/db/C2/12C2/8states"
C2_EXOMOL_SHA256 = {
    "12C2__8states.states.bz2": "d63229ec3639301f7682c4f1aaea15271c540eaad0bab094d243add57b71e858",
    "12C2__8states.trans.bz2": "f7ec89707578a36b418a6493339a288875fb2eecfa2fb8d991fb262e65681d46",
    "12C2__8states.pf": "f71f6f4b4be4fc8ce64dc11148d640737d01122d7197a33e0d37225968b108f2",
    "12C2__8states.def": "5259f176b70d90062bb93611fe0b528624392227be1fe8c9c7a82e64ae7a995e",
}


@dataclass(frozen=True)
class C2CrossSectionTable:
    wavelength_angstrom: np.ndarray
    temperature_K: np.ndarray
    cross_section: np.ndarray
    partition_temperature_K: np.ndarray
    partition_function: np.ndarray
    source: str = C2_EXOMOL_REFERENCE
    swan_cross_section: object = None
    swan_rotational_overlap_cross_section: object = None

    def __post_init__(self):
        for name in ("wavelength_angstrom", "temperature_K", "partition_temperature_K"):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if (
                a.ndim != 1
                or len(a) < 2
                or np.any(~np.isfinite(a))
                or np.any(a <= 0)
                or np.any(np.diff(a) <= 0)
            ):
                raise ValueError(f"{name} must be finite, positive and increasing")
            a.setflags(write=False)
            object.__setattr__(self, name, a)

        for name, shape in (
            ("cross_section", (len(self.wavelength_angstrom), len(self.temperature_K))),
            ("partition_function", self.partition_temperature_K.shape),
        ):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if (
                a.shape != shape
                or np.any(~np.isfinite(a))
                or np.any(a < 0)
                or (name == "partition_function" and np.any(a == 0))
            ):
                raise ValueError(f"invalid {name}")
            a.setflags(write=False)
            object.__setattr__(self, name, a)

        if self.swan_cross_section is not None:
            swan = np.array(self.swan_cross_section, dtype=float, copy=True)
            if (
                swan.shape != self.cross_section.shape
                or np.any(~np.isfinite(swan))
                or np.any(swan < 0)
                or np.any(swan > self.cross_section * (1 + 1e-12))
            ):
                raise ValueError(
                    "Swan component must be a nonnegative subset of total C2 opacity"
                )
            swan.setflags(write=False)
            object.__setattr__(self, "swan_cross_section", swan)
        if self.swan_rotational_overlap_cross_section is not None:
            overlap = np.array(
                self.swan_rotational_overlap_cross_section, dtype=float, copy=True
            )
            if (
                self.swan_cross_section is None
                or overlap.shape != self.cross_section.shape
                or np.any(~np.isfinite(overlap))
                or np.any(overlap < 0)
            ):
                raise ValueError("invalid Swan rotational-overlap component")
            overlap.setflags(write=False)
            object.__setattr__(self, "swan_rotational_overlap_cross_section", overlap)

    def molecular_partition_function(self, temperature):
        t = np.asarray(temperature, dtype=float)
        if (
            np.any(~np.isfinite(t))
            or np.any(t < self.partition_temperature_K[0])
            or np.any(t > self.partition_temperature_K[-1])
        ):
            raise ValueError(
                "C2 partition-function temperature outside supplied data; no endpoint clamping"
            )
        return np.exp(
            np.interp(
                np.log(t),
                np.log(self.partition_temperature_K),
                np.log(self.partition_function),
            )
        )

    def cross_section_for_wavelength_temperature(
        self, wavelength_angstrom, temperature_K
    ):
        w, t = np.asarray(wavelength_angstrom, dtype=float), np.asarray(
            temperature_K, dtype=float
        )
        if (
            w.ndim != 1
            or t.ndim != 1
            or np.any(~np.isfinite(w))
            or np.any(w <= 0)
            or np.any(~np.isfinite(t))
        ):
            raise ValueError("wavelength and temperature must be finite 1D arrays")
        if np.any(t < self.temperature_K[0]) or np.any(t > self.temperature_K[-1]):
            raise ValueError(
                "C2 cross-section temperature outside supplied data; no endpoint clamping"
            )
        index = np.clip(
            np.searchsorted(self.temperature_K, t) - 1, 0, len(self.temperature_K) - 2
        )
        f = (t - self.temperature_K[index]) / np.diff(self.temperature_K)[index]
        out = np.empty((len(w), len(t)))
        for j, i in enumerate(index):
            # Linear wavelength interpolation preserves zero intervals and
            # avoids geometric interpolation destroying line-bin integrals.
            lo = np.interp(
                w, self.wavelength_angstrom, self.cross_section[:, i], left=0, right=0
            )
            hi = np.interp(
                w,
                self.wavelength_angstrom,
                self.cross_section[:, i + 1],
                left=0,
                right=0,
            )
            out[:, j] = (1 - f[j]) * lo + f[j] * hi
        return out


def read_c2_cross_section_table(path):
    with np.load(Path(path), allow_pickle=False) as a:
        return C2CrossSectionTable(
            **{
                k: a[k]
                for k in (
                    "wavelength_angstrom",
                    "temperature_K",
                    "cross_section",
                    "partition_temperature_K",
                    "partition_function",
                )
            },
            source=str(a["source"].item()),
            swan_cross_section=(
                a["swan_cross_section"] if "swan_cross_section" in a else None
            ),
            swan_rotational_overlap_cross_section=(
                a["swan_rotational_overlap_cross_section"]
                if "swan_rotational_overlap_cross_section" in a
                else None
            ),
        )


def c2_dissociation_constant(
    temperature, atomic_partition_function, molecular_partition_function
):
    t, qa, qm = np.broadcast_arrays(
        temperature, atomic_partition_function, molecular_partition_function
    )
    if any(np.any(~np.isfinite(x)) or np.any(x <= 0) for x in (t, qa, qm)):
        raise ValueError(
            "partition functions and temperature must be finite and positive"
        )
    return np.exp(
        1.5 * np.log(2 * np.pi * 6 * HYDROGEN_MASS * BOLTZMANN * t / PLANCK**2)
        + 2 * np.log(qa)
        - np.log(qm)
        - C2_DISSOCIATION_ENERGY_EV * EV / (BOLTZMANN * t)
    )


def carbon_atomic_pool(total_carbon, neutral_fraction, dissociation_constant):
    """Return all unbound C atoms/ions, with N_C=N_atoms+2*n(C2)."""
    n, f, k = np.broadcast_arrays(total_carbon, neutral_fraction, dissociation_constant)
    if (
        any(np.any(~np.isfinite(x)) for x in (n, f, k))
        or np.any(n < 0)
        or np.any((f < 0) | (f > 1))
        or np.any(k <= 0)
    ):
        raise ValueError("invalid carbon mass-action inputs")
    return 2 * n / (1 + np.sqrt(1 + 8 * n * f * f / k))


def carbon_helium_lte_state(
    atmosphere,
    database,
    log_carbon_to_helium,
    table,
    *,
    nonideal_ionization=True,
    helium_reos3_table=None,
):
    """Charge and particle-pressure closure, retaining the host HM partitions.

    He/C atomic partition functions use the existing data. C2 formation is
    ideal; pressure-shifted/distorted DQp bands and a dense-mixture free energy
    are not supplied. This is restricted to trace carbon by the DQ interface.
    """
    he = atmosphere.helium_lte_state
    if he is None or not np.isfinite(log_carbon_to_helium):
        raise ValueError("finite C/He and a helium atmosphere are required")
    t = atmosphere.temperature
    host_pressure = np.array(atmosphere.gas_pressure, copy=True)
    if helium_reos3_table is not None:
        he = _reos_host_state(t, host_pressure, helium_reos3_table)
    stages = database.ion_stages("C")
    if len(stages) < 2:
        raise ValueError("carbon requires at least C I and C II")
    parts = {("C", ion.charge): ion.partition_function(t) for ion in stages}
    k = c2_dissociation_constant(
        t, parts[("C", 0)], table.molecular_partition_function(t)
    )
    a = 10.0**log_carbon_to_helium
    log_s = np.log(2) + 1.5 * np.log(
        2 * np.pi * ELECTRON_MASS * BOLTZMANN * t / PLANCK**2
    )
    helium_saha = np.stack(
        (
            log_s
            + np.log(
                he.singly_ionized_partition_function / he.neutral_partition_function
            )
            - 24.587389011 * EV / (BOLTZMANN * t),
            log_s
            - np.log(he.singly_ionized_partition_function)
            - 54.4177655282 * EV / (BOLTZMANN * t),
        )
    )
    particle = atmosphere.gas_pressure / (BOLTZMANN * t)
    nhe = he.helium_nuclei_density.copy()

    def populations(ne, nhe):
        ratios = []
        for lo, hi in zip(stages[:-1], stages[1:]):
            shift = (
                dense_helium_ionization_potential_shift_ev("C", nhe * HELIUM_MASS, t)
                if nonideal_ionization and lo.charge == 0
                else 0.0
            )
            ratios.append(
                log_s
                + np.log(parts[("C", hi.charge)] / parts[("C", lo.charge)])
                - np.maximum(lo.ionization_energy_ev + shift, 0.05)
                * EV
                / (BOLTZMANN * t)
                - np.log(ne)
            )
        f = _ion_fractions(np.stack(ratios))
        atoms = carbon_atomic_pool(a * nhe, f[0], k)
        c = atoms[None, :] * f
        c2 = c[0] ** 2 / k
        host = nhe[None, :] * _ion_fractions(helium_saha - np.log(ne)[None, :])
        charge = (
            host[1] + 2 * host[2] + np.sum(np.arange(len(stages))[:, None] * c, axis=0)
        )
        return c, c2, host, charge

    for _ in range(16):
        lower = np.full_like(t, -690.0)
        upper = np.log(np.maximum((2 + len(stages) * a) * nhe, 1.0))
        for _ in range(48):
            middle = 0.5 * (lower + upper)
            ne = np.exp(middle)
            need = populations(ne, nhe)[3]
            high = ne > need
            upper, lower = np.where(high, middle, upper), np.where(high, lower, middle)
        ne = np.exp(0.5 * (lower + upper))
        c, c2, host, charge = populations(ne, nhe)
        if helium_reos3_table is None:
            calculated = nhe + np.sum(c, axis=0) + c2 + ne
            pressure_error = np.max(np.abs(calculated / particle - 1))
        else:
            # REOS already contains the electrons of the isolated He host.
            # Add only the *change* in electron pressure from charge coupling,
            # plus the translational pressure of free C species and C2.
            impurity_pressure = (
                (np.sum(c, axis=0) + c2 + ne - he.electron_density) * BOLTZMANN * t
            )
            pressure_error = np.max(
                np.abs(
                    (host_pressure + impurity_pressure) / atmosphere.gas_pressure - 1
                )
            )
        if pressure_error < 1e-10:
            break
        if helium_reos3_table is None:
            nhe *= particle / calculated
        else:
            host_pressure = atmosphere.gas_pressure - impurity_pressure
            if np.any(host_pressure <= 0):
                raise ValueError(
                    "trace-carbon pressure exceeds the available gas pressure"
                )
            he = _reos_host_state(t, host_pressure, helium_reos3_table)
            nhe = he.helium_nuclei_density.copy()
            helium_saha = np.stack(
                (
                    log_s
                    + np.log(
                        he.singly_ionized_partition_function
                        / he.neutral_partition_function
                    )
                    - 24.587389011 * EV / (BOLTZMANN * t),
                    log_s
                    - np.log(he.singly_ionized_partition_function)
                    - 54.4177655282 * EV / (BOLTZMANN * t),
                )
            )
    else:
        raise ValueError("He/C/C2 particle-pressure closure failed")
    charge_error = float(np.max(np.abs(ne / np.maximum(charge, 1e-300) - 1)))
    if charge_error > 1e-9:
        raise ValueError("He/C/C2 charge neutrality failed")
    rho = HELIUM_MASS * nhe + 12 * HYDROGEN_MASS * a * nhe
    helium = replace(
        he,
        mass_density=HELIUM_MASS * nhe,
        helium_nuclei_density=nhe,
        neutral_he_density=host[0],
        singly_ionized_he_density=host[1],
        doubly_ionized_he_density=host[2],
        electron_density=ne,
        mean_ion_charge=(host[1] + 2 * host[2]) / nhe,
        neutral_level_population_density=he.neutral_level_population_density
        * (host[0] / np.maximum(he.neutral_he_density, 1e-300))[:, None],
    )
    metal = MetalLTEState(
        "He",
        MappingProxyType({"C": log_carbon_to_helium}),
        MappingProxyType({"C": a * nhe}),
        MappingProxyType({"C": c}),
        MappingProxyType(parts),
        ne,
        np.sum(np.arange(len(stages))[:, None] * c, axis=0),
        host_ion_number_density=host,
        nonideal_ionization=nonideal_ionization,
        total_mass_density=rho,
    )
    result = replace(
        atmosphere,
        mass_density=rho,
        electron_density=ne,
        helium_lte_state=helium,
        metadata={
            **atmosphere.metadata,
            "dq_charge_error": charge_error,
            "dq_pressure_error": float(pressure_error),
            "dq_maximum_carbon_molecular_fraction": float(np.max(2 * c2 / (a * nhe))),
            **(
                dict(
                    dq_helium_host_pressure=host_pressure.tolist(),
                    dq_helium_host_electron_density=he.electron_density.tolist(),
                    dq_bulk_eos="He-REOS.3 host + ideal trace-carbon pressure; not a mixture free energy",
                )
                if helium_reos3_table is not None
                else {}
            ),
        },
    )
    return result, metal, c2


def _reos_host_state(temperature, pressure, table):
    """Strict DQ experiment: no silent dilute-EOS substitute or extrapolation."""
    from .eos import hummer_mihalas_helium_lte_at_nuclei_density

    rho, _, inside = table.evaluate(pressure, temperature)
    if not np.all(inside):
        raise ValueError("DQ He-REOS.3 host state outside supplied EOS domain")
    return hummer_mihalas_helium_lte_at_nuclei_density(
        temperature,
        rho / HELIUM_MASS,
        correlated_microfields=True,
        neutral_radius_scale=0.5,
    )


def c2_band_mass_absorption_coefficient(
    atmosphere, wavelength_angstrom, metal_state, table
):
    """Compatibility diagnostic for explicitly supplied atomic-only states.

    DQ generation uses carbon_helium_lte_state instead; this helper only
    supports the pre-existing low-level fixed-state C2 spectrum keyword.
    """
    if (
        metal_state.reference_species != "He"
        or "C" not in metal_state.ion_number_density
    ):
        raise ValueError("C2 opacity requires helium and carbon")
    k = c2_dissociation_constant(
        atmosphere.temperature,
        metal_state.partition_function[("C", 0)],
        table.molecular_partition_function(atmosphere.temperature),
    )
    pool = metal_state.ion_number_density["C"][0]
    atom = carbon_atomic_pool(pool, np.ones_like(pool), k)
    return (
        table.cross_section_for_wavelength_temperature(
            wavelength_angstrom, atmosphere.temperature
        )
        * (atom**2 / k / atmosphere.mass_density)[None, :]
    )
