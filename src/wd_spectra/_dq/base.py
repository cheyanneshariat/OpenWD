"""Classical helium-dominated DQs: a material adapter, not another solver.

All DQ-specific operations live here and in carbon_molecular.py. The central
Newton/ML2/transfer implementation is used unchanged through its callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import logging
from pathlib import Path
import warnings
import numpy as np

from ..adaptive_structure import solve_adaptive_lte_structure
from .._domain import solve_with_screened_boundary
from .._rosseland import rosseland_mean_from_opacity_grid
from ..atmosphere import (
    Atmosphere,
    helium_continuum_atmosphere,
    _metal_line_opacity_sampling_grid,
)
from ..carbon_molecular import (
    read_c2_cross_section_table,
    carbon_helium_lte_state,
    C2_DISSOCIATION_ENERGY_EV,
    C2_DISSOCIATION_REFERENCE,
)
from ..c2_profiles import (
    C2ImpactProfiles,
    C2RotationalProfiles,
    ROTATIONAL_PROFILE_DESCRIPTION,
    PROFILE_DESCRIPTION,
    EXOMOL_PROFILE_DESCRIPTION,
    swan_density_shift_wavenumber,
)
from ..eos import hummer_mihalas_helium_lte, hummer_mihalas_helium_thermodynamics
from ..dense_eos import read_helium_reos3_table
from ..dense_helium_continuum import DenseHeliumCorrectionTable
from ..helium import (
    helium_continuum_mass_absorption_coefficient,
    helium_i_line_mass_absorption_coefficient,
    helium_i_resonance_line_mass_absorption_coefficient,
    helium_ii_line_mass_absorption_coefficient,
    helium_rayleigh_scattering_mass_coefficient,
)
from ..metals import (
    read_stout_atomic_database,
    read_verner_photoionization_database,
    metal_bound_free_mass_absorption_coefficient,
    metal_line_mass_absorption_coefficient,
    selected_metal_lines,
)
from ..opacity import (
    electron_scattering_mass_coefficient,
    optical_depth_from_mass_opacity,
)
from ..radiative_transfer import emergent_flux
from ..spectrum import Spectrum, planck_lambda_angstrom, solve_spectrum_source
from ..models.common import (
    ModelData,
    ModelResult,
    numerical_resolution,
    validate_wavelength,
    model_request_fingerprint,
    atmosphere_with_model_request_fingerprint,
    fixed_synthesis_atmosphere,
    warn_if_atmosphere_not_converged,
)
from ..models.stellar import _helium_tables, _metal_structure_line_budget


class DQApproximationWarning(RuntimeWarning):
    """DQ atmosphere convergence is not a qualification of its band profiles."""


def _complete_stationarity(solve_step, seed, max_iterations, iteration_callback):
    """Finish an early small-*accepted*-step exit within this one calculation.

    A small damped step is not necessarily a small unrestricted correction.
    Rebuild the shared solver's Jacobian only when all interior physical gates
    pass and stationarity alone prevents certification. No stored/neighboring
    model is read, no physics changes and no tolerance is relaxed. Progress
    must strictly reduce the measured correction; the original iteration
    budget bounds all additional work. Failed attempts remain in metadata.
    """
    used, segments = 0, []
    previous = np.inf
    while True:

        def report(i, a, d):
            if iteration_callback is not None:
                iteration_callback(
                    used + i, a, {**d, "dq_stationarity_pass": len(segments)}
                )

        result = solve_step(seed, max_iterations - used, bool(segments), report)
        meta = result.metadata
        count = int(meta.get("radiative_equilibrium_iterations", 0))
        used += count
        cert = meta.get("equilibrium_certificate", {})
        segments.append(dict(iterations=count, certificate=cert))
        failures = set(cert.get("failures", []))
        check = cert.get("checks", {}).get("temperature_stationarity", {})
        correction = check.get("value")
        if (
            not failures
            or not failures <= {"temperature_stationarity", "boundary_screening"}
            or "temperature_stationarity" not in failures
            or not check.get("measured")
            or correction is None
            or not np.isfinite(correction)
            or correction >= previous
            or count < 1
            or used >= max_iterations
        ):
            break
        previous = correction
        logging.getLogger(__name__).info(
            "DQ same-calculation stationarity completion: unrestricted correction %.6g; "
            "fresh Jacobian, unchanged physics and tolerance, %d iterations remain",
            correction,
            max_iterations - used,
        )
        seed = result
    return replace(
        result,
        metadata={
            **result.metadata,
            "radiative_equilibrium_iterations": used,
            "dq_stationarity_segments": segments,
            "dq_stationarity_completion_passes": len(segments) - 1,
        },
    )


def warn_dq_approximation(profile="rotational_overlap"):
    description = (
        "C2 Swan bands use a continuous rotational-overlap approximation, "
        "not measured dense-fluid molecular profiles. "
        if profile == "rotational_overlap"
        else (
            "C2 profiles use approximate neutral-He impact broadening, not measured "
            "C2-He coefficients or dense-fluid profiles. "
            if profile != "line_bin"
            else "The diagnostic line-bin molecular opacity has resolution-dependent line widths. "
        )
    )
    warnings.warn(
        "Preliminary DQ module: "
        + description
        + "Atmosphere convergence does not certify spectral accuracy; "
        "use these spectra for development/exploration, not abundance fitting.",
        DQApproximationWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class DQConfig:
    """Trace-carbon He atmosphere; carbon abundance is log10 N(C)/N(He).

    Initial module: nonmagnetic, hydrogen-free classical DQ. Hot carbon-
    dominated DQs, CH, DQp pressure distortions and dense-mixture thermodynamics
    are not qualified. C2 data must be explicitly built/supplied; no download
    or alternate-physics retry occurs during spectrum generation.
    """

    effective_temperature: float = 8000.0
    logg: float = 8.0
    log_carbon_to_helium: float = -5.0
    quality: str = "standard"
    c2_table_path: str | None = None
    mixing_length_alpha: float = 1.25
    nonideal_carbon_ionization: bool = True
    molecular_structure_resolving_power: float = 1500.0
    # Smooth band opacity before transfer; no post-synthesis smoothing.
    molecular_line_profile: str = "rotational_overlap"
    swan_pressure_shift: str = "none"
    helium_eos: str = "ideal"
    # Research constitutive data, not an enabled production default. Its
    # bounded wavelength domain is also used for structure convergence.
    helium_dense_continuum_path: str | None = None


class DQMaterial:
    """EOS, opacity and convection inputs owned exclusively by DQ."""

    _ATOMIC_WAVELENGTH_BOUNDS_ANGSTROM = (100.0, 100_000.0)
    _ATOMIC_MINIMUM_OSCILLATOR_STRENGTH = 1.0e-4

    def __init__(self, config, data, table):
        self.config, self.table = config, table
        if config.helium_dense_continuum_path and config.helium_eos != "reos3":
            raise ValueError("dense DQ continuum experiment requires helium_eos='reos3'")
        self.dense_continuum = (
            DenseHeliumCorrectionTable(config.helium_dense_continuum_path)
            if config.helium_dense_continuum_path else None
        )
        self.helium_reos3 = (
            read_helium_reos3_table(data.helium_reos3)
            if config.helium_eos == "reos3"
            else None
        )
        self.atomic = read_stout_atomic_database(
            data.stout, elements=("C",), maximum_charge=3
        )
        self.photo = read_verner_photoionization_database(
            data.verner_photoionization, elements=("C",), maximum_charge=3
        )
        self.he_i, self.he_ii = _helium_tables(data)
        self.cached = None
        # Declared once from a fixed physical wavelength domain, then reused
        # by both the structure and formal spectrum.  In particular, the
        # requested output interval must never select a different set of
        # carbon lines.
        self.atomic_transitions = None
        self.atomic_transition_lines = None
        # Backward-compatible diagnostic name used by existing research
        # scripts.  It refers to the same immutable declaration.
        self.structure_transitions = None
        self.atomic_transition_policy = None
        self.c2_profiles = (
            C2RotationalProfiles(table)
            if config.molecular_line_profile == "rotational_overlap"
            else (
                C2ImpactProfiles(
                    table,
                    prescription=(
                        "exomol_default"
                        if config.molecular_line_profile == "exomol_default"
                        else "synspec_standard"
                    ),
                )
                if config.molecular_line_profile != "line_bin"
                else None
            )
        )
        self.c2_swan_profiles = (
            (
                C2RotationalProfiles(table, component="swan")
                if config.molecular_line_profile == "rotational_overlap"
                else C2ImpactProfiles(
                    table, component="swan", prescription=self.c2_profiles.prescription
                )
            )
            if config.swan_pressure_shift == "blouin2019"
            else None
        )

    def chemistry(self, current):
        if self.cached is not None and self.cached[0] is current:
            return self.cached
        answer = carbon_helium_lte_state(
            current,
            self.atomic,
            self.config.log_carbon_to_helium,
            self.table,
            nonideal_ionization=self.config.nonideal_carbon_ionization,
            helium_reos3_table=self.helium_reos3,
        )
        self.cached = answer
        return answer

    def with_temperature(self, seed, temperature):
        if self.helium_reos3 is not None:
            # The coupled closure constructs its own strict REOS host state;
            # do not first compute and then discard an ideal-pressure state.
            return self.chemistry(replace(seed, temperature=np.asarray(temperature)))[0]
        he = hummer_mihalas_helium_lte(
            temperature,
            seed.gas_pressure,
            correlated_microfields=True,
            neutral_radius_scale=0.5,
        )
        a = replace(
            seed,
            temperature=np.asarray(temperature),
            helium_lte_state=he,
            mass_density=he.mass_density,
            electron_density=he.electron_density,
        )
        return self.chemistry(a)[0]

    def absorption(self, current, wavelength, *, include_c2=True, structure=False):
        a, carbon, c2 = self.chemistry(current)
        self._declare_atomic_transitions(a, carbon)
        dense = (
            self.dense_continuum.correction(
                wavelength, a.temperature, a.helium_lte_state.neutral_he_density
            ) if self.dense_continuum is not None else None
        )
        opacity = helium_continuum_mass_absorption_coefficient(
            a,
            wavelength,
            include_electron_scattering=False,
            include_rayleigh_scattering=False,
            helium_minus_correction=dense,
        )
        opacity += helium_i_line_mass_absorption_coefficient(
            a, wavelength, self.he_i, neutral_broadening="unsold"
        )
        opacity += helium_i_resonance_line_mass_absorption_coefficient(a, wavelength)
        opacity += helium_ii_line_mass_absorption_coefficient(
            a, wavelength, stark_table=self.he_ii
        )
        opacity += metal_bound_free_mass_absorption_coefficient(
            a, wavelength, self.atomic, carbon, self.photo
        )
        opacity += metal_line_mass_absorption_coefficient(
            a,
            wavelength,
            self.atomic,
            carbon,
            minimum_oscillator_strength=self._ATOMIC_MINIMUM_OSCILLATOR_STRENGTH,
            maximum_lines=None,
            transition_keys=self.atomic_transitions,
        )
        if include_c2:
            cross_section = (
                self.c2_profiles.cross_section(
                    wavelength, a.temperature, carbon.host_ion_number_density[0]
                )
                if self.c2_profiles is not None
                else self.table.cross_section_for_wavelength_temperature(
                    wavelength, a.temperature
                )
            )
            if self.c2_swan_profiles is not None:
                neutral = carbon.host_ion_number_density[0]
                unshifted = self.c2_swan_profiles.cross_section(
                    wavelength, a.temperature, neutral
                )
                shifted = self.c2_swan_profiles.cross_section(
                    wavelength,
                    a.temperature,
                    neutral,
                    wavenumber_shift=swan_density_shift_wavenumber(a.mass_density),
                )
                # Shift d-a Swan opacity alone, not Phillips/Ballik-Ramsay/
                # other C2 systems, and never the atomic carbon spectrum.
                cross_section = np.maximum(cross_section - unshifted, 0) + shifted
            opacity += cross_section * (c2 / a.mass_density)[None, :]
        return opacity

    def _declare_atomic_transitions(self, atmosphere, carbon=None):
        """Freeze a query-independent carbon-line declaration for this model.

        The finite line budget remains the public DQ structure budget, but its
        selection is made over the full declared radiative domain and only
        once.  Formal spectra therefore use precisely the lines whose opacity
        shaped the converged atmosphere, including distant line wings.
        """

        if self.atomic_transition_lines is not None:
            return self.atomic_transition_lines
        if carbon is None:
            carbon = self.chemistry(atmosphere)[1]
        weights = {
            ("C", i): float(np.max(p / carbon.element_number_density["C"]))
            for i, p in enumerate(carbon.ion_number_density["C"])
        }
        budget, budget_policy, metal_number_fraction = _metal_structure_line_budget(
            self.config.quality,
            {"C": self.config.log_carbon_to_helium},
            None,
        )
        lower, upper = self._ATOMIC_WAVELENGTH_BOUNDS_ANGSTROM
        lines = tuple(selected_metal_lines(
            self.atomic,
            {"C": self.config.log_carbon_to_helium},
            lower,
            upper,
            self._ATOMIC_MINIMUM_OSCILLATOR_STRENGTH,
            budget,
            self.config.effective_temperature,
            weights,
            flux_weighted=True,
        ))
        keys = tuple(
            (ion.element, ion.charge, line.lower_index, line.upper_index)
            for ion, line in lines
        )
        self.atomic_transition_lines = lines
        self.atomic_transitions = keys
        self.structure_transitions = keys
        self.atomic_transition_policy = {
            "wavelength_bounds_angstrom": [lower, upper],
            "minimum_oscillator_strength": self._ATOMIC_MINIMUM_OSCILLATOR_STRENGTH,
            "maximum_lines": budget,
            "budget_policy": budget_policy,
            "metal_number_fraction": metal_number_fraction,
            "selected_lines": len(keys),
            "query_independent": True,
            "shared_by_structure_and_synthesis": True,
        }
        return lines

    def scattering(self, current, wavelength):
        a = self.chemistry(current)[0]
        return electron_scattering_mass_coefficient(a)[
            None, :
        ] + helium_rayleigh_scattering_mass_coefficient(a, wavelength)

    def thermodynamics(self, current):
        # Same declared trace-species approximation as the DZ adapter; do
        # not silently add pure-He dense-fluid chemistry to a carbon mixture.
        if self.helium_reos3 is not None:
            current = self.chemistry(current)[0]
            pressure = np.asarray(current.metadata["dq_helium_host_pressure"])
            for temperature in (
                current.temperature * np.exp(-2e-4),
                current.temperature * np.exp(2e-4),
            ):
                if not np.all(self.helium_reos3.evaluate(pressure, temperature)[2]):
                    raise ValueError(
                        "DQ He-REOS.3 thermodynamic derivative outside supplied EOS domain"
                    )
        else:
            pressure = current.gas_pressure
        return hummer_mihalas_helium_thermodynamics(
            current.temperature,
            pressure,
            correlated_microfields=True,
            neutral_radius_scale=0.5,
            helium_reos3_table=self.helium_reos3,
        )

    def hydrostatic_seed(self, teff, logg, n_depth):
        """Construct the common target-parameter seed used by cold starts."""

        return helium_continuum_atmosphere(
            teff,
            logg,
            n_depth=n_depth,
            correlated_microfields=True,
            metal_database=self.atomic,
            metal_abundances={"C": self.config.log_carbon_to_helium},
            include_dense_helium_metal_ionization=(
                self.config.nonideal_carbon_ionization
            ),
        )

    def solve_adaptive_structure(self, seed, wavelength, **options):
        """Run one structure segment with this material's transfer policy.

        Subclasses may provide explicit per-solve radiation and convection
        callbacks through ``options``.  Keeping this dispatch on the material
        avoids process-wide replacement of the shared solver function.
        """

        return solve_adaptive_lte_structure(seed, wavelength, **options)

    def structure_grid(self, seed, n_continuum):
        c = self.config
        lo, hi = self.table.wavelength_angstrom[[0, -1]]
        molecular = np.geomspace(
            lo,
            hi,
            int(np.ceil(c.molecular_structure_resolving_power * np.log(hi / lo))) + 1,
        )
        lines = self._declare_atomic_transitions(seed)
        metal, _ = _metal_line_opacity_sampling_grid(
            np.array([line.wavelength_vacuum_angstrom for _, line in lines])
        )
        grid = np.unique(
            np.r_[
                np.geomspace(100, 100000, n_continuum),
                molecular,
                metal,
                np.arange(480, 700.1, 1.0),
            ]
        )
        if self.dense_continuum is not None:
            lower, upper = self.dense_continuum.wavelength_bounds
            grid = np.unique(np.r_[lower, grid[(grid >= lower) & (grid <= upper)], upper])
        return grid

    def solve(
        self,
        teff,
        logg,
        *,
        n_depth,
        n_continuum,
        max_iterations,
        n_angle,
        iteration_callback=None,
        initial_temperature=None,
        initial_column_mass=None,
        initial_gas_pressure=None,
        initial_rosseland_optical_depth=None,
        resume_supplied_structure_in_formal_flux_phase=False,
    ):
        if initial_temperature is None:
            seed = self.hydrostatic_seed(teff, logg, n_depth)
        else:
            he = hummer_mihalas_helium_lte(
                initial_temperature, initial_gas_pressure, correlated_microfields=True
            )
            seed = Atmosphere(
                teff,
                logg,
                initial_rosseland_optical_depth,
                initial_column_mass,
                initial_temperature,
                initial_gas_pressure,
                he.mass_density,
                np.zeros(n_depth),
                np.zeros(n_depth),
                he.electron_density,
                {},
                helium_lte_state=he,
            )
        seed = self.with_temperature(seed, seed.temperature)
        wave = self.structure_grid(seed, n_continuum)
        cached_absorption = [None, None]

        def absorption(current):
            value = self.absorption(current, wave, structure=True)
            cached_absorption[:] = [current, value]
            return value

        def rosseland(current):
            value = (
                cached_absorption[1]
                if cached_absorption[0] is current
                else absorption(current)
            )
            return rosseland_mean_from_opacity_grid(
                wave, value + self.scattering(current, wave), current.temperature
            )

        def solve_step(current_seed, remaining, polish, callback):
            # A lower-domain extension contains extrapolated, unsolved cells,
            # unlike a same-mesh stationarity polish. Initialize convection
            # on that new mesh before enforcing the exact flux equations.
            # This is part of the same cold calculation, not a physics change
            # or a continuation from an external atmosphere.
            return self.solve_adaptive_structure(
                current_seed,
                wave,
                with_temperature=lambda t: self.with_temperature(current_seed, t),
                true_absorption=absorption,
                scattering_opacity=lambda a: self.scattering(a, wave),
                rosseland_opacity=rosseland,
                thermodynamics=self.thermodynamics,
                mixing_length_alpha=self.config.mixing_length_alpha,
                max_iterations=remaining,
                temperature_tolerance=2e-4,
                flux_tolerance=2e-3,
                n_angle=n_angle,
                enforce_local_energy_balance=True,
                initial_temperature_was_supplied=polish
                or initial_temperature is not None,
                resume_supplied_structure_in_formal_flux_phase=polish,
                project_initial_convective_gradient=not polish,
                use_initial_bolometric_rescaling=not polish,
                iteration_callback=callback,
                metadata=dict(
                    composition="He/C/C2",
                    eos=(
                        "He-REOS.3 host + coupled trace-carbon pressure"
                        if self.helium_reos3 is not None
                        else "coupled-charge-particle-pressure-He-C-C2"
                    ),
                    dq_material_revision=7,
                    c2_dissociation_energy_ev=C2_DISSOCIATION_ENERGY_EV,
                    c2_dissociation_reference=C2_DISSOCIATION_REFERENCE,
                    dq_domain_initialization="recondition extrapolated cells; exact-flux stationarity polish",
                    carbon_abundance=self.config.log_carbon_to_helium,
                    carbon_molecules_in_structure=True,
                    carbon_molecular_opacity=self.table.source,
                    atomic_transition_policy=self.atomic_transition_policy,
                    molecular_line_profile=self.config.molecular_line_profile,
                    swan_pressure_shift=self.config.swan_pressure_shift,
                    dense_helium_continuum=(self.dense_continuum.metadata if self.dense_continuum else None),
                    dense_helium_continuum_sha256=(self.dense_continuum.sha256 if self.dense_continuum else None),
                    convection_thermodynamics=(
                        "trace-carbon approximation: He-REOS.3 host derivatives"
                        if self.helium_reos3 is not None
                        else "trace-carbon approximation: existing HM helium derivatives"
                    ),
                    helium_neutral_radius_scale=0.5,
                ),
            )

        return _complete_stationarity(
            solve_step, seed, max_iterations, iteration_callback
        )

    def spectrum(self, atmosphere, wavelength, n_angle, *, include_c2=True):
        a = self.chemistry(atmosphere)[0]
        absorption = self.absorption(a, wavelength, include_c2=include_c2)
        scattering = self.scattering(a, wavelength)
        tau = optical_depth_from_mass_opacity(a.column_mass, absorption + scattering)
        planck = planck_lambda_angstrom(wavelength[:, None], a.temperature[None, :])
        source, _, meta = solve_spectrum_source(
            tau,
            planck,
            absorption,
            scattering,
            wavelength=wavelength,
            n_angle=n_angle,
            discretization="formal-linear",
        )
        flux = emergent_flux(tau, source, n_angle=n_angle)
        return Spectrum(
            wavelength,
            flux,
            {
                **meta,
                "transfer_discretization": "formal-linear",
                "composition": "He/C/C2",
                "wavelength_medium": "vacuum",
                "flux_convention": "surface F_lambda",
                "flux_unit": "erg s^-1 cm^-2 Angstrom^-1",
                "molecular_line_profile": self.config.molecular_line_profile,
                "swan_pressure_shift": self.config.swan_pressure_shift,
                "dense_helium_continuum_sha256": (self.dense_continuum.sha256 if self.dense_continuum else None),
                "c2_opacity": (
                    self.table.source if include_c2 else "diagnostically omitted"
                ),
                "atomic_transition_policy": self.atomic_transition_policy,
            },
        )
