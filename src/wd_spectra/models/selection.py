"""One-time, recorded physics selection from a provisional atmosphere.

This is a screening policy, not a new EOS or an equilibrium certificate.
Selection is frozen before relaxation. It never reacts to solver failure,
loads a checkpoint implicitly, or changes equations inside a Newton solve.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Mapping
import numpy as np

from .stellar import DAConfig, DBConfig, DABConfig, DZConfig
from .daz import DAZConfig
from .dq import DQConfig
from .hot import DOConfig, DAOConfig, validate_config
from .common import ModelData


@dataclass(frozen=True)
class PhysicsSelectionPolicy:
    """Explicit screening accuracy scales, not star-specific Teff cutoffs."""

    dense_density_fraction: float = 0.05
    molecular_nuclei_fraction: float = 1e-4
    molecular_electron_fraction: float = 1e-3

    def __post_init__(self):
        if any(not np.isfinite(v) or not 0 < v < 1 for v in asdict(self).values()):
            raise ValueError("physics screening tolerances must lie in (0,1)")


@dataclass(frozen=True)
class PhysicsSelection:
    workflow: str
    reason: str
    diagnostics: Mapping[str, object]
    tested_point: bool
    experimental: bool


def _helium_probe(config, data):
    from ..atmosphere import helium_continuum_atmosphere
    from ..dense_eos import read_helium_reos3_table
    from ..eos import hummer_mihalas_helium_lte

    a = helium_continuum_atmosphere(
        config.effective_temperature,
        config.logg,
        n_depth=20,
        rosseland_frequency_points=40,
    )
    x = np.log(a.rosseland_optical_depth)
    t = float(np.exp(np.interp(np.log(2 / 3), x, np.log(a.temperature))))
    p = float(np.exp(np.interp(np.log(2 / 3), x, np.log(a.gas_pressure))))
    state = hummer_mihalas_helium_lte(
        np.array([t]),
        np.array([p]),
        correlated_microfields=True,
        neutral_radius_scale=0.5,
    )
    density, _, inside = read_helium_reos3_table(data.helium_reos3).evaluate(p, t)
    ion = float(state.electron_density[0] / state.helium_nuclei_density[0])
    return dict(
        probe="continuum hydrostatic seed at Rosseland tau=2/3",
        temperature=t,
        gas_pressure=p,
        mass_density=float(state.mass_density[0]),
        electron_fraction=ion,
        reos_in_domain=bool(inside),
        density_fractional_change=(
            float(abs(density / state.mass_density[0] - 1)) if inside else None
        ),
    )


def _molecular_probe(config):
    from ..atmosphere import hydrogen_helium_continuum_atmosphere
    from .._mixed_molecules import molecular_hydrogen_helium_lte

    a = hydrogen_helium_continuum_atmosphere(
        config.effective_temperature,
        config.logg,
        config.log_hydrogen_to_helium,
        n_depth=20,
        rosseland_frequency_points=40,
    )
    state = molecular_hydrogen_helium_lte(
        a.temperature, a.gas_pressure, config.log_hydrogen_to_helium
    )
    region = (a.rosseland_optical_depth >= 0.01) & (a.rosseland_optical_depth <= 3.0)
    nuclei = (
        2
        * state.hydrogen_lte_state.molecular_hydrogen_density
        / state.hydrogen_nuclei_density
    )
    return dict(
        probe="continuum hydrostatic seed, 0.01<=Rosseland tau<=3",
        maximum_h2_nuclei_fraction=float(np.max(nuclei[region])),
        maximum_electron_fractional_change=float(
            np.max(abs(state.electron_density[region] / a.electron_density[region] - 1))
        ),
    )


from .pg1159 import PG1159Config, validate_config as validate_pg1159_config
from .d6 import D6Config
from .dah import DAHConfig


def select_physics(config, *, data=None, policy=PhysicsSelectionPolicy()):
    """Select a workflow from composition and local material diagnostics.

    The documented tested points are provenance only; they never select the
    equations. Provisional structures are not reused as converged solutions.
    Actual dense runs retain their stricter local table/trace-ion guards.
    """
    if not isinstance(config, (DAConfig, DAZConfig, DBConfig, DABConfig, DZConfig,
                               DQConfig, DOConfig, DAOConfig, PG1159Config, D6Config,
                               DAHConfig)):
        raise TypeError('expected a DAConfig, DAZConfig, DBConfig, DABConfig, DZConfig, DQConfig, DOConfig, DAOConfig, PG1159Config, D6Config or DAHConfig')
    data = ModelData.default() if data is None else data
    t, g = config.effective_temperature, config.logg
    if not np.isfinite(t) or t <= 0 or not np.isfinite(g):
        raise ValueError(
            "effective temperature and logg must be finite, with Teff positive"
        )
    if isinstance(config, DAHConfig):
        from .dah import _validate, dah_surface_cells
        from ..magnetic import WEAK_FIELD_MAXIMUM_MEGAGAUSS

        _validate(config)
        maximum = dah_surface_cells(config).maximum_field_megagauss
        regime = (
            "normal Zeeman triplets"
            if maximum <= WEAK_FIELD_MAXIMUM_MEGAGAUSS
            else "H2db Halpha-H12 components"
        )
        return PhysicsSelection(
            "dah",
            f"Magnetic pure-H LTE ({regime}; maximum visible field {maximum:.4g} MG), {config.balmer_profile} profiles on {config.atmosphere_structure} structure",
            {"maximum_visible_field_megagauss": maximum,
             "balmer_profile": config.balmer_profile,
             "atmosphere_structure": config.atmosphere_structure},
            False,
            False,
        )
    if isinstance(config, D6Config):
        return PhysicsSelection(
            "d6",
            "Hydrogen/helium-free bulk-metal LTE mixture with shared trust-region solver",
            {"bulk_metal_charge_pressure_closure": True, "trace_host": None},
            False,
            False,
        )
    if isinstance(config, PG1159Config):
        validate_pg1159_config(config)
        return PhysicsSelection("pg1159", "Bulk He/C/O NLTE with shared trust-region solver",
            {"nlte_charge_feedback": True, "trace_opacity_in_structure": False}, False, True)
    if isinstance(config, (DOConfig, DAOConfig)):
        validate_config(config)
        return PhysicsSelection(
            "dao" if isinstance(config, DAOConfig) else "do",
            "Restricted H/He NLTE with shared thermal Newton solver",
            {"nlte_charge_feedback": False, "metals": False}, False, True)
    if isinstance(config, DQConfig):
        return PhysicsSelection(
            "dq",
            "Refractive He/C/C2 cold-start protocol in an isolated worker",
            {
                "carbon_molecular_equilibrium": True,
                "scope": "trace-carbon, hydrogen-free, nonmagnetic; no DQp pressure distortion",
            },
            False,
            True,
        )
    if isinstance(config, DAConfig):
        return PhysicsSelection(
            "da",
            "Established hydrogen EOS/opacity policy",
            {},
            g == 8 and t in (3000, 4000, 5000, 20000),
            False,
        )
    if isinstance(config, DAZConfig):
        return PhysicsSelection(
            "daz",
            "Hydrogen host with coupled H/metal charge closure and opacity",
            {},
            False,
            False,
        )
    if isinstance(config, DZConfig):
        return PhysicsSelection(
            "dz",
            "Trace-metal helium mixture; never substitute the pure-He dense EOS",
            {},
            False,
            False,
        )
    if isinstance(config, DBConfig):
        d = _helium_probe(config, data)
        if not np.isfinite(d["electron_fraction"]) or d["electron_fraction"] < 0:
            raise ValueError(
                "Invalid ionization diagnostic; no physics choice can be certified by this screen"
            )
        if not d["reos_in_domain"] and d["electron_fraction"] < 1e-3:
            raise ValueError(
                "Neutral-helium screen is outside the REOS table; cannot assess dense physics, and no ideal-EOS substitute is selected"
            )
        if d["reos_in_domain"] and (
            not np.isfinite(d["density_fractional_change"])
            or d["density_fractional_change"] < 0
        ):
            raise ValueError(
                "Invalid dense-EOS diagnostic; no alternate physics is selected"
            )
        dense = (
            d["reos_in_domain"]
            and d["electron_fraction"] < 1e-3
            and d["density_fractional_change"] >= policy.dense_density_fraction
        )
        d["selection_policy"] = asdict(policy)
        return PhysicsSelection(
            "dense-db" if dense else "db",
            (
                "Neutral photosphere has significant nonideal bulk density correction"
                if dense
                else "Dense neutral trace-ion treatment is not indicated by the photospheric screen"
            ),
            d,
            g == 8 and t in ((5000, 8000) if dense else (10000, 22000)),
            bool(dense),
        )
    if isinstance(config, DABConfig):
        d = _molecular_probe(config)
        if any(
            not np.isfinite(d[key]) or d[key] < 0
            for key in (
                "maximum_h2_nuclei_fraction",
                "maximum_electron_fractional_change",
            )
        ):
            raise ValueError(
                "Invalid molecular diagnostic; no atomic substitute is selected"
            )
        molecular = (
            config.include_molecules
            or d["maximum_h2_nuclei_fraction"] >= policy.molecular_nuclei_fraction
            or d["maximum_electron_fractional_change"]
            >= policy.molecular_electron_fraction
        )
        d["selection_policy"] = asdict(policy)
        return PhysicsSelection(
            "molecular-dab" if molecular else "dab",
            (
                "Molecular species/electron donation require the mixed molecular workflow"
                if molecular
                else "Molecular chemistry is negligible in the provisional line-forming region"
            ),
            d,
            g == 8
            and config.log_hydrogen_to_helium == -2
            and t in ((7500, 8000, 9000, 10000) if molecular else (20000,)),
            bool(molecular),
        )
    raise AssertionError('unhandled supported model configuration')
