"""Stable one-shot interfaces for the initial OpenWD release."""

from .common import (
    AtmosphereConvergenceWarning,
    ModelData,
    ModelResult,
    NumericalResolution,
    Quality,
    atmosphere_convergence_status,
    default_wavelength_grid,
    load_atmosphere_checkpoint,
    numerical_resolution,
    save_model_result,
)
from .stellar import (
    DAConfig,
    DABConfig,
    DBConfig,
    DZConfig,
    compute_da,
    compute_dab,
    compute_db,
    compute_dz,
)
from .selection import PhysicsSelection, PhysicsSelectionPolicy, select_physics
from .daz import DAZConfig, compute_daz
from .dq import DQConfig, compute_dq
from .automatic import ModelRun, run_model

from .hot import DOConfig, DAOConfig, compute_do, compute_dao

__all__ = [
    "DOConfig", "DAOConfig", "compute_do", "compute_dao",
    "AtmosphereConvergenceWarning",
    "DAConfig",
    "DAZConfig",
    "DQConfig",
    "DABConfig",
    "DBConfig",
    "DZConfig",
    "ModelData",
    "ModelResult",
    "ModelRun",
    "PhysicsSelection",
    "PhysicsSelectionPolicy",
    "select_physics",
    "run_model",
    "NumericalResolution",
    "Quality",
    "atmosphere_convergence_status",
    "compute_da",
    "compute_daz",
    "compute_dq",
    "compute_dab",
    "compute_db",
    "compute_dz",
    "default_wavelength_grid",
    "load_atmosphere_checkpoint",
    "numerical_resolution",
    "save_model_result",
]

from .pg1159 import PG1159Config, compute_pg1159
__all__ += ["PG1159Config", "compute_pg1159"]

from .d6 import D6Config, compute_d6
__all__ += ["D6Config", "compute_d6"]

from .dah import DAHConfig, compute_dah
__all__ += ["DAHConfig", "compute_dah"]
