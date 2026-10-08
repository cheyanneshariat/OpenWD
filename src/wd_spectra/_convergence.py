"""Shared, explicit structure-grid certification; not full-physics validation.

Solver termination, equilibrium, boundary adequacy and resolution are distinct.
In particular an unchanged restart is not a measured stationary correction.
The absorption escape bound is sufficient, not necessary: a failed bound
means boundary screening is unverified, not proof of a spurious spectrum.
"""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np


def equilibrium_certificate(
    metadata: Mapping,
    *,
    flux_tolerance=3e-3,
    surface_flux_tolerance=None,
    photospheric_flux_tolerance=None,
    local_energy_tolerance=None,
    temperature_tolerance=3e-4,
    source_tolerance=1e-6,
    required_checks=None,
    profile=None,
) -> dict:
    """Evaluate finite recorded diagnostics with their actual normalizations."""
    if local_energy_tolerance is None:
        local_energy_tolerance = flux_tolerance
    if surface_flux_tolerance is None:
        surface_flux_tolerance = flux_tolerance
    if photospheric_flux_tolerance is None:
        photospheric_flux_tolerance = flux_tolerance
    for tolerance in (
        flux_tolerance,
        surface_flux_tolerance,
        photospheric_flux_tolerance,
        local_energy_tolerance,
        temperature_tolerance,
        source_tolerance,
    ):
        if (
            isinstance(tolerance, (bool, np.bool_))
            or not isinstance(tolerance, (float, int, np.floating, np.integer))
            or not np.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise ValueError("certification tolerances must be finite and positive")

    def check(value, tolerance, *, measured=True):
        valid = isinstance(
            value, (float, int, np.floating, np.integer)
        ) and not isinstance(value, (bool, np.bool_))
        finite = valid and bool(np.isfinite(value))
        return dict(
            value=float(value) if finite else None,
            tolerance=float(tolerance),
            passed=bool(finite and measured and 0 <= value < tolerance),
            measured=bool(measured and finite),
        )

    checks = {
        "all_depth_flux": check(
            metadata.get("maximum_all_depth_total_flux_residual"), flux_tolerance
        ),
        "surface_flux": check(
            abs(float(metadata["surface_flux_ratio"]) - 1.0)
            if isinstance(
                metadata.get("surface_flux_ratio"),
                (float, int, np.floating, np.integer),
            )
            and not isinstance(metadata.get("surface_flux_ratio"), (bool, np.bool_))
            else None,
            surface_flux_tolerance,
        ),
        "photospheric_flux": check(
            metadata.get("maximum_photospheric_total_flux_residual"),
            photospheric_flux_tolerance,
        ),
        "local_energy": check(
            metadata.get("maximum_relative_cell_energy_balance_residual"),
            local_energy_tolerance,
        ),
        "temperature_stationarity": check(
            metadata.get(
                "maximum_unrestricted_log_temperature_correction",
                metadata.get(
                    "radiative_equilibrium_maximum_log_temperature_correction"
                ),
            ),
            temperature_tolerance,
            measured=metadata.get("temperature_correction_measured") is True,
        ),
        "source_closure": check(
            metadata.get("electron_scattering_source_final_maximum_relative_residual"),
            source_tolerance,
        ),
        "boundary_screening": check(
            metadata.get("lower_boundary_absorption_escape_bound"), flux_tolerance
        ),
    }
    if required_checks is None:
        required_checks = (
            "all_depth_flux",
            "local_energy",
            "temperature_stationarity",
            "source_closure",
            "boundary_screening",
        )
    else:
        required_checks = tuple(required_checks)
        if (
            not required_checks
            or len(set(required_checks)) != len(required_checks)
            or any(name not in checks for name in required_checks)
        ):
            raise ValueError("required_checks must name distinct certificate checks")
    solver = metadata.get("radiative_equilibrium_solver_converged") is True
    failures = [name for name in required_checks if not checks[name]["passed"]]
    if not solver:
        failures.insert(0, "solver_completion")
    if profile is not None and profile != "pg1159-spectrum-gate-v1":
        raise ValueError("unknown equilibrium-certificate profile")
    return dict(
        schema=1 if profile is None else 2,
        **({} if profile is None else {"profile": profile}),
        scope="declared equations on the structure grid",
        verified=not failures,
        checks=checks,
        required_checks=required_checks,
        failures=failures,
        independent_grid_validation=False,
        full_physics_validation=False,
    )


def recorded_equilibrium_status(metadata: Mapping) -> str:
    """Fail closed on legacy/missing certification, preserving exploration."""
    if metadata.get("fixed_synthesis_request_verified") is False:
        return "unconverged"
    certificate = metadata.get("equilibrium_certificate")
    if isinstance(certificate, Mapping) and certificate.get("schema") in (1, 2):
        profile = certificate.get("profile")
        if certificate.get("schema") == 1:
            if profile is not None:
                return "unknown"
        elif profile != "pg1159-spectrum-gate-v1":
            return "unknown"
        # Re-evaluate the diagnostics: a copied True field is not authority.
        old_checks = certificate.get("checks", {})
        if not isinstance(old_checks, Mapping) or any(
            not isinstance(old_checks.get(name), Mapping)
            for name in ("all_depth_flux", "temperature_stationarity", "source_closure")
        ):
            return "unknown"
        tolerances = {
            "flux_tolerance": old_checks.get("all_depth_flux", {}).get(
                "tolerance", 3e-3
            ),
            "surface_flux_tolerance": old_checks.get("surface_flux", {}).get(
                "tolerance", 3e-3
            ),
            "photospheric_flux_tolerance": old_checks.get(
                "photospheric_flux", {}
            ).get("tolerance", 3e-3),
            "local_energy_tolerance": old_checks.get("local_energy", {}).get(
                "tolerance", 3e-3
            ),
            "temperature_tolerance": old_checks.get("temperature_stationarity", {}).get(
                "tolerance", 3e-4
            ),
            "source_tolerance": old_checks.get("source_closure", {}).get(
                "tolerance", 1e-6
            ),
        }
        required_checks = certificate.get("required_checks")
        if required_checks is not None and (
            not isinstance(required_checks, (list, tuple))
            or not required_checks
            or len(set(required_checks)) != len(required_checks)
            or any(name not in old_checks for name in required_checks)
        ):
            return "unknown"
        if any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not np.isfinite(v)
            or v <= 0
            for v in tolerances.values()
        ):
            return "unknown"
        verified = equilibrium_certificate(
            metadata,
            required_checks=required_checks,
            profile=profile,
            **tolerances,
        )["verified"]
        if not verified:
            return "unconverged"
        return "converged" if profile is None else "spectrum-qualified"
    if metadata.get("radiative_equilibrium_converged") is False:
        return "unconverged"
    return "unknown"
