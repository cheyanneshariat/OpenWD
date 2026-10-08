"""One checked coherent-scattering source law for all LTE spectra.

The established piecewise-linear formal solver and the explicit Feautrier
variants each solve their own scattering equations directly. An independent
prescribed-source solve verifies closure; repeating the algebra used to
construct S would not be a test. No flux normalization is applied.
"""

from __future__ import annotations

import numpy as np

from ._stable_feautrier import cancellation_safe_field


def solve_spectrum_source(
    optical_depth,
    planck,
    absorption,
    scattering,
    *,
    wavelength,
    n_angle,
    column_mass=None,
    discretization="optical-depth",
):
    """Return source, coupled field and independently measured diagnostics."""
    solver = cancellation_safe_field
    options = dict(n_angle=n_angle)
    if discretization == "formal-pchip":
        if column_mass is not None:
            raise ValueError("formal-pchip uses optical-depth coordinates")
        from ._monotone_formal import solve_cubic_source

        source, field, _, record = solve_cubic_source(
            optical_depth, planck, absorption, scattering,
            wavelength=wavelength, n_angle=n_angle,
        )
        return source, field, {
            "transfer_discretization": "formal-pchip",
            "scattering_source_solver": "monotone cubic defect correction / Newton",
            "source_iterations": record["iterations"],
            "source_converged": True,
            "maximum_relative_source_change": None,
            "independent_radiation_scaled_source_error": record[
                "independent_radiation_scaled_source_error"
            ],
            "source_closure_tolerance": 1e-10,
        }
    if column_mass is not None:
        from ._mass_feautrier import mass_field

        solver = mass_field
        options["column_mass"] = column_mass
        discretization = "column-mass"
    if discretization == "formal-linear":
        from ._linear_scattering import linear_scattering_source
        from .radiative_transfer import radiation_field

        source, field = linear_scattering_source(
            optical_depth, planck, absorption, scattering, **options
        )
        independent = radiation_field(optical_depth, source, n_angle=n_angle)
    else:
        if discretization not in ("optical-depth", "column-mass"):
            raise ValueError("unknown spectrum discretization")
        source, field = solver(optical_depth, planck, absorption, scattering, **options)
        total = absorption + scattering
        _, independent = solver(
            optical_depth, source, total, np.zeros_like(total), **options
        )
    total = absorption + scattering
    defect = (
        source - (absorption * planck + scattering * independent.mean_intensity) / total
    )
    weight = np.asarray(wavelength, dtype=float)[:, None]
    scale = max(float(np.max(weight * np.abs(source))), np.finfo(float).tiny)
    error = float(np.max(weight * np.abs(defect)) / scale)
    if not np.isfinite(error) or error > 1e-10:
        raise RuntimeError(f"Spectrum failed independent source closure: {error:.3g}")
    return (
        source,
        field,
        {
            "transfer_discretization": discretization,
            "scattering_source_solver": (
                "direct exact Lambda system"
                if discretization == "formal-linear"
                else "direct coupled Feautrier"
            ),
            "source_iterations": 1,
            "source_converged": True,
            "maximum_relative_source_change": None,
            "independent_radiation_scaled_source_error": error,
            "source_closure_tolerance": 1e-10,
        },
    )
