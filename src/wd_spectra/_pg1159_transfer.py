"""PG 1159 access to OpenWD's conservative mass-volume formal solver."""
import numpy as np
from .opacity import optical_depth_from_mass_opacity
from ._mass_feautrier import (
    InvalidRadiationFieldError,
    mass_emissivity_field,
    mass_field,
)


def transfer_field(
    atmosphere, coefficients, *, n_angle=3, check_source=True, wavelength_chunk_size=512
):
    """Solve scattering directly with the release mass-volume transfer solver."""
    a, eta, s = (
        np.asarray(x)
        for x in (
            coefficients.true_absorption,
            coefficients.thermal_emissivity,
            coefficients.scattering,
        )
    )
    shape = (len(coefficients.wavelength_angstrom), atmosphere.n_depth)
    if any(x.shape != shape for x in (a, eta, s)):
        raise ValueError("NLTE coefficients must be wavelength-by-depth arrays")
    if any(np.any(~np.isfinite(x)) for x in (a, eta, s)):
        raise InvalidRadiationFieldError("NLTE coefficients must be finite")
    if np.any(a + s <= 0) or np.any(a[:, -1] <= 0) or np.any(eta < 0) or np.any(s < 0):
        raise InvalidRadiationFieldError(
            "NLTE transfer requires positive total extinction and bottom absorption, and nonnegative emission/scattering"
        )
    tau = optical_depth_from_mass_opacity(atmosphere.column_mass, a + s)
    # Use eta directly: stimulated emission can change the sign of net
    # absorption inside the atmosphere. Only the thermal bottom condition
    # requires eta/kappa, where positive absorption is checked above.
    source, field = mass_emissivity_field(
        tau,
        eta,
        a,
        s,
        bottom_source=eta[:, -1] / a[:, -1],
        column_mass=atmosphere.column_mass,
        n_angle=n_angle,
        wavelength_chunk_size=wavelength_chunk_size,
    )
    if not check_source:
        return source, field, None
    # Independent fixed-source formal solution on the SAME mass volumes.
    _, check = mass_field(
        tau,
        source,
        a + s,
        np.zeros_like(s),
        column_mass=atmosphere.column_mass,
        n_angle=n_angle,
        wavelength_chunk_size=wavelength_chunk_size,
        reconstruct_intensity=True,
    )
    closure = (eta + s * check.mean_intensity) / (a + s)
    error = float(np.max(abs(source - closure) / np.maximum(abs(source), 1e-100)))
    return source, field, error
