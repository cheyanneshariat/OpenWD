"""Local material thermal response on the shared mass-volume operator."""
import numpy as np
from ._compat import trapezoid
from .constants import LIGHT_SPEED
from .nonlinear import RecoverableEvaluationError
from ._mass_feautrier import (
    MassResponseOperator,
    mass_response,
    mass_width,
    mass_emissivity_energy,
)
from ._pg1159_transfer import transfer_field
from .opacity import optical_depth_from_mass_opacity


def thermal_response(
    atmosphere,
    wavelength,
    coefficients,
    probes,
    target,
    n_angle,
    *,
    hydrostatic=False,
    radiative_acceleration_scale=1.0,
    local_energy_mask=None,
    flux_profile_residual=False,
    hybrid_band=None,
):
    """Return all depth derivatives with bounded wavelength-chunk memory.

    Each probe pair perturbs one material coordinate at every depth. Local
    opacity/emissivity derivatives are separated into independent columns by
    the transfer response operator. No statistical-equilibrium solve or
    wavelength-by-depth-by-depth array is required for each column.
    """
    a = atmosphere
    c = coefficients
    wave = wavelength
    nd = a.n_depth
    source, field, _ = transfer_field(a, c, n_angle=n_angle, check_source=False)
    ext = c.total_extinction
    response_args = (
        optical_depth_from_mass_opacity(a.column_mass, ext),
        wave,
        source,
        c.scattering / ext,
        a.column_mass,
    )
    response_options = dict(
        extinction=ext,
        n_angle=n_angle,
        wavelength_chunk_size=128,
        allow_stimulated_gain=True,
        reconstruct_intensity=True,
    )
    # One coordinate needs only one pass. Retaining every wavelength's
    # factors would consume hundreds of MB without reusing any of them.
    response = (
        MassResponseOperator(*response_args, **response_options)
        if len(probes) > 1
        else None
    )
    energy, emission = mass_emissivity_energy(
        wave,
        a.column_mass,
        c.thermal_emissivity,
        field.mean_intensity,
        c.true_absorption,
    )
    scale = np.maximum(abs(emission), 1e-30 * target)
    if local_energy_mask is None:
        local_energy_mask = abs(emission) < target
    local_energy_mask = np.asarray(local_energy_mask, dtype=bool)
    if local_energy_mask.shape != (nd - 1,):
        raise ValueError("local_energy_mask must have one value per structure cell")
    widths = 4 * np.pi * mass_width(a.column_mass)
    spacing = np.diff(wave)
    weights = 0.5 * (np.r_[0.0, spacing] + np.r_[spacing, 0.0])
    matrices = []
    acceleration = trapezoid(ext * field.flux, wave, axis=0) / LIGHT_SPEED
    effective = a.gravity - radiative_acceleration_scale * acceleration
    target_pressure = np.r_[
        effective[0] * a.column_mass[0],
        effective[0] * a.column_mass[0]
        + np.cumsum(0.5 * (effective[1:] + effective[:-1]) * np.diff(a.column_mass)),
    ]
    for coordinate, (plus, minus, step) in enumerate(probes):
        da = (plus.true_absorption - minus.true_absorption) / (2 * step)
        ds = (plus.scattering - minus.scattering) / (2 * step)
        deta = (plus.thermal_emissivity - minus.thermal_emissivity) / (2 * step)
        direct = (deta + ds * field.mean_intensity - (da + ds) * source) / ext
        bottom = np.zeros_like(deta)
        bottom[:, -1] = (
            plus.thermal_emissivity[:, -1] / plus.true_absorption[:, -1]
            - minus.thermal_emissivity[:, -1] / minus.true_absorption[:, -1]
        ) / (2 * step)
        energy_j = np.zeros((nd - 1, nd))

        def consume(start, stop, mean):
            energy_j[:] += widths[:, None] * np.einsum(
                "wdk,w->dk",
                c.true_absorption[start:stop, :-1, None] * mean[:, :-1, :],
                weights[start:stop],
            )

        acceleration_j = np.zeros((nd, nd))

        def consume_flux(start, stop, nodal):
            acceleration_j[:] += (
                np.einsum(
                    "wdk,w->dk", ext[start:stop, :, None] * nodal, weights[start:stop]
                )
                / LIGHT_SPEED
            )

        consumers = dict(
            return_auxiliary_response=False,
            mean_response_consumer=consume,
            nodal_flux_response_consumer=consume_flux if hydrostatic else None,
        )
        if response is None:
            tau, _, _, fraction, mass = response_args
            df, _, _ = mass_response(
                tau,
                wave,
                source,
                direct,
                bottom,
                fraction,
                mass,
                da + ds,
                **response_options,
                **consumers
            )
        else:
            df, _, _ = response.apply(direct, bottom, da + ds, **consumers)
        df /= target
        cells = np.arange(nd - 1)
        energy_j[cells, cells] += widths * trapezoid(
            da[:, :-1] * field.mean_intensity[:, :-1] - deta[:, :-1], wave, axis=0
        )
        thermal = energy_j / scale[:, None]
        emission_j = widths * trapezoid(deta[:, :-1], wave, axis=0)
        thermal[cells, cells] -= np.where(
            abs(emission) > 1e-30 * target, energy * emission_j / scale ** 2, 0.0
        )
        if flux_profile_residual:
            # The release gate is stated in terms of the bolometric flux at
            # every interface.  Differentiating that same profile gives every
            # temperature coordinate an explicit equation and avoids the
            # accumulated drift that can satisfy small cell-wise heating
            # ratios while missing the target flux by tens of percent.
            matrix = df
            if hybrid_band is not None:
                # Cells whose emission is below the stellar flux barely move
                # the interface fluxes; solve their local heating instead.
                first, last = hybrid_band
                matrix = np.vstack((df[:first], thermal[first:last], df[last:]))
        else:
            thermal = np.where(
                local_energy_mask[:, None], thermal, np.diff(df, axis=0)
            )
            # Match the legacy PG1159 correction: flux constancy supplies the
            # cell equations and the emergent bolometric flux fixes the remaining
            # nearly constant temperature mode.  The deepest interface is the
            # least reliable place to anchor that mode on a finite atmosphere.
            matrix = np.vstack((thermal, df[0]))
        if hydrostatic:
            indices = np.arange(nd)
            acceleration_j[indices, indices] += (
                trapezoid((da + ds) * field.flux, wave, axis=0) / LIGHT_SPEED
            )
            pressure_j = np.empty_like(acceleration_j)
            pressure_j[0] = acceleration_j[0] * a.column_mass[0]
            pressure_j[1:] = pressure_j[0] + np.cumsum(
                0.5
                * (acceleration_j[1:] + acceleration_j[:-1])
                * np.diff(a.column_mass)[:, None],
                axis=0,
            )
            pressure_j *= radiative_acceleration_scale
            pressure_j /= target_pressure[:, None]
            if coordinate == 1:
                pressure_j += np.eye(nd)
            matrix = np.vstack((matrix, pressure_j))
        matrices.append(matrix)
    return np.hstack(matrices)


def refine_population_response(
    matrix,
    residual,
    measure,
    propose,
    *,
    relative_tolerance=0.05,
    maximum_probes=None,
):
    """Learn the eliminated population response without erasing earlier probes.

    ``measure(unit_direction)`` returns the fully reclosed residual derivative.
    Orthogonal probes retain every previously measured column action. A single
    rank-one correction need not describe the *new* Newton direction it creates.
    Continue until that direction lies in the measured subspace, or an independent
    measurement verifies it. This changes only the caller's approximate Jacobian;
    the common driver must still evaluate and accept the resulting physical trial.
    """
    matrix = np.array(matrix, dtype=float, copy=True)
    if maximum_probes is not None and (
        isinstance(maximum_probes, (bool, np.bool_))
        or not isinstance(maximum_probes, (int, np.integer))
        or maximum_probes < 1
    ):
        raise ValueError("maximum_probes must be a positive integer or None")
    basis = []
    errors = []
    validated = False
    probe_failure = None
    for _ in range(len(residual) + 1):
        direction = np.asarray(propose(matrix), dtype=float)
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm):
            raise ValueError("nonfinite PG1159 thermal response direction")
        if norm < 1e-14:
            validated = True
            break
        direction = direction / norm
        remaining = direction.copy()
        # Reorthogonalize to prevent loss of the measured subspace when the
        # atmosphere Jacobian has very weak thermal modes.
        for _pass in range(2):
            for q in basis:
                remaining -= np.dot(q, remaining) * q
        size = np.linalg.norm(remaining)
        if size < 1e-8:
            validated = True
            break
        if maximum_probes is not None and len(errors) >= maximum_probes:
            break
        # A nearly repeated direction cannot determine its small unmeasured
        # component accurately: dividing two noisy actions by that small
        # projection amplifies material/finite-difference error. Probe the
        # orthogonal component directly in that case, retaining prior actions.
        orthogonal_probe = size < 0.25
        probe = remaining / size if orthogonal_probe else direction
        try:
            measured = np.asarray(measure(probe), dtype=float)
        except RecoverableEvaluationError as exc:
            # This probe only refines an approximate Jacobian. The accepted
            # atmosphere remains valid, so retain the response learned so far
            # and let the common driver test/backtrack its physical trial.
            probe_failure = str(exc)
            break
        if measured.shape != residual.shape or np.any(~np.isfinite(measured)):
            raise ValueError("invalid PG1159 measured thermal response")
        predicted = matrix @ probe
        error = float(
            np.linalg.norm(measured - predicted) / max(np.linalg.norm(measured), 1e-30)
        )
        errors.append(error)
        if error <= relative_tolerance and not orthogonal_probe:
            validated = True
            break
        q = remaining / size
        matrix += np.outer(measured - predicted, q) / np.dot(q, probe)
        basis.append(q)
        if len(basis) == len(residual):
            validated = True
            break
    return (
        matrix,
        {
            "population_response_probe_count": len(errors),
            "population_response_directional_defects": errors,
            "population_response_direction_validated": validated,
            "population_response_measured_rank": len(basis),
            "population_response_probe_failed": probe_failure is not None,
            "population_response_probe_failure": probe_failure,
        },
    )
