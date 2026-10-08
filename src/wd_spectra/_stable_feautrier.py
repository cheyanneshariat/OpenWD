"""Cancellation-resistant block Feautrier elimination; identical discretization.

Write a row as Q*u + A*(u-u_prev) + C*(u-u_next) = b, with A,C
diagonal. Eliminate using G=Q+A*solve(P_prev,G_prev), P=C+G instead
of subtracting A*solve(P_prev,C_prev) from A+C+Q. Back substitution
also retains intensity increments instead of differencing nearly equal u.
Used by an explicitly selected final flux-polishing phase.
"""
from __future__ import annotations

import numpy as np
from .radiative_transfer import RadiationField, angular_quadrature, _validate_inputs


def _positive_surface_depth(tau):
    if np.any(tau[:, 0] <= 0):
        raise ValueError("material optical depth must start above zero")


def _validate_chunk_size(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("wavelength_chunk_size must be a positive integer")


class _DifferenceFactors:
    def __init__(self, tau, scattering_fraction, n_angle):
        self.mu, self.weight = angular_quadrature(n_angle)
        nw, nd = tau.shape
        nr = len(self.mu)
        self.h = np.diff(tau, axis=1, prepend=0.0)
        self.a = np.zeros((nw, nd + 1, nr))
        self.c = np.zeros_like(self.a)
        self.c[:, 0] = self.mu / self.h[:, :1]
        self.a[:, 1:-1] = (
            2
            * self.mu ** 2
            / (self.h[:, :-1, None] * (self.h[:, :-1] + self.h[:, 1:])[:, :, None])
        )
        self.c[:, 1:-1] = (
            2
            * self.mu ** 2
            / (self.h[:, 1:, None] * (self.h[:, :-1] + self.h[:, 1:])[:, :, None])
        )
        eye = np.eye(nr)
        q = np.broadcast_to(eye, (nw, nd + 1, nr, nr)).copy()
        q[:, 1:-1] -= scattering_fraction[:, :-1, None, None] * self.weight
        self.g = np.empty_like(q)
        self.p = np.empty_like(q)
        self.g[:, 0] = q[:, 0]
        self.p[:, 0] = self.g[:, 0] + self.c[:, 0, :, None] * eye
        for i in range(1, nd + 1):
            self.g[:, i] = q[:, i] + self.a[:, i, :, None] * np.linalg.solve(
                self.p[:, i - 1], self.g[:, i - 1]
            )
            self.p[:, i] = self.g[:, i] + self.c[:, i, :, None] * eye

    def solve(self, rhs, *, reconstruct_intensity=False):
        """Return symmetric intensities and stable forward depth increments.

        Emissivity-form NLTE callers request sum-form reconstruction of faint
        intensities; the established LTE arithmetic remains the default.
        """
        rhs = np.asarray(rhs)
        vector = rhs.ndim == 3
        reduced = rhs[..., None].copy() if vector else rhs.copy()
        nw, nd, nr, nk = reduced.shape
        for i in range(1, nd):
            reduced[:, i] += self.a[:, i, :, None] * np.linalg.solve(
                self.p[:, i - 1], reduced[:, i - 1]
            )
        u = np.empty_like(reduced)
        jump = np.empty((nw, nd - 1, nr, nk))
        u[:, -1] = np.linalg.solve(self.p[:, -1], reduced[:, -1])
        for i in range(nd - 2, -1, -1):
            jump[:, i] = np.linalg.solve(
                self.p[:, i], self.g[:, i] @ u[:, i + 1] - reduced[:, i]
            )
            u[:, i] = u[:, i + 1] - jump[:, i]
            if reconstruct_intensity:
                # The increment form preserves small fluxes when adjacent
                # intensities are nearly equal. In the opposite limit it can
                # lose a small intensity by subtracting two large values (and
                # even turn positive emission into a negative intensity).
                # Re-solve the same eliminated row in its sum form where the
                # subtraction cancels more than half the operand magnitudes.
                cancellation = np.abs(u[:, i]) < .5 * (
                    np.abs(u[:, i + 1]) + np.abs(jump[:, i])
                )
                affected = np.any(cancellation, axis=(1, 2))
                if np.any(affected):
                    direct = np.linalg.solve(
                        self.p[affected, i],
                        reduced[affected, i]
                        + self.c[affected, i, :, None] * u[affected, i + 1],
                    )
                    u[affected, i] = np.where(
                        cancellation[affected], direct, u[affected, i]
                    )
        return (u[..., 0], jump[..., 0]) if vector else (u, jump)


def cancellation_safe_field(
    optical_depth,
    planck_function,
    true_absorption,
    scattering,
    *,
    n_angle=4,
    wavelength_chunk_size=64
):
    """Solve the existing coupled isotropic-scattering equations without cancellation.

    The inserted vacuum surface and thermalized lower boundary are unchanged.
    Chunking bounds angular block storage independently of wavelength count.
    """
    tau, planck = _validate_inputs(optical_depth, planck_function)
    tau = np.broadcast_to(tau, planck.shape)
    _positive_surface_depth(tau)
    _validate_chunk_size(wavelength_chunk_size)
    absorption = np.broadcast_to(np.asarray(true_absorption, dtype=float), planck.shape)
    scatter = np.broadcast_to(np.asarray(scattering, dtype=float), planck.shape)
    extinction = absorption + scatter
    if (
        np.any(~np.isfinite(extinction))
        or np.any(absorption < 0)
        or np.any(scatter < 0)
        or np.any(extinction <= 0)
    ):
        raise ValueError(
            "opacities must be finite nonnegative with positive extinction"
        )
    epsilon = absorption / extinction
    fraction = scatter / extinction
    nw, nd = planck.shape
    mean, flux, interface = (np.empty_like(planck) for _ in range(3))
    for start in range(0, nw, wavelength_chunk_size):
        stop = min(nw, start + wavelength_chunk_size)
        factor = _DifferenceFactors(tau[start:stop], fraction[start:stop], n_angle)
        rhs = np.zeros((stop - start, nd + 1, n_angle))
        rhs[:, 1:-1] = (epsilon[start:stop, :-1] * planck[start:stop, :-1])[:, :, None]
        rhs[:, -1] = planck[start:stop, -1, None]
        u, jump = factor.solve(rhs)
        mean[start:stop] = u[:, 1:] @ factor.weight
        derivative = jump / factor.h[:, :, None]
        interface[start:stop] = derivative @ (
            4 * np.pi * factor.weight * factor.mu ** 2
        )
        node_derivative = derivative.copy()
        node_derivative[:, :-1] = (
            factor.h[:, 1:, None] * derivative[:, :-1]
            + factor.h[:, :-1, None] * derivative[:, 1:]
        ) / (factor.h[:, 1:] + factor.h[:, :-1])[:, :, None]
        flux[start:stop] = node_derivative @ (
            4 * np.pi * factor.weight * factor.mu ** 2
        )
    return (
        epsilon * planck + fraction * mean,
        RadiationField(mean_intensity=mean, flux=flux, interface_flux=interface),
    )


def cancellation_safe_scalar_field(optical_depth, source_function, *, n_angle=4):
    """Independent prescribed-source sweep, without scattering feedback."""
    return cancellation_safe_field(
        optical_depth, source_function, 1.0, 0.0, n_angle=n_angle
    )[1]


def cancellation_safe_response(
    optical_depth,
    wavelength_angstrom,
    source_function,
    direct_source_derivative,
    planck_derivative,
    scattering_fraction,
    column_mass,
    mass_opacity_derivative,
    *,
    n_angle=4,
    wavelength_chunk_size=16,
    return_auxiliary_response=True,
    mean_response_consumer=None
):
    """Differentiate the same coupled field on fixed column-mass nodes.

    The source derivative excludes feedback through J; this solve includes
    that feedback exactly, as well as opacity-motion terms and the lower
    Planck boundary. The optional consumer receives read-only chunks of dJ.
    """
    tau, source = _validate_inputs(optical_depth, source_function)
    tau = np.broadcast_to(tau, source.shape)
    _positive_surface_depth(tau)
    _validate_chunk_size(wavelength_chunk_size)
    fraction = np.broadcast_to(
        np.asarray(scattering_fraction, dtype=float), source.shape
    )
    mass = np.asarray(column_mass, dtype=float)
    nw, nd = source.shape
    direct_source_derivative = np.asarray(direct_source_derivative, dtype=float)
    planck_derivative = np.asarray(planck_derivative, dtype=float)
    mass_opacity_derivative = np.asarray(mass_opacity_derivative, dtype=float)
    if any(
        a.shape != source.shape or np.any(~np.isfinite(a))
        for a in (direct_source_derivative, planck_derivative, mass_opacity_derivative)
    ):
        raise ValueError("derivatives must be finite and match source shape")
    if np.any(~np.isfinite(fraction)) or np.any((fraction < 0) | (fraction > 1)):
        raise ValueError("scattering fraction must be finite between zero and one")
    if (
        mass.shape != (nd,)
        or np.any(~np.isfinite(mass))
        or np.any(mass <= 0)
        or np.any(np.diff(mass) <= 0)
    ):
        raise ValueError("column mass must be positive and strictly increasing")
    integrated = np.zeros((nd, nd))
    mean_response = np.empty((nw, nd, nd)) if return_auxiliary_response else None
    source_response = np.empty((nw, nd, nd)) if return_auxiliary_response else None
    wave = np.asarray(wavelength_angstrom, dtype=float)
    if (
        wave.shape != (nw,)
        or nw < 2
        or np.any(~np.isfinite(wave))
        or np.any(wave <= 0)
        or np.any(np.diff(wave) <= 0)
    ):
        raise ValueError("wavelengths must match, be positive, and strictly increase")
    weights = np.diff(wave, prepend=wave[0], append=wave[-1])
    weights = 0.5 * (weights[:-1] + weights[1:])
    for start in range(0, nw, wavelength_chunk_size):
        stop = min(nw, start + wavelength_chunk_size)
        nc = stop - start
        local_tau = tau[start:stop]
        # Recover angular intensities for the supplied, coupled source with
        # a scalar-source solve, using the same stable elimination.
        scalar = _DifferenceFactors(local_tau, np.zeros((nc, nd)), n_angle)
        rhs = np.zeros((nc, nd + 1, n_angle))
        rhs[:, 1:] = source[start:stop, :, None]
        _, jumps = scalar.solve(rhs)
        coupled = _DifferenceFactors(local_tau, fraction[start:stop], n_angle)
        dh = np.zeros((nc, nd, nd))
        dk = np.asarray(mass_opacity_derivative)[start:stop]
        dh[:, 0, 0] = mass[0] * dk[:, 0]
        for i in range(1, nd):
            dh[:, i, i - 1] = 0.5 * (mass[i] - mass[i - 1]) * dk[:, i - 1]
            dh[:, i, i] = 0.5 * (mass[i] - mass[i - 1]) * dk[:, i]
        h = coupled.h
        tangent_rhs = np.zeros((nc, nd + 1, n_angle, nd))
        for i in range(nd - 1):
            tangent_rhs[:, i + 1, :, i] = np.asarray(direct_source_derivative)[
                start:stop, i, None
            ]
        tangent_rhs[:, -1, :, -1] = np.asarray(planck_derivative)[start:stop, -1, None]
        tangent_rhs[:, 0] = (
            -coupled.mu[None, :, None]
            * dh[:, 0, None, :]
            / h[:, 0, None, None] ** 2
            * jumps[:, 0, :, None]
        )
        summed = (dh[:, :-1] + dh[:, 1:]) / (h[:, :-1] + h[:, 1:])[:, :, None]
        da = -coupled.a[:, 1:-1, :, None] * (
            dh[:, :-1, None, :] / h[:, :-1, None, None] + summed[:, :, None]
        )
        dc = -coupled.c[:, 1:-1, :, None] * (
            dh[:, 1:, None, :] / h[:, 1:, None, None] + summed[:, :, None]
        )
        tangent_rhs[:, 1:-1] += (
            -da * jumps[:, :-1, :, None] + dc * jumps[:, 1:, :, None]
        )
        response, jump_response = coupled.solve(tangent_rhs)
        local_mean = np.einsum("wdrk,r->wdk", response[:, 1:], coupled.weight)
        if return_auxiliary_response:
            mean_response[start:stop] = local_mean
            sr = fraction[start:stop, :, None] * local_mean
            sr[:, np.arange(nd), np.arange(nd)] += np.asarray(direct_source_derivative)[
                start:stop
            ]
            source_response[start:stop] = sr
        if mean_response_consumer is not None:
            local_mean.flags.writeable = False
            mean_response_consumer(start, stop, local_mean)
        local_flux = np.einsum(
            "wdrk,r->wdk",
            jump_response / h[:, :, None, None]
            - jumps[:, :, :, None] * dh[:, :, None, :] / h[:, :, None, None] ** 2,
            4 * np.pi * coupled.weight * coupled.mu ** 2,
        )
        integrated += np.einsum("wdk,w->dk", local_flux, weights[start:stop])
    return integrated, mean_response, source_response
