import numpy as np
import pytest

from wd_spectra._compat import trapezoid
from wd_spectra.radiative_transfer import (
    coherent_scattering_feautrier_field,
    compiled_backend_available,
    emergent_flux,
    emergent_specific_intensity,
    emergent_stokes_specific_intensity,
    feautrier_radiation_field,
    integrated_emergent_flux_state_response,
    integrated_coherent_scattering_feautrier_state_response,
    integrated_lambda_response,
    radiation_field,
)


def test_isothermal_semi_infinite_atmosphere_has_pi_s_flux():
    tau = np.geomspace(1.0e-8, 1.0e3, 100)
    source = np.vstack([np.ones_like(tau), 7.5 * np.ones_like(tau)])
    flux = emergent_flux(tau, source, backend="python")
    np.testing.assert_allclose(flux, np.pi * np.array([1.0, 7.5]), rtol=2e-12)


def test_isothermal_specific_intensity_is_source_function():
    tau = np.geomspace(1.0e-8, 1.0e3, 100)
    source = np.vstack([np.ones_like(tau), 7.5 * np.ones_like(tau)])
    intensity = emergent_specific_intensity(tau, source, 0.37)
    np.testing.assert_allclose(intensity, np.array([1.0, 7.5]), rtol=2e-12)


@pytest.mark.parametrize("explicit_emission", [False, True])
def test_linear_stokes_preserves_lte_with_variable_propagation(explicit_emission):
    mass = np.geomspace(1e-5, 100., 40)
    eta_i = np.ones((1, mass.size))
    eta_q = 0.7 * eta_i * mass / (1 + mass)
    eta_v = 0.2 * eta_i / (1 + mass)
    rho_q = 2. * eta_i / (1 + mass)
    rho_v = 3. * eta_i * mass / (1 + mass)
    emission = None
    if explicit_emission:
        emission = np.zeros(eta_i.shape + (4,))
        emission[..., 0] = eta_i
        emission[..., 1] = eta_q
        emission[..., 3] = eta_v
    stokes = emergent_stokes_specific_intensity(
        mass, eta_i, eta_q, eta_v, rho_q, rho_v, eta_i, 0.63,
        emission_stokes=emission, formal_solver="delo-linear",
    )
    np.testing.assert_allclose(stokes.i, 1., atol=1e-12, rtol=0)
    for polarization in (stokes.q, stokes.u, stokes.v):
        np.testing.assert_allclose(polarization, 0., atol=1e-12)


def test_linear_stokes_matches_constant_matrix_analytic_solution():
    from scipy.linalg import expm

    mass = np.linspace(0., 40., 12)
    eta_i = np.ones((1, mass.size))
    matrix = np.array([[1, .3, 0, .1], [.3, 1, .7, 0],
                       [0, -.7, 1, .4], [.1, 0, -.4, 1]])
    offset = np.array([1., .03, .02, -.01])
    gradient = np.array([.1, .002, -.003, .004])
    equilibrium = offset + mass[:, None] * gradient
    emission = (equilibrium @ matrix.T)[None, :, :]
    mu = .63
    stokes = emergent_stokes_specific_intensity(
        mass, eta_i, .3 * eta_i, .1 * eta_i, .4 * eta_i, .7 * eta_i,
        emission[..., 0], mu, emission_stokes=emission, formal_solver="delo-linear",
    )
    correction = mu * np.linalg.solve(matrix, gradient)
    expected = offset + correction - expm(-matrix * mass[-1] / mu) @ correction
    np.testing.assert_allclose(
        np.array([stokes.i[0], stokes.q[0], stokes.u[0], stokes.v[0]]), expected,
        rtol=1e-12, atol=1e-12,
    )


def test_linear_stokes_variable_modes_converge_to_independent_quadrature():
    from scipy.integrate import quad

    mu = .7
    bottom = 15.
    modes = []
    for sign in (-1, 1):
        def depth(m):
            return m + sign * .6 * (m - np.log1p(m))

        def integrand(m):
            opacity = 1 + sign * .6 * m / (1 + m)
            return (1 + .3 * m) * opacity / mu * np.exp(-depth(m) / mu)

        modes.append((1 + .3 * bottom) * np.exp(-depth(bottom) / mu)
                     + quad(integrand, 0., bottom, epsabs=1e-12)[0])
    expected = np.array([(modes[1] + modes[0]) / 2, (modes[1] - modes[0]) / 2])
    errors = []
    for count in (40, 80, 160):
        mass = np.linspace(0., bottom, count)
        eta_i = np.ones((1, count))
        eta_q = .6 * eta_i * mass / (1 + mass)
        zero = np.zeros_like(eta_i)
        stokes = emergent_stokes_specific_intensity(
            mass, eta_i, eta_q, zero, zero, zero, 1 + .3 * mass[None, :], mu,
            formal_solver="delo-linear",
        )
        errors.append(np.max(np.abs(np.array([stokes.i[0], stokes.q[0]]) - expected)))
    assert errors[-1] < 2e-4
    assert errors[1] < errors[0] / 3
    assert errors[2] < errors[1] / 3


def test_stokes_solver_has_exact_scalar_limit():
    mass = np.geomspace(1.0e-8, 1.0e3, 80)
    opacity = np.vstack(
        (0.7 + 0.2 * mass / (1.0 + mass), 2.0 + 0.1 * mass / (1.0 + mass))
    )
    source = np.vstack((1.0 + 0.03 * np.log1p(mass), 4.0 + 0.2 * np.log1p(mass)))
    tau = np.empty_like(opacity)
    tau[:, 0] = opacity[:, 0] * mass[0]
    tau[:, 1:] = tau[:, [0]] + np.cumsum(
        0.5 * (opacity[:, 1:] + opacity[:, :-1]) * np.diff(mass), axis=1
    )
    expected = emergent_specific_intensity(tau, source, 0.47)
    zero = np.zeros_like(opacity)
    actual = emergent_stokes_specific_intensity(
        mass, opacity, zero, zero, zero, zero, source, 0.47
    )
    np.testing.assert_array_equal(actual.i, expected)
    np.testing.assert_array_equal(actual.q, np.zeros(2))
    np.testing.assert_array_equal(actual.u, np.zeros(2))
    np.testing.assert_array_equal(actual.v, np.zeros(2))


def test_stokes_solver_thermalizes_and_rotates_generated_polarization():
    mass = np.geomspace(1.0e-7, 1.0e3, 240)
    shape = (3, mass.size)
    eta_i = np.full(shape, 1.0)
    eta_q = np.full(shape, 0.18)
    eta_v = np.full(shape, -0.11)
    rho_q = np.full(shape, 0.23)
    rho_v = np.full(shape, -0.31)
    source = np.broadcast_to(np.asarray([[1.0], [2.0], [5.0]]), shape)
    stokes = emergent_stokes_specific_intensity(
        mass, eta_i, eta_q, eta_v, rho_q, rho_v, source, 0.63
    )
    # An isothermal, semi-infinite LTE atmosphere has the unpolarized Planck
    # vector as its exact equilibrium solution even when the modes are coupled.
    np.testing.assert_allclose(stokes.i, np.asarray([1.0, 2.0, 5.0]), rtol=2e-8)
    np.testing.assert_allclose(stokes.q, 0.0, atol=2e-8)
    np.testing.assert_allclose(stokes.u, 0.0, atol=2e-8)
    np.testing.assert_allclose(stokes.v, 0.0, atol=2e-8)


def test_matrix_exponential_stokes_solver_thermalizes():
    mass = np.geomspace(1.0e-7, 1.0e3, 80)
    shape = (2, mass.size)
    eta_i = np.full(shape, 1.0)
    eta_q = np.full(shape, 0.18)
    eta_v = np.full(shape, -0.11)
    rho_q = np.full(shape, 2.3)
    rho_v = np.full(shape, -3.1)
    source = np.broadcast_to(np.asarray([[1.0], [5.0]]), shape)
    stokes = emergent_stokes_specific_intensity(
        mass,
        eta_i,
        eta_q,
        eta_v,
        rho_q,
        rho_v,
        source,
        0.63,
        formal_solver="matrix-exponential",
    )
    np.testing.assert_allclose(stokes.i, np.asarray([1.0, 5.0]), rtol=2e-10)
    np.testing.assert_allclose(stokes.q, 0.0, atol=2e-10)
    np.testing.assert_allclose(stokes.u, 0.0, atol=2e-10)
    np.testing.assert_allclose(stokes.v, 0.0, atol=2e-10)


def test_stokes_solver_accepts_physical_polarized_emission_vector():
    mass = np.geomspace(1.0e-7, 1.0e3, 160)
    shape = (2, mass.size)
    eta_i = np.full(shape, 1.2)
    eta_q = np.full(shape, 0.16)
    eta_v = np.full(shape, -0.09)
    rho_q = np.full(shape, 0.27)
    rho_v = np.full(shape, -0.19)
    planck = np.broadcast_to(np.asarray([[2.0], [5.0]]), shape)
    emission = np.zeros(shape + (4,))
    emission[..., 0] = eta_i * planck
    emission[..., 1] = eta_q * planck
    emission[..., 3] = eta_v * planck
    stokes = emergent_stokes_specific_intensity(
        mass,
        eta_i,
        eta_q,
        eta_v,
        rho_q,
        rho_v,
        planck,
        0.52,
        emission_stokes=emission,
    )
    np.testing.assert_allclose(stokes.i, np.asarray([2.0, 5.0]), rtol=2e-8)
    np.testing.assert_allclose(stokes.q, 0.0, atol=2e-8)
    np.testing.assert_allclose(stokes.u, 0.0, atol=2e-8)
    np.testing.assert_allclose(stokes.v, 0.0, atol=2e-8)


def test_transfer_rejects_decreasing_optical_depth():
    with pytest.raises(ValueError, match="increase strictly"):
        emergent_flux([0.0, 2.0, 1.0], [1.0, 1.0, 1.0])


def test_radiation_field_surface_flux_matches_emergent_flux():
    tau = np.r_[0.0, np.geomspace(1.0e-6, 1.0e3, 79)]
    source = np.vstack((1.0 + 0.4 * tau, 2.0 + 0.1 * tau))
    field = radiation_field(tau, source, n_angle=4)
    expected = emergent_flux(tau, source, n_angle=4, backend="python")
    np.testing.assert_allclose(field.flux[:, 0], expected, rtol=2e-14)
    assert np.all(field.mean_intensity >= 0.0)


def test_feautrier_isothermal_surface_flux_and_deep_mean_intensity():
    tau = np.geomspace(1.0e-8, 100.0, 80)
    source = np.vstack((np.ones_like(tau), 7.5 * np.ones_like(tau)))
    field = feautrier_radiation_field(tau, source, n_angle=6)
    assert field.interface_flux is not None
    assert field.interface_flux.shape == source.shape
    np.testing.assert_allclose(
        field.interface_flux[:, 0], np.pi * source[:, 0], rtol=3.0e-3
    )
    np.testing.assert_allclose(
        field.mean_intensity[:, -3], source[:, -3], rtol=2.0e-6
    )


def test_coupled_coherent_scattering_satisfies_source_equation():
    tau = np.geomspace(1.0e-6, 100.0, 40)
    planck = np.vstack((1.0 + 0.3 * np.log1p(tau),))
    absorption = np.full_like(planck, 0.03)
    scattering = np.full_like(planck, 0.97)
    source, field = coherent_scattering_feautrier_field(
        tau, planck, absorption, scattering, n_angle=3
    )
    expected = 0.03 * planck + 0.97 * field.mean_intensity
    np.testing.assert_allclose(source, expected, rtol=3.0e-12, atol=1.0e-12)


def test_coupled_source_closes_an_independent_formal_transfer_solve():
    # Five wavelengths and chunks of two also exercise a partial final batch
    # whose size differs from the angle count (important on NumPy 2).
    tau = np.geomspace(1.0e-5, 50.0, 24)
    planck = (np.arange(1.0, 6.0)[:, None]
              * (1.0 + 0.3 * np.log1p(tau)))
    for epsilon in (0.3, 0.03, 1.2e-6):
        source, coupled = coherent_scattering_feautrier_field(
            tau, planck, np.full_like(planck, epsilon),
            np.full_like(planck, 1.0 - epsilon), n_angle=3,
            wavelength_chunk_size=2,
        )
        # Not just S=epsilon*B+(1-epsilon)*J using the J that constructed S:
        # propagate S through the independent scalar Feautrier implementation.
        formal = feautrier_radiation_field(tau, source, n_angle=3)
        np.testing.assert_allclose(formal.mean_intensity,
                                   coupled.mean_intensity, rtol=2e-8, atol=2e-10)
        np.testing.assert_allclose(source, epsilon*planck
                                   + (1-epsilon)*formal.mean_intensity,
                                   rtol=2e-8, atol=2e-10)
        np.testing.assert_allclose(formal.interface_flux,
                                   coupled.interface_flux, rtol=2e-8, atol=2e-10)


def test_coupled_scattering_reduces_exactly_to_absorption_only():
    tau = np.vstack(
        (
            np.geomspace(2.0e-6, 60.0, 24),
            np.geomspace(9.0e-5, 140.0, 24),
        )
    )
    depth = np.arange(tau.shape[1], dtype=np.float64)
    planck = np.vstack((1.0 + 0.02 * depth**2, 2.0 + 0.15 * depth))
    source, actual = coherent_scattering_feautrier_field(
        tau,
        planck,
        np.ones_like(planck),
        np.zeros_like(planck),
        n_angle=4,
    )
    expected = feautrier_radiation_field(tau, planck, n_angle=4)
    np.testing.assert_array_equal(source, planck)
    np.testing.assert_array_equal(actual.mean_intensity, expected.mean_intensity)
    np.testing.assert_array_equal(actual.interface_flux, expected.interface_flux)


@pytest.mark.parametrize("step", [0.1, 0.2])
def test_coupled_scattering_state_response_matches_finite_difference(step):
    wavelength = np.array([900.0, 1800.0, 4200.0, 9000.0])
    mass = np.geomspace(1.0e-5, 20.0, 6)
    depth = np.arange(mass.size, dtype=np.float64)
    planck = (
        (1.0 + wavelength[:, np.newaxis] / 7000.0)
        * (1.0 + 0.12 * depth[np.newaxis, :] ** 1.3)
    )
    absorption = (
        8.0e-7
        * (1.0 + wavelength[:, np.newaxis] / 12_000.0)
        * (1.0 + 0.04 * depth[np.newaxis, :])
    )
    scattering = (
        0.35
        * (1.0 + 0.03 * depth[np.newaxis, :])
        * (1.0 + 0.0 * wavelength[:, np.newaxis])
    )
    planck_derivative = 0.17 * planck
    absorption_derivative = 0.08 * absorption
    scattering_derivative = -0.025 * scattering

    def optical_depth(extinction):
        result = np.empty_like(extinction)
        result[:, 0] = extinction[:, 0] * mass[0]
        result[:, 1:] = result[:, [0]] + np.cumsum(
            0.5
            * (extinction[:, 1:] + extinction[:, :-1])
            * np.diff(mass),
            axis=1,
        )
        return result

    extinction = absorption + scattering
    tau = optical_depth(extinction)
    source, field = coherent_scattering_feautrier_field(
        tau, planck, absorption, scattering, n_angle=3
    )
    direct_source_derivative = (
        absorption_derivative * planck
        + absorption * planck_derivative
        + scattering_derivative * field.mean_intensity
        - (absorption_derivative + scattering_derivative) * source
    ) / extinction
    flux_response, mean_response, source_response = (
        integrated_coherent_scattering_feautrier_state_response(
            tau,
            wavelength,
            source,
            direct_source_derivative,
            planck_derivative,
            scattering / extinction,
            mass,
            absorption_derivative + scattering_derivative,
            n_angle=3,
            wavelength_chunk_size=2,
        )
    )
    assert mean_response is not None and source_response is not None
    flux_only, omitted_mean, omitted_source = (
        integrated_coherent_scattering_feautrier_state_response(
            tau,
            wavelength,
            source,
            direct_source_derivative,
            planck_derivative,
            scattering / extinction,
            mass,
            absorption_derivative + scattering_derivative,
            n_angle=3,
            wavelength_chunk_size=2,
            return_auxiliary_response=False,
        )
    )
    np.testing.assert_array_equal(flux_only, flux_response)
    assert omitted_mean is None and omitted_source is None

    # Thin surface cells amplify cancellation in a small, second-order
    # finite difference. Use a fourth-order reference at two wider steps:
    # this reduces roundoff without weakening the derivative tolerances.
    for state_depth in range(mass.size):
        changed = []
        for direction in (-2.0, -1.0, 1.0, 2.0):
            changed_planck = planck.copy()
            changed_absorption = absorption.copy()
            changed_scattering = scattering.copy()
            changed_planck[:, state_depth] += (
                direction * step * planck_derivative[:, state_depth]
            )
            changed_absorption[:, state_depth] += (
                direction * step * absorption_derivative[:, state_depth]
            )
            changed_scattering[:, state_depth] += (
                direction * step * scattering_derivative[:, state_depth]
            )
            changed_extinction = changed_absorption + changed_scattering
            changed.append(
                coherent_scattering_feautrier_field(
                    optical_depth(changed_extinction),
                    changed_planck,
                    changed_absorption,
                    changed_scattering,
                    n_angle=3,
                )
            )
        (far_lower_source, far_lower_field), (lower_source, lower_field), (upper_source, upper_field), (far_upper_source, far_upper_field) = changed
        def derivative(far_lower, lower, upper, far_upper):
            return (8 * (upper - lower) - (far_upper - far_lower)) / (12 * step)
        assert lower_field.interface_flux is not None
        assert upper_field.interface_flux is not None
        numerical_flux = trapezoid(
            derivative(far_lower_field.interface_flux, lower_field.interface_flux,
                       upper_field.interface_flux, far_upper_field.interface_flux),
            wavelength,
            axis=0,
        )
        np.testing.assert_allclose(
            numerical_flux,
            flux_response[:, state_depth],
            rtol=2.0e-4,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            derivative(far_lower_field.mean_intensity, lower_field.mean_intensity,
                       upper_field.mean_intensity, far_upper_field.mean_intensity),
            mean_response[:, :, state_depth],
            rtol=2.0e-4,
            atol=2.0e-5,
        )
        np.testing.assert_allclose(
            derivative(far_lower_source, lower_source, upper_source, far_upper_source),
            source_response[:, :, state_depth],
            rtol=2.0e-4,
            atol=2.0e-5,
        )


def test_feautrier_linear_source_has_constant_deep_flux():
    tau = np.geomspace(1.0e-7, 100.0, 120)
    slope = 0.37
    source = (2.0 + slope * tau)[np.newaxis, :]
    field = feautrier_radiation_field(tau, source, n_angle=6)
    assert field.interface_flux is not None
    expected = 4.0 * np.pi * slope / 3.0
    interface_tau = np.sqrt(np.r_[0.0, tau[:-1]] * tau)
    selected = (interface_tau > 5.0) & (interface_tau < 40.0)
    np.testing.assert_allclose(
        field.interface_flux[0, selected], expected, rtol=6.0e-3
    )


def test_feautrier_interface_flux_obeys_discrete_moment_equation():
    tau = np.vstack(
        (
            np.geomspace(2.0e-5, 80.0, 31),
            np.geomspace(8.0e-4, 200.0, 31),
        )
    )
    depth_index = np.arange(tau.shape[1], dtype=np.float64)
    source = np.vstack(
        (
            1.0 + 0.11 * depth_index + 0.015 * depth_index**2,
            2.0 + 0.08 * depth_index + 0.009 * depth_index**2,
        )
    )
    field = feautrier_radiation_field(tau, source, n_angle=5)
    assert field.interface_flux is not None

    expanded_tau = np.column_stack((np.zeros(tau.shape[0]), tau))
    control_width = 0.5 * (
        expanded_tau[:, 2:] - expanded_tau[:, :-2]
    )
    flux_divergence = np.diff(field.interface_flux, axis=1) / control_width
    local_emission_imbalance = 4.0 * np.pi * (
        field.mean_intensity[:, :-1] - source[:, :-1]
    )
    np.testing.assert_allclose(
        flux_divergence,
        local_emission_imbalance,
        # Different compiled BLAS/toolchain evaluation orders move the most
        # cancellation-sensitive values by about two parts in 10 million.
        rtol=3.0e-7,
        atol=3.0e-8,
    )


def test_integrated_feautrier_interface_flux_response_matches_perturbations():
    from wd_spectra.radiative_transfer import (
        integrated_feautrier_interface_flux_response,
    )

    wavelength = np.asarray([900.0, 1300.0, 2400.0, 5100.0, 9000.0])
    depth_index = np.arange(6, dtype=np.float64)
    tau = np.multiply.outer(
        (wavelength / 2400.0) ** 0.7,
        np.geomspace(2.0e-4, 40.0, depth_index.size),
    )
    source = (
        0.7
        + 0.12 * depth_index[np.newaxis, :]
        + 0.03 * (wavelength[:, np.newaxis] / 1000.0)
    )
    derivative = 0.2 + 0.04 * depth_index[np.newaxis, :]
    derivative = np.broadcast_to(derivative, source.shape).copy()
    response = integrated_feautrier_interface_flux_response(
        tau,
        wavelength,
        derivative,
        n_angle=5,
        wavelength_chunk_size=2,
    )

    trapezoid_weight = np.empty(wavelength.size)
    spacing = np.diff(wavelength)
    trapezoid_weight[0] = 0.5 * spacing[0]
    trapezoid_weight[-1] = 0.5 * spacing[-1]
    trapezoid_weight[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    step = 1.0e-3
    finite_difference = np.empty_like(response)
    for perturbed_depth in range(source.shape[1]):
        plus = source.copy()
        minus = source.copy()
        plus[:, perturbed_depth] += step * derivative[:, perturbed_depth]
        minus[:, perturbed_depth] -= step * derivative[:, perturbed_depth]
        plus_flux = feautrier_radiation_field(
            tau, plus, n_angle=5
        ).interface_flux
        minus_flux = feautrier_radiation_field(
            tau, minus, n_angle=5
        ).interface_flux
        assert plus_flux is not None and minus_flux is not None
        finite_difference[:, perturbed_depth] = np.sum(
            (plus_flux - minus_flux)
            * trapezoid_weight[:, np.newaxis],
            axis=0,
        ) / (2.0 * step)
    # The reference is a centered finite difference and is cancellation
    # limited across compiled backends.  A few-ppm threshold still catches
    # physically meaningful response errors.
    np.testing.assert_allclose(
        response, finite_difference, rtol=2.0e-6, atol=2.0e-6
    )


def test_feautrier_state_response_includes_optical_depth_motion():
    from wd_spectra.radiative_transfer import (
        integrated_feautrier_interface_state_response,
    )

    wavelength = np.asarray([950.0, 1500.0, 3100.0, 7200.0])
    mass = np.geomspace(2.0e-4, 30.0, 6)
    depth = np.arange(mass.size, dtype=np.float64)
    opacity = (
        (0.4 + (wavelength[:, np.newaxis] / 3000.0) ** 0.6)
        * (0.8 + 0.12 * depth[np.newaxis, :])
    )
    source = (
        0.9
        + 0.16 * depth[np.newaxis, :]
        + 0.025 * wavelength[:, np.newaxis] / 1000.0
    )
    source_derivative = np.broadcast_to(
        0.18 + 0.03 * depth[np.newaxis, :], source.shape
    ).copy()
    opacity_derivative = opacity * (
        -0.25 + 0.07 * depth[np.newaxis, :]
    )

    def optical_depth(values):
        result = np.empty_like(values)
        result[:, 0] = values[:, 0] * mass[0]
        result[:, 1:] = result[:, [0]] + np.cumsum(
            0.5 * (values[:, 1:] + values[:, :-1])
            * np.diff(mass)[np.newaxis, :],
            axis=1,
        )
        return result

    response = integrated_feautrier_interface_state_response(
        optical_depth(opacity),
        wavelength,
        source,
        source_derivative,
        mass,
        opacity_derivative,
        n_angle=4,
        wavelength_chunk_size=2,
    )
    python_response = integrated_feautrier_interface_state_response(
        optical_depth(opacity),
        wavelength,
        source,
        source_derivative,
        mass,
        opacity_derivative,
        n_angle=4,
        wavelength_chunk_size=2,
        backend="python",
    )
    np.testing.assert_allclose(
        response, python_response, rtol=5.0e-9, atol=1.0e-10
    )
    trapezoid_weight = np.empty(wavelength.size)
    spacing = np.diff(wavelength)
    trapezoid_weight[0] = 0.5 * spacing[0]
    trapezoid_weight[-1] = 0.5 * spacing[-1]
    trapezoid_weight[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
    finite_difference = np.empty_like(response)
    step = 1.0e-3
    for perturbed_depth in range(mass.size):
        plus_source = source.copy()
        minus_source = source.copy()
        plus_opacity = opacity.copy()
        minus_opacity = opacity.copy()
        plus_source[:, perturbed_depth] += (
            step * source_derivative[:, perturbed_depth]
        )
        minus_source[:, perturbed_depth] -= (
            step * source_derivative[:, perturbed_depth]
        )
        plus_opacity[:, perturbed_depth] += (
            step * opacity_derivative[:, perturbed_depth]
        )
        minus_opacity[:, perturbed_depth] -= (
            step * opacity_derivative[:, perturbed_depth]
        )
        plus_flux = feautrier_radiation_field(
            optical_depth(plus_opacity), plus_source, n_angle=4
        ).interface_flux
        minus_flux = feautrier_radiation_field(
            optical_depth(minus_opacity), minus_source, n_angle=4
        ).interface_flux
        assert plus_flux is not None and minus_flux is not None
        finite_difference[:, perturbed_depth] = np.sum(
            (plus_flux - minus_flux)
            * trapezoid_weight[:, np.newaxis],
            axis=0,
        ) / (2.0 * step)
    np.testing.assert_allclose(
        response, finite_difference, rtol=1.0e-5, atol=4.0e-6
    )


def test_thick_unresolved_surface_cell_has_thermalized_inward_intensity():
    tau = np.asarray([20.0, 30.0, 50.0])
    source = np.ones((1, tau.size))
    field = radiation_field(tau, source, n_angle=6)
    np.testing.assert_allclose(field.mean_intensity[0, 0], 1.0, rtol=2.0e-9)
    np.testing.assert_allclose(field.flux[0, 0], 0.0, atol=2.0e-8)


def test_diagonal_lambda_matches_direct_source_perturbations():
    tau = np.vstack(
        (
            np.geomspace(1.0e-5, 30.0, 7),
            np.geomspace(0.03, 300.0, 7),
        )
    )
    source = np.asarray(
        (
            (0.7, 0.9, 1.4, 1.2, 2.0, 2.4, 3.1),
            (2.0, 1.7, 1.5, 2.1, 2.7, 2.5, 3.0),
        )
    )
    field = radiation_field(
        tau, source, n_angle=5, calculate_tridiagonal_lambda=True
    )
    assert field.diagonal_lambda is not None
    assert field.subdiagonal_lambda is not None
    assert field.superdiagonal_lambda is not None
    assert np.all((field.diagonal_lambda >= 0.0) & (field.diagonal_lambda <= 1.0))
    step = 1.0e-4
    for depth in range(source.shape[1]):
        perturbed = source.copy()
        perturbed[:, depth] += step
        changed = radiation_field(tau, perturbed, n_angle=5)
        numerical = (
            changed.mean_intensity[:, depth]
            - field.mean_intensity[:, depth]
        ) / step
        np.testing.assert_allclose(
            numerical,
            field.diagonal_lambda[:, depth],
            rtol=2.0e-9,
            atol=2.0e-10,
        )
        if depth + 1 < source.shape[1]:
            numerical_below = (
                changed.mean_intensity[:, depth + 1]
                - field.mean_intensity[:, depth + 1]
            ) / step
            np.testing.assert_allclose(
                numerical_below,
                field.subdiagonal_lambda[:, depth + 1],
                rtol=1.0e-7,
                atol=4.0e-10,
            )
        if depth > 0:
            numerical_above = (
                changed.mean_intensity[:, depth - 1]
                - field.mean_intensity[:, depth - 1]
            ) / step
            np.testing.assert_allclose(
                numerical_above,
                field.superdiagonal_lambda[:, depth - 1],
                rtol=1.0e-7,
                atol=4.0e-10,
            )


def test_integrated_lambda_response_matches_formal_solution_jacobian():
    wavelength = np.asarray([900.0, 1100.0, 1700.0, 3000.0, 7000.0])
    depth_scale = np.geomspace(1.0e-4, 20.0, 6)
    tau = wavelength[:, np.newaxis] ** -0.4 * 20.0 * depth_scale[np.newaxis, :]
    source = 1.0 + 0.2 * np.arange(6)[np.newaxis, :] + wavelength[:, np.newaxis] / 1.0e4
    response_weight = np.broadcast_to(
        0.5 + wavelength[:, np.newaxis] / 8000.0, source.shape
    )
    source_derivative = (
        0.7
        + 0.1 * np.arange(6)[np.newaxis, :]
        + 0.2 * wavelength[:, np.newaxis] / 7000.0
    )
    jacobian, surface_flux_jacobian = integrated_lambda_response(
        tau,
        wavelength,
        response_weight,
        source_derivative,
        n_angle=4,
        wavelength_chunk_size=2,
        return_surface_flux_response=True,
    )
    baseline = radiation_field(tau, source, n_angle=4)
    baseline_surface_flux = emergent_flux(
        tau, source, n_angle=4, backend="python"
    )
    step = 1.0e-4
    for source_depth in range(source.shape[1]):
        perturbed = source.copy()
        perturbed[:, source_depth] += (
            step * source_derivative[:, source_depth]
        )
        changed = radiation_field(tau, perturbed, n_angle=4)
        changed_surface_flux = emergent_flux(
            tau, perturbed, n_angle=4, backend="python"
        )
        numerical = trapezoid(
            response_weight
            * (changed.mean_intensity - baseline.mean_intensity)
            / step,
            wavelength,
            axis=0,
        )
        np.testing.assert_allclose(
            numerical,
            jacobian[:, source_depth],
            rtol=2.0e-9,
            atol=2.0e-8,
        )
        numerical_surface_flux = trapezoid(
            (changed_surface_flux - baseline_surface_flux) / step,
            wavelength,
        )
        np.testing.assert_allclose(
            numerical_surface_flux,
            surface_flux_jacobian[source_depth],
            rtol=2.0e-9,
            atol=2.0e-8,
        )


def test_integrated_emergent_flux_state_response_includes_opacity_changes():
    wavelength = np.asarray([700.0, 1100.0, 1800.0, 3200.0, 7000.0])
    mass = np.geomspace(2.0e-5, 30.0, 7)
    depth = np.arange(mass.size, dtype=np.float64)
    opacity = (
        0.08
        + (wavelength[:, np.newaxis] / 2500.0) ** 0.3
        * (0.5 + 0.08 * depth[np.newaxis, :])
    )
    source = (
        1.0
        + 0.15 * depth[np.newaxis, :]
        + wavelength[:, np.newaxis] / 9000.0
    )
    source_derivative = 0.4 + 0.03 * depth[np.newaxis, :] + 0.0 * wavelength[:, np.newaxis]
    opacity_derivative = 0.025 + 0.004 * depth[np.newaxis, :] + 0.0 * wavelength[:, np.newaxis]

    def optical_depth(values):
        result = np.empty_like(values)
        result[:, 0] = values[:, 0] * mass[0]
        result[:, 1:] = result[:, [0]] + np.cumsum(
            0.5 * (values[:, 1:] + values[:, :-1]) * np.diff(mass),
            axis=1,
        )
        return result

    tau = optical_depth(opacity)
    jacobian = integrated_emergent_flux_state_response(
        tau,
        wavelength,
        source,
        source_derivative,
        mass,
        opacity_derivative,
        n_angle=4,
        wavelength_chunk_size=2,
    )
    baseline = emergent_flux(tau, source, n_angle=4, backend="python")
    step = 2.0e-6
    for state_depth in range(mass.size):
        changed_source = source.copy()
        changed_opacity = opacity.copy()
        changed_source[:, state_depth] += (
            step * source_derivative[:, state_depth]
        )
        changed_opacity[:, state_depth] += (
            step * opacity_derivative[:, state_depth]
        )
        changed = emergent_flux(
            optical_depth(changed_opacity),
            changed_source,
            n_angle=4,
            backend="python",
        )
        numerical = trapezoid((changed - baseline) / step, wavelength)
        np.testing.assert_allclose(
            numerical,
            jacobian[state_depth],
            rtol=3.0e-6,
            atol=2.0e-6,
        )


@pytest.mark.skipif(
    not compiled_backend_available(), reason="optional C extension is not built"
)
def test_c_and_python_backends_agree():
    rng = np.random.default_rng(19)
    tau = np.geomspace(1.0e-7, 1.0e3, 90)
    source = np.exp(rng.normal(size=(17, tau.size)))
    for optical_depth in (
        tau,
        np.ascontiguousarray(
            np.geomspace(0.2, 5.0, source.shape[0])[:, np.newaxis]
            * tau[np.newaxis, :]
        ),
    ):
        expected = emergent_flux(
            optical_depth, source, n_angle=6, backend="python"
        )
        actual = emergent_flux(optical_depth, source, n_angle=6, backend="c")
        np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0.0)
