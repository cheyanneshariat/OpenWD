import numpy as np
import pytest
from dense_helium_fluid_experiment import (
    RadialGrid, neutral_pair_ev, ion_pair_ev, electron_insertion_ev, bound_neighbour_count,
    solve_neutral_py, solve_trace_py, solve_hnc, KB_EV,
)


def test_spherical_transform_roundtrip_and_absolute_normalization():
    grid = RadialGrid(1023, 24.)
    a = .7
    f = np.exp(-a*grid.r**2)
    expected = (np.pi/a)**1.5*np.exp(-grid.k**2/(4*a))
    np.testing.assert_allclose(grid.forward(f), expected, atol=3e-14)
    np.testing.assert_allclose(grid.inverse(grid.forward(f)), f, atol=2e-14)
    np.testing.assert_allclose(grid.zero(f), (np.pi/a)**1.5, rtol=1e-13)


def test_published_pair_minima_and_separate_channels():
    np.testing.assert_allclose(neutral_pair_ev(2.9673), -10.8*KB_EV)
    assert np.isinf(neutral_pair_ev(.01))
    pair = ion_pair_ev(1.081)
    np.testing.assert_allclose(pair[0], -2.4730)
    assert pair[1] > 0


def test_zero_density_fluid_is_boltzmann_and_trace_is_virial():
    grid = RadialGrid()
    state = solve_neutral_py(grid, 5000., 0.)
    np.testing.assert_array_equal(state.structure, 1.)
    np.testing.assert_allclose(state.g, np.exp(-neutral_pair_ev(grid.r)/(5000*KB_EV)), atol=1e-15)
    for curve in ion_pair_ev(grid.r):
        c = solve_trace_py(grid, 5000., state, curve)
        np.testing.assert_allclose(c, np.expm1(-curve/(5000*KB_EV)), atol=1e-13)


def test_identical_trace_particle_recovers_neutral_correlations():
    grid = RadialGrid()
    state = solve_neutral_py(grid, 6000., .02)
    c = solve_trace_py(grid, 6000., state, neutral_pair_ev(grid.r))
    np.testing.assert_allclose(c, state.direct, rtol=2e-8, atol=2e-9)
    assert state.residual < 2e-8


def test_electron_fit_is_zero_at_zero_density_and_has_lenz_limit():
    rho = np.array([0., 1e-9, 1e-8])
    v = electron_insertion_ev(5000., rho)
    assert v[0] == 0
    np.testing.assert_allclose(v[1:]/rho[1:], 4.5382, rtol=1e-6)
    np.testing.assert_allclose(electron_insertion_ev(5000., .05-1e-10),
                               electron_insertion_ev(5000., .05+1e-10), rtol=1e-8)
    with pytest.raises(ValueError, match='outside'):
        electron_insertion_ev(5000., 20.)


def test_hnc_identical_trace_and_zero_density_limits():
    grid = RadialGrid()
    v = neutral_pair_ev(grid.r)
    state, mu = solve_hnc(grid, 5000., 0., v)
    assert mu == 0
    np.testing.assert_allclose(state.g, np.exp(-v/(5000*KB_EV)), atol=1e-15)
    bulk, mu_bulk = solve_hnc(grid, 6000., .02, v)
    trace, mu_trace = solve_hnc(grid, 6000., .02, v, solvent=bulk)
    np.testing.assert_allclose(trace.direct, bulk.direct, atol=2e-9)
    np.testing.assert_allclose(mu_trace, mu_bulk, rtol=2e-9)
    dilute, mu_dilute = solve_hnc(grid, 6000., 1e-10, v)
    virial = -1e-10*KB_EV*6000*grid.zero(np.expm1(-v/(6000*KB_EV)))
    np.testing.assert_allclose(mu_dilute, virial, rtol=1e-7)


def test_hnc_mesh_and_box_convergence_for_trace_ion():
    potentials = []
    for points, extent in ((1023, 24.), (2047, 24.), (2047, 48.)):
        grid = RadialGrid(points, extent)
        # Density continuation is an initial guess only, never a substitute
        # for evaluating the requested state's own OZ/closure residual.
        bulk, trace = None, None
        for density in np.linspace(0., .12, 13):
            bulk, mu0 = solve_hnc(grid, 6000., density, neutral_pair_ev(grid.r),
                                 initial=None if bulk is None else bulk.gamma)
            trace, mu1 = solve_hnc(grid, 6000., density, ion_pair_ev(grid.r),
                solvent=bulk, initial=None if trace is None else trace.gamma)
        assert trace.g.min() >= 0
        assert trace.residual < 2e-8
        potentials.append([mu0, mu1])
    np.testing.assert_allclose(potentials[1:], [potentials[0], potentials[0]], atol=1e-4)


@pytest.mark.parametrize('temperature,density', [(0, 0.1), (5000, -1), (np.nan, .1), (5000, np.nan)])
def test_hnc_rejects_invalid_state(temperature, density):
    grid = RadialGrid()
    with pytest.raises(ValueError, match='invalid HNC'):
        solve_hnc(grid, temperature, density, neutral_pair_ev(grid.r))


def test_bound_neighbours_reduce_to_the_chang_association_constant():
    from helium_dimer_state_sum import calculate_bound_states
    from wd_spectra.constants import HELIUM_MASS
    grid = RadialGrid()
    levels = calculate_bound_states()
    for temperature in (5000., 7000., 10000.):
        n = 1e-5/(HELIUM_MASS*1e24)
        neutral, _ = solve_hnc(grid, temperature, n, neutral_pair_ev(grid.r))
        ion, _ = solve_hnc(grid, temperature, n, ion_pair_ev(grid.r), solvent=neutral)
        expected = n*1e24/levels.equilibrium(np.array([temperature]))[0][0]
        np.testing.assert_allclose(bound_neighbour_count(grid, temperature, n, ion.gamma), expected, rtol=.06)
