import numpy as np
import pytest
from dense_helium_chemical_equilibrium import (ExcessPotentials, solve_mass_action, solve_chemistry,
                                              solve_solute_chemistry)
from dense_helium_atomic_experiment import atomic_ion_fraction


def test_full_mass_action_charge_and_nuclei_over_extreme_molecular_ratios():
    n = np.full((9, 7), 1e22)
    s = np.geomspace(1e-15, 1e27, 9)[:, None]
    k = np.geomspace(1e-5, 1e50, 7)[None, :]
    (n0,n1,nd,ne), charge_error, nuclei_error = solve_mass_action(np.log(n),np.log(s),np.log(k))
    assert charge_error < 1e-12 and nuclei_error < 1e-12
    np.testing.assert_allclose(n0+n1+2*nd, n, rtol=1e-12)
    np.testing.assert_allclose(n1+nd, ne, rtol=1e-12)
    np.testing.assert_allclose(np.log(n1)+np.log(ne)-np.log(n0),np.broadcast_to(np.log(s),n.shape),atol=5e-14)
    np.testing.assert_allclose(np.log(n0)+np.log(n1)-np.log(nd),np.broadcast_to(np.log(k),n.shape),atol=5e-14)


def test_molecular_free_limit_recovers_exact_atomic_solution():
    logs = np.linspace(-100, 20, 40)
    (n0,n1,nd,ne), *_ = solve_mass_action(0., logs, 600.)
    np.testing.assert_allclose(ne, atomic_ion_fraction(logs), rtol=2e-14)
    assert nd.max() < 1e-200


def test_chemical_potentials_obey_nuclear_reference_invariance():
    original = ExcessPotentials(.3, -.8, 1.2, .6, 'synthetic algebra test')
    shifted = ExcessPotentials(7.3, 6.2, 15.2, .6, 'synthetic algebra test')
    np.testing.assert_allclose(original.reaction_shifts(), shifted.reaction_shifts(), atol=2e-15)
    a = solve_chemistry(6000., .1, original)
    b = solve_chemistry(6000., .1, shifted)
    np.testing.assert_allclose(a.electron, b.electron, rtol=3e-14)


def test_no_missing_dimer_potential_or_unmarked_source_allowed():
    with pytest.raises(ValueError, match='all four'):
        ExcessPotentials(0., 0., None, .5, 'incomplete').reaction_shifts()
    with pytest.raises(ValueError, match='all four'):
        ExcessPotentials(0., 0., 0., .5, '').reaction_shifts()
    with pytest.raises(ValueError, match='tabulated'):
        solve_chemistry(3000., .1, ExcessPotentials(0., 0., 0., 0., 'unit test'))


def test_electron_insertion_suppresses_ionization_not_enhances_it():
    original = solve_chemistry(6000., .1, ExcessPotentials(0., 0., 0., 0., 'unit test'))
    extra_electron_energy = solve_chemistry(6000., .1, ExcessPotentials(0., 0., 0., 1., 'unit test'))
    assert extra_electron_energy.electron < original.electron


def test_solute_chemistry_conserves_and_reduces_to_atomic_without_bound_pairs():
    potentials = ExcessPotentials(.3, -.8, 1.2, .6, 'synthetic algebra test')
    rho = np.geomspace(1e-6, 1., 7)
    free = solve_solute_chemistry(6000., rho, potentials, 0.)
    explicit = solve_chemistry(6000., rho, potentials, equilibrium=lambda t: (np.full_like(t, 1e300),))
    np.testing.assert_allclose(free.electron, explicit.electron, rtol=1e-12)
    assert np.all(free.molecular_ion == 0)
    bound = solve_solute_chemistry(6000., rho, potentials, 3*rho)
    np.testing.assert_allclose(bound.electron, free.electron, rtol=1e-12)
    np.testing.assert_allclose(bound.molecular_ion/bound.electron, -np.expm1(-3*rho), rtol=1e-12)
    from wd_spectra.constants import HELIUM_MASS
    nuclei = rho/HELIUM_MASS
    np.testing.assert_allclose(bound.neutral+bound.atomic_ion+2*bound.molecular_ion, nuclei, rtol=1e-14)
    assert bound.maximum_relative_charge_residual < 1e-14
