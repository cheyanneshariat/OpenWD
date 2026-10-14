import numpy as np
import pytest
from wd_spectra import atmosphere
from extended_thermal_wavelength_experiment import extend_grid,extended_quadrature


def test_existing_nodes_are_bitwise_preserved():
    old=np.geomspace(100,1e5,600)
    new=extend_grid(old,1e8)
    np.testing.assert_array_equal(new[:len(old)],old)
    assert new[-1]==1e8
    assert np.max(np.diff(np.log(new[len(old)-1:])))<=np.log(old[-1]/old[-2])*(1+1e-11)


def test_extension_is_scoped_and_restored_after_error():
    original=atmosphere.np
    with pytest.raises(RuntimeError):
        with extended_quadrature(1e8):
            assert atmosphere.np.geomspace(100,1e5,600)[-1]==1e8
            np.testing.assert_array_equal(atmosphere.np.geomspace(1,100,10),np.geomspace(1,100,10))
            assert np.geomspace(100,1e5,600)[-1]==1e5
            raise RuntimeError('probe')
    assert atmosphere.np is original


def test_invalid_extent_rejected():
    with pytest.raises(ValueError):extend_grid(np.geomspace(100,1e5,600),1e5)


@pytest.mark.parametrize('temperature', [5000., 8000.])
@pytest.mark.parametrize('mass_conservative', [False, True])
def test_driver_forwards_transfer_choice_through_helium_solver(
        monkeypatch, temperature, mass_conservative):
    import sys
    from wd_spectra import adaptive_structure as adaptive, gray_helium_atmosphere
    from wd_spectra._cool import (
        check_cool_db_transport_seed as runner,
        dense_helium_molecular_experiment as experiment,
        extended_thermal_wavelength_experiment as driver,
        stable_feautrier_experiment as stable,
    )

    seed = gray_helium_atmosphere(temperature, 8., n_depth=8)
    monkeypatch.setattr(atmosphere, 'helium_continuum_atmosphere',
                        lambda *args, **kwargs: seed)
    calls = []

    def solve(initial, wavelength, **options):
        calls.append((initial, wavelength, options))
        return initial

    monkeypatch.setattr(adaptive, 'solve_adaptive_lte_structure', solve)

    def run_worker():
        # Exercise the real public helium forwarding and scoped transfer
        # guard, replacing only the expensive iteration and continuum seed.
        with stable.stable_transfer_experiment():
            result = runner.radiative_equilibrium_helium_atmosphere(
                temperature, 8., stark_table=object(), n_depth=8,
                n_continuum_wavelength=80, include_lines=False,
                include_helium_ii_lines=False, mixing_length_alpha=None)
        assert result is calls[-1][0]

    monkeypatch.setattr(experiment, 'main', run_worker)
    arguments = ['extended-thermal', '--direct-spectrum', '--stable-transfer',
                 '--structure-angles', '8']
    if mass_conservative:
        arguments.append('--mass-conservative-transfer')
    monkeypatch.setattr(sys, 'argv', arguments)
    driver.main()

    assert len(calls) == 1
    initial, wavelength, options = calls[0]
    np.testing.assert_array_equal(initial.column_mass, seed.column_mass)
    np.testing.assert_array_equal(initial.temperature, seed.temperature)
    assert wavelength[-1] == 1e8
    assert options['n_angle'] == 8
    assert options['transfer_discretization'] == (
        'column-mass' if mass_conservative else 'optical-depth')
    assert options['metadata'].get('experimental_mass_conservative_transfer', False) == mass_conservative
    assert adaptive.solve_adaptive_lte_structure is solve
