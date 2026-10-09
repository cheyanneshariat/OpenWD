"""Requested gravity must survive hydrostatics, synthesis and native audits."""
import numpy as np
import pytest
from wd_spectra._cool.check_cool_db_transport_seed import atmosphere_at
from wd_spectra._cool.run_cool_db import commands
from wd_spectra._cool.audit_dense_spectrum import saved_parameters
from wd_spectra._cool.dense_helium_molecular_experiment import run_coordinates


def test_gravity_changes_hydrostatic_mass_not_material_state():
    pressure = np.array([1e6, 1e7, 1e8])
    temperature = np.array([5000., 7000., 10000.])
    tau = np.array([1e-4, .1, 10.])
    a = atmosphere_at(5000., pressure, temperature, tau, logg=8.)
    b = atmosphere_at(5000., pressure, temperature, tau, logg=7.5)
    assert a.logg == 8. and b.logg == 7.5
    np.testing.assert_allclose(a.column_mass * a.gravity, pressure, rtol=1e-14)
    np.testing.assert_allclose(b.column_mass * b.gravity, pressure, rtol=1e-14)
    np.testing.assert_allclose(b.column_mass/a.column_mass, 10**.5, rtol=1e-14)
    np.testing.assert_array_equal(a.mass_density, b.mass_density)
    np.testing.assert_array_equal(a.electron_density, b.electron_density)
    legacy = atmosphere_at(5000., pressure, temperature, tau)
    np.testing.assert_array_equal(a.column_mass, legacy.column_mass)


def test_independent_audit_can_retain_exact_saved_mass_volumes():
    pressure = np.array([1e6, 1e7, 1e8])
    mass = pressure/10**7.75
    a = atmosphere_at(5000., pressure, np.array([5000., 7000., 10000.]),
                      np.array([1e-4, .1, 10.]), logg=7.75, column_mass=mass)
    np.testing.assert_array_equal(a.column_mass, mass)
    assert a.logg == 7.75


@pytest.mark.parametrize('logg', [np.nan, np.inf, -np.inf])
def test_nonfinite_gravity_is_rejected(logg):
    with pytest.raises(ValueError, match='finite'):
        atmosphere_at(5000., np.ones(3), np.full(3,5000.), np.ones(3), logg=logg)


def test_worker_and_audit_commands_share_requested_gravity(tmp_path):
    solve, audit = commands(5000, tmp_path/'out', tmp_path/'table.npz', logg=7.75)
    assert solve[solve.index('--logg')+1] == '7.75'
    assert audit[audit.index('--expected-logg')+1] == '7.75'
    assert solve[solve.index('--max-iterations')+1] == '40'
    assert solve[solve.index('--pseudo-time-sweeps')+1] == '80'


def test_intermediate_parser_preserves_gravity_before_temperature(tmp_path):
    args=run_coordinates(['--logg','7.75','5000','--physical-only','--output-root',str(tmp_path)])
    assert args.temperature == 5000 and args.logg == 7.75 and args.n_depth == 80


def test_audit_rejects_metadata_structure_and_request_disagreement():
    meta = {'effective_temperature':5000., 'logg':7.75, 'atmosphere':{}}
    saved = {'experimental_logg':np.array(7.75), 'experimental_effective_temperature':np.array(5000.)}
    assert saved_parameters(meta, saved, 5000., expected_logg=7.75) == (5000.,7.75)
    with pytest.raises(ValueError, match='requested independent audit'):
        saved_parameters(meta, saved, 5000., expected_logg=8.)
    with pytest.raises(ValueError, match='structure experimental_logg'):
        saved_parameters(meta, {**saved,'experimental_logg':np.array(8.)}, 5000.)
    with pytest.raises(ValueError, match='temperature'):
        saved_parameters(meta, saved, 6000.)
    assert saved_parameters({'atmosphere':{}}, {}, 5000., expected_logg=8.) == (5000.,8.)
