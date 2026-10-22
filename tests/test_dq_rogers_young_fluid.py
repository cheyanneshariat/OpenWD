"""Rogers-Young dense-He structure: closure limits and physical consistency."""
import numpy as np
import pytest

from wd_spectra.constants import BOLTZMANN, HELIUM_MASS
from wd_spectra._cool.dense_helium_fluid_experiment import solve_neutral_py
from wd_spectra._dq.dq_rogers_young_fluid import (
    RadialGrid, solve_consistent_rogers_young, solve_rogers_young,
)

GRID = RadialGrid(1023, 24.)


def _n(rho):
    return rho/HELIUM_MASS*1e-24


def test_small_alpha_is_percus_yevick():
    n = _n(.8)
    py = solve_neutral_py(GRID, 4000., n)
    ry, _, _ = solve_rogers_young(GRID, 4000., n, 1e-7, py.gamma)
    np.testing.assert_allclose(ry.structure, py.structure, rtol=1e-5)


def _march(t, rho_top, steps=40):
    gamma, alpha = None, .1
    for rho in np.geomspace(.01, rho_top, steps):
        fluid = solve_consistent_rogers_young(GRID, t, _n(rho), gamma, alpha)
        gamma, alpha = fluid.state.gamma, fluid.alpha
    return fluid


@pytest.mark.parametrize("rho", [1.0, 1.778])
def test_consistent_alpha_equates_virial_and_fluctuation_compressibility(rho):
    fluid = _march(3000., rho)
    assert fluid.structure_zero == pytest.approx(fluid.virial_structure_zero, rel=2e-3)
    # Ab initio He-REOS.3 compressibility, S(0)=(kT/m)(d rho/dP)_T, from the
    # tabulated 3000 K isotherm: 0.088 at 1.0 and 0.040 at 1.778 g/cm3.
    # Percus-Yevick gives 0.066 and 0.013 (3x low at 1.778); HNC 0.098, 0.039.
    reos = {1.0: .088, 1.778: .040}[rho]
    py = {1.0: .0662, 1.778: .0132}[rho]
    assert abs(fluid.structure_zero/reos-1) < .25
    assert abs(fluid.structure_zero/reos-1) < abs(py/reos-1)


def test_dilute_states_are_closure_independent():
    fluid = solve_consistent_rogers_young(GRID, 6000., _n(.01))
    py = solve_neutral_py(GRID, 6000., _n(.01))
    np.testing.assert_allclose(fluid.state.structure, py.structure, rtol=2e-4)


def test_reos_compressibility_reference_values():
    """Guard the REOS numbers quoted above against the shipped EOS table."""
    from pathlib import Path
    from wd_spectra.dense_eos import read_helium_reos3_table
    path = (Path(__file__).resolve().parents[1]
            / "src/wd_spectra/data/runtime/cache/dense-eos/he-reos3-table3.dat")
    if not path.exists():
        pytest.skip("REOS table not present in this checkout")
    table = read_helium_reos3_table(path)
    i = int(np.flatnonzero(table.temperature_grid == 3000.)[0])
    rho, p = table.density_by_temperature[i], table.pressure_by_temperature[i]
    s0 = BOLTZMANN*3000./HELIUM_MASS*rho/p*np.gradient(np.log(rho), np.log(p))
    np.testing.assert_allclose(np.interp([1.0, 1.778], rho, s0), [.0878, .0398], rtol=2e-2)
