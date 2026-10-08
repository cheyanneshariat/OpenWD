"""Independent quadrature diagnostics; never inherit a solver's local flag."""
import numpy as np
from wd_spectra._energy_balance import discrete_radiative_cell_energy_balance


def thermal_diagnostics(wave,depths,planck,mean,absorption_fraction,convective_flux,*,
                        column_mass=None,absorption=None):
    if (column_mass is None)!=(absorption is None):
        raise ValueError('mass diagnostics require both actual mass and absorption')
    def balance(selection):
        if column_mass is not None:
            from .mass_conservative_feautrier import mass_energy
            return mass_energy(wave[selection],column_mass,planck[selection],mean[selection],absorption[selection])
        return discrete_radiative_cell_energy_balance(wave[selection],depths[selection],
            planck[selection],mean[selection],absorption_fraction[selection])
    exchange,thermal=balance(slice(None))
    conv=np.asarray(convective_flux)
    local=(exchange+np.diff(conv))/(thermal+abs(conv[:-1])+abs(conv[1:]))
    infrared=wave>1e5
    tail=np.zeros_like(thermal)
    if np.sum(infrared)>1:
        _,tail=balance(infrared)
    return dict(maximum_independent_cell_energy_defect=float(np.max(abs(local))),
        worst_independent_energy_node=int(np.argmax(abs(local))),
        surface_independent_cell_energy_defect=float(local[0]),
        surface_thermal_emission_beyond_10_micron_fraction=float(tail[0]/thermal[0]))
