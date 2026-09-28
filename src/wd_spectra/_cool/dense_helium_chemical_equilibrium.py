"""Research chemical-equilibrium kernel with explicit, REQUIRED species data.

He, He+, He2+ (molecular singly charged), e, in a neutral-dominated dense bath.
Bulk rho and chemical potentials must be supplied by a documented provider.
This module does not invent the unavailable He2+--He excess potential.
Definitions follow Kowalski's 2006 thesis, section 4.2.3; no He++ is included.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from wd_spectra import eos
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS
from .dense_helium_fluid_experiment import EV, KB_EV
from .helium_dimer_eos_experiment import dissociation_equilibrium


@dataclass(frozen=True)
class ExcessPotentials:
    """Excess chemical potentials, eV/particle; molecular entry is mandatory."""
    neutral: object
    atomic_ion: object
    molecular_ion: object
    electron: object
    source: str

    def reaction_shifts(self):
        if not self.source or any(v is None for v in (
                self.neutral, self.atomic_ion, self.molecular_ion, self.electron)):
            raise ValueError('all four chemical potentials and their source are required')
        atom, ion, dimer, electron = np.broadcast_arrays(*[
            np.asarray(v, float) for v in (self.neutral, self.atomic_ion,
                                          self.molecular_ion, self.electron)])
        if any(np.any(~np.isfinite(v)) for v in (atom, ion, dimer, electron)):
            raise ValueError('chemical potentials must be finite')
        return ion+electron-atom, dimer-atom-ion


@dataclass(frozen=True)
class DenseHeliumChemistry:
    neutral: np.ndarray
    atomic_ion: np.ndarray
    molecular_ion: np.ndarray
    electron: np.ndarray
    log_ionization_constant: np.ndarray
    log_dissociation_constant: np.ndarray
    maximum_relative_charge_residual: float
    maximum_relative_nuclei_residual: float
    chemical_potential_source: str


def solve_chemistry(temperature, density, potentials, *, equilibrium=None, neutral_partition=1.):
    """Exact nuclear/charge balance and both nonideal mass-action equations.

    Domain guards belong to each provider, but we also reject the trace-fluid
    model outside its <=0.1% ionization approximation. Ideal Stancil K uses
    the existing smooth research interpolation, with no extrapolation here.
    No opacity is evaluated by this algebra kernel.
    """
    t, rho = np.broadcast_arrays(np.asarray(temperature, float), np.asarray(density, float))
    if (np.any(~np.isfinite(t)) or np.any(~np.isfinite(rho)) or np.any(rho <= 0)
            or np.any(t <= 0)):
        raise ValueError('chemical kernel requires positive finite T and density')
    if equilibrium is None:
        if np.any(t < 4200) or np.any(t > 50400):
            raise ValueError('default chemical kernel requires tabulated 4200--50400 K Stancil support')
        equilibrium = dissociation_equilibrium
    ion_shift, diss_shift = potentials.reaction_shifts()
    t, rho, ion_shift, diss_shift = np.broadcast_arrays(t, rho, ion_shift, diss_shift)
    q0=np.asarray(neutral_partition,float)
    if np.any(~np.isfinite(q0)) or np.any(q0 < 1):
        raise ValueError('invalid neutral partition function')
    log_s = (np.log(4./q0)+1.5*np.log(2*np.pi*eos.ELECTRON_MASS*BOLTZMANN*t/eos.PLANCK**2)
             -(eos.HELIUM_FIRST_IONIZATION_ENERGY/EV+ion_shift)/(KB_EV*t))
    ideal_k = np.asarray(equilibrium(t)[0])
    if np.any(~np.isfinite(ideal_k)) or np.any(ideal_k <= 0):
        raise ValueError('invalid supplied dissociation constant')
    log_k = np.log(ideal_k*q0)+diss_shift/(KB_EV*t)
    densities, charge_error, nuclei_error = solve_mass_action(np.log(rho/HELIUM_MASS), log_s, log_k)
    if np.any(densities[3]/(rho/HELIUM_MASS) > 1e-3):
        from .dense_helium_limits import DenseHeliumDomainError
        raise DenseHeliumDomainError('dense trace-ion chemical model exceeded 0.1% ionization')
    return DenseHeliumChemistry(*densities, log_s, log_k, charge_error, nuclei_error, potentials.source)


def solve_mass_action(log_nuclei, log_s, log_k):
    """Algebra-only solver, also tested outside the physical trace-ion regime."""
    log_n, log_s, log_k = np.broadcast_arrays(*[
        np.asarray(v, float) for v in (log_nuclei, log_s, log_k)])
    if any(np.any(~np.isfinite(v)) for v in (log_n, log_s, log_k)):
        raise ValueError('nonfinite mass-action input')

    def species(log_ne):
        f0 = -np.logaddexp(0., log_s-log_ne)
        f1 = -np.logaddexp(0., log_ne-log_s)
        radical = .5*np.logaddexp(0., np.log(8.)+f0+f1+log_n-log_k)
        log_atoms = np.log(2.)+log_n-np.logaddexp(0., radical)
        return log_atoms+f0, log_atoms+f1, f0+f1+2*log_atoms-log_k

    lower = np.minimum(log_n-2, .5*(log_s+log_n)-2)
    upper = log_n.copy()
    def charge_residual(log_ne):
        _, ion, dimer = species(log_ne)
        return log_ne-np.logaddexp(ion, dimer)
    if np.any(charge_residual(lower) >= 0) or np.any(charge_residual(upper) <= 0):
        raise ValueError('failed to bracket chemical charge equilibrium')
    for _ in range(64):
        middle = .5*(lower+upper)
        positive = charge_residual(middle) > 0
        upper = np.where(positive, middle, upper)
        lower = np.where(positive, lower, middle)
    log_ne = .5*(lower+upper)
    logs = species(log_ne)
    charge_error = float(np.max(abs(np.expm1(charge_residual(log_ne)))))
    log_sum = logsumexp(np.stack((logs[0], logs[1], logs[2]+np.log(2.))), axis=0)
    nuclei_error = float(np.max(abs(np.expm1(log_sum-log_n))))
    if max(charge_error, nuclei_error) > 1e-10:
        raise ValueError('chemical conservation failed')
    return tuple(np.exp(v) for v in (*logs, log_ne)), charge_error, nuclei_error
