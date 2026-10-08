"""Bound rovibrational sum from the published Chang He2+ potential.

As in Stancil (1994), solve the radial nuclear Schroedinger equation, rather
than extrapolating a short K(T) table. Ground 4He2+ has odd rotational N only
(spin-zero nuclei; ungerade electronic state). This potential is not Stancil's
and agreement with its equilibrium table must be checked, not forced.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.special import logsumexp
from wd_spectra.constants import BOLTZMANN, HELIUM_MASS, ELECTRON_MASS, PLANCK
from .chang_helium_potential import dimer_ground_hartree, HARTREE_EV
from .dense_helium_fluid_experiment import KB_EV


@dataclass(frozen=True)
class DimerBoundStates:
    energy_ev: np.ndarray
    rotational_quantum_number: np.ndarray
    source: str = 'bound odd-N 4He2+ levels; Chang 2002 QCISD(T) potential; no quasibound levels'
    minimum_temperature: float = 1000.
    maximum_temperature: float = 17000.

    def __post_init__(self):
        e,n=np.asarray(self.energy_ev),np.asarray(self.rotational_quantum_number)
        if (e.ndim != 1 or not e.size or n.shape != e.shape
                or np.any(~np.isfinite(e)) or np.any(e >= 0)
                or np.any(~np.isfinite(n)) or np.any(n < 1) or np.any(n % 2 != 1)
                or not self.source or self.minimum_temperature <= 0
                or self.maximum_temperature <= self.minimum_temperature):
            raise ValueError('invalid molecular bound-state data')

    def equilibrium(self, temperature):
        t=np.asarray(temperature,float)
        if np.any(~np.isfinite(t)) or np.any(t < self.minimum_temperature) or np.any(t > self.maximum_temperature):
            raise ValueError('rovibrational chemical experiment outside tested T domain')
        # Electronic doublet degeneracy cancels against atomic He+ doublet.
        energies=self.energy_ev
        log_weights=np.log(2*self.rotational_quantum_number+1)-energies/(KB_EV*t[...,None])
        log_q=logsumexp(log_weights,axis=-1)
        probability=np.exp(log_weights-log_q[...,None])
        mean_energy=np.sum(probability*energies,axis=-1)
        trans=1.5*np.log(2*np.pi*(HELIUM_MASS/2)*BOLTZMANN*t/PLANCK**2)
        return np.exp(trans-log_q), 1.5-mean_energy/(KB_EV*t)


def calculate_bound_states(points=16383, extent=40.):
    """Dirichlet radial FD Hamiltonian; verify by grid and box refinement."""
    if points < 1000 or extent < 15.:
        raise ValueError('radial domain/resolution too small for this experiment')
    lower=.4
    dr=(extent-lower)/(points+1)
    r=lower+np.arange(1,points+1)*dr
    reduced_mass=HELIUM_MASS/(2*ELECTRON_MASS)
    kinetic=1/(2*reduced_mass*dr**2)
    potential=dimer_ground_hartree(r)
    off=np.full(points-1,-kinetic)
    all_energies,all_n=[],[]
    for n in range(1,201,2):
        effective=potential+n*(n+1)/(2*reduced_mass*r*r)
        if effective.min() >= 0:
            break
        levels=eigh_tridiagonal(2*kinetic+effective,off,eigvals_only=True,
            select='v',select_range=(effective.min(),0.))
        all_energies.extend(levels*HARTREE_EV)
        all_n.extend([n]*len(levels))
    else:
        raise RuntimeError('rotational bound-state sum did not terminate')
    return DimerBoundStates(np.asarray(all_energies),np.asarray(all_n))
