"""Temperature dependence of He2+ photodissociation below Stancil's 4200 K edge.

Stancil (1994) tabulates He2+ X 2Sigma_u+ -> A 2Sigma_g+ bound-free cross
sections only from 4200 K; cool dense-He layers (1000-4200 K) previously held
the 4200 K values, which overestimates the optical cross section there by up
to ~15-50x (fewer excited vibrational levels) and puts a slope discontinuity
in a dominant opacity. This computes sigma(lambda, T) from potentials:

  bound X levels  Chang (2002) potential, odd N (4He2+), FD Hamiltonian as in
                  _cool/helium_dimer_state_sum.py;
  continuum A     Bruno et al. (2010) antibonding curve plus the same
                  -alpha/(2R^4) polarization tail as Chang's X curve,
                  energy-normalized (Numerov);
  dipole          charge-resonance R/2 (exact asymptotically);
  sigma(omega) = 4 pi^2 omega / (3 c) |<E,N|d|v,N>|^2 (atomic units),
  Boltzmann-averaged over the bound levels.

Against Stancil at 4200-12600 K this gives the same spectral shape and is
1.1-1.4x higher at 1500 A - 2 um (likely the R/2 dipole at short R). Only its
temperature DEPENDENCE is used: the shipped table is
log[sigma(lambda, T)/sigma(lambda, 4200 K)] for 600-4200 K, applied to
Stancil's 4200 K values, so the opacity stays continuous with Stancil's table.

usage: python tools/build_he2_photodissociation.py OUTPUT.npz
"""
import json
import sys
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal

from wd_spectra.constants import HELIUM_MASS, ELECTRON_MASS
from wd_spectra import helium_molecular as molecular
from wd_spectra._cool.chang_helium_potential import dimer_ground_hartree, HARTREE_EV, BOHR_ANGSTROM

C_AU = 137.035999
A0_CM2 = (0.529177210903e-8)**2
MU = HELIUM_MASS/(2*ELECTRON_MASS)
POLARIZABILITY = 1.3793          # He dipole polarizability, a0^3 (same as Chang's tail)


def upper_hartree(r):
    ra = r*BOHR_ANGSTROM
    bruno = 359.*np.exp(-4.184*ra+.649*ra**2-.08528*ra**3)/HARTREE_EV
    return bruno - POLARIZABILITY/(2*r**4)


def bound_levels(r, dr):
    kinetic = 1/(2*MU*dr**2)
    v = dimer_ground_hartree(r)
    off = np.full(len(r)-1, -kinetic)
    out = []
    for n in range(1, 201, 2):
        eff = v+n*(n+1)/(2*MU*r*r)
        if eff.min() >= 0:
            break
        w, vec = eigh_tridiagonal(2*kinetic+eff, off, select='v', select_range=(eff.min(), 0.))
        for e, psi in zip(w, vec.T):
            out.append((n, e, psi/np.sqrt(dr)))       # int psi^2 dr = 1
    return out


def continuum(r, dr, energies, n):
    """Energy-normalized regular solutions of the A curve (Numerov), shape (len(E), len(r))."""
    veff = upper_hartree(r)+n*(n+1)/(2*MU*r*r)
    E = np.asarray(energies)[:, None]
    q = 2*MU*(E-veff[None, :])                       # psi'' = -q psi
    f = 1+dr*dr*q/12
    psi = np.zeros((len(energies), len(r)))
    psi[:, 1] = 1e-30
    for i in range(1, len(r)-1):
        psi[:, i+1] = ((12-10*f[:, i])*psi[:, i]-f[:, i-1]*psi[:, i-1])/f[:, i+1]
        big = np.abs(psi[:, i+1]) > 1e200
        if np.any(big):
            psi[big, :i+2] *= 1e-200
    # Amplitude from the last stretch where the potential is negligible:
    k = np.sqrt(2*MU*energies)
    tail = slice(len(r)-400, len(r))
    a = psi[:, tail]; x = r[tail]
    # psi ~ A sin(kx+delta): A^2 = psi^2 + (psi'/k)^2 (WKB-local, potential ~0)
    dpsi = np.gradient(a, x, axis=1)
    amp = np.sqrt(np.median(a**2+(dpsi/k[:, None])**2, axis=1))
    return psi*(np.sqrt(2*MU/(np.pi*k))/amp)[:, None]


KB_HARTREE = 3.166811563e-6


def thermal_average(energy, rotational, sigma, temperature):
    log_w = np.log(2*rotational+1)-energy/(KB_HARTREE*temperature)
    p = np.exp(log_w-log_w.max())
    return (p/p.sum())@sigma


def main():
    output = sys.argv[1]
    lower, extent, points = 0.6, 60., 12000
    dr = (extent-lower)/(points+1)
    r = lower+np.arange(1, points+1)*dr
    started = time.monotonic()
    levels = bound_levels(r, dr)
    wave = molecular._WAVELENGTH_BOUND_FREE
    photon = 1e8/wave/219474.6313
    dipole = r/2
    by_n = {}
    for n, e, psi in levels:
        by_n.setdefault(n, []).append((e, psi))
    energy, rotational, sigma = [], [], []
    for n, group in by_n.items():
        energies, owners = [], []
        for idx, (e, psi) in enumerate(group):
            ok = np.nonzero(photon+e > 1e-6)[0]
            energies.extend(photon[ok]+e); owners.extend((idx, j) for j in ok)
        chi = continuum(r, dr, np.array(energies), n) if energies else None
        rows = np.zeros((len(group), len(wave)))
        for row, (idx, j) in enumerate(owners):
            m = np.dot(chi[row], dipole*group[idx][1])*dr
            rows[idx, j] = 4*np.pi**2*photon[j]/(3*C_AU)*m**2*A0_CM2
        for idx, (e, psi) in enumerate(group):
            energy.append(e); rotational.append(n); sigma.append(rows[idx])
    energy, rotational, sigma = np.array(energy), np.array(rotational), np.array(sigma)
    temperature = np.geomspace(600., 4200., 49)
    reference = thermal_average(energy, rotational, sigma, 4200.)
    log_ratio = np.stack([np.log(np.maximum(thermal_average(energy, rotational, sigma, t), 1e-300)/reference)
                          for t in temperature], axis=1)
    validation = {f"{t:.0f}": [float(x) for x in thermal_average(energy, rotational, sigma, t)/molecular._BOUND_FREE[:, k]]
                  for k, t in enumerate(molecular._TEMPERATURE[:4])}
    metadata = dict(
        description="log sigma(lambda,T)/sigma(lambda,4200 K), He2+ X->A photodissociation, 600-4200 K",
        sources=dict(bound="Chang 2002 X potential (odd N 4He2+)", continuum="Bruno et al. 2010 A curve + -alpha/2R^4",
                     dipole="charge resonance R/2", reference="applied to Stancil 1994 at 4200 K"),
        bound_levels=int(len(energy)), radial_points=points, radial_extent_bohr=extent,
        computed_over_stancil=validation, parameters_fitted=False)
    np.savez_compressed(output, wavelength_angstrom=wave, temperature=temperature, log_ratio=log_ratio,
                        metadata_json=json.dumps(metadata))
    print(f"wrote {output}: {len(energy)} levels in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
