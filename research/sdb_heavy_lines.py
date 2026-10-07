"""LTE optical line opacity of trans-iron elements (Z > 30) for the sdB driver.

The package atomic database (Stout) stops at zinc, but the intermediate
He-sdOBs LS IV-14 116 and Feige 46 show strong lines of Ge, Sr, Y, Zr, Sn and
other heavy elements (Dorsch et al. 2020, who also treated Z > 30 in LTE).
This module adds their optical lines to the fixed sdB background:

- ion populations: Saha ladder at the host's (unchanged) electron density,
  NIST ionization energies, Kurucz partition-function tables where available,
  otherwise the ground-level statistical weight (documented per ion);
- level populations: Boltzmann within each ion (LTE), source function B;
- profiles: Voigt with thermal Doppler width and Kurucz radiative and
  quadratic-Stark widths (log Gamma per electron at 10^4 K, scaled as
  T^(1/6)); missing Kurucz widths fall back to the classical estimates;
- line data: Kurucz atomic line lists (positions with laboratory energies),
  plus explicitly cited literature lines (LITERATURE_LINES).

Only lines in DISPLAY_RANGE are kept: the purpose is the optical spectrum, not
blanketing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import wofz

from wd_spectra.light_metal_nlte import _classical_electron_stark_rate_per_electron

ROOT = Path(__file__).resolve().parents[1]
KURUCZ = ROOT / 'results/sdb/atomic/kurucz'
DISPLAY_RANGE = (3600.0, 7100.0)  # vacuum A
MINIMUM_LOG_GF = -3.0
LIGHT_SPEED = 2.99792458e10
PLANCK = 6.62607015e-27
BOLTZMANN = 1.380649e-16
ELECTRON_MASS = 9.1093837015e-28
ATOMIC_MASS_UNIT = 1.66053906660e-24
EV = 1.602176634e-12
WAVENUMBER_TO_ERG = PLANCK * LIGHT_SPEED
CLASSICAL_CROSS_SECTION = 0.02654008  # pi e^2 / (m_e c), cm^2 Hz

# NIST ASD ionization energies (eV) and ground-level statistical weights, for
# each stage in the Saha ladder; 'kurucz' names the partition-function table.
# Ladders start at the lowest stage with a non-negligible population at
# 20-40 kK and end one stage above the highest stage with lines.
ELEMENTS = {
    'Sr': dict(mass=87.62, stages={1: dict(ionization=11.0302765, kurucz='3801'),
                                   2: dict(ionization=42.88353, kurucz='3802'),
                                   3: dict(ionization=56.280, ground_weight=4.0)}),  # 4p5 2P3/2
    'Zr': dict(mass=91.224, stages={2: dict(ionization=23.170, kurucz='4002'),
                                    3: dict(ionization=34.41836, kurucz='4003'),
                                    4: dict(ionization=80.348, ground_weight=1.0)}),  # 4p6 1S0
    # Ge III 4s2 1S0 (first excited level 4s4p 3P0 at 7.0 eV), Ge IV 4s 2S1/2
    # (4p at 9.1 eV), Ge V 3d10 1S0.
    'Ge': dict(mass=72.630, stages={2: dict(ionization=34.0576, ground_weight=1.0),
                                    3: dict(ionization=45.7155, ground_weight=2.0),
                                    4: dict(ionization=90.500, ground_weight=1.0)}),
    # Y III: NIST levels 4d 2D3/2 (0), 2D5/2 (724.15 cm-1), 5s 2S1/2 (7467.1 cm-1);
    # Y IV 4p6 1S0; Y V 4p5 2P3/2.
    # Supplementary N II 3d-4f lines (absent from the Stout N II atom, whose
    # NLTE solution supplies all other N lines): LTE ladder, NIST levels
    # N II 2p2 3P0,1,2 / 1D2 / 1S0 / 2p3 5S2, N III 2p 2P / 4P, N IV 2s2 1S0 / 2s2p 3P.
    'N': dict(mass=14.007, supplementary=('0701', 'gf0701.all', 1, '4f'),
              stages={1: dict(ionization=29.6013, levels=((0.0, 1.0), (48.7, 3.0), (130.8, 5.0), (15316.2, 5.0),
                                                           (32688.8, 1.0), (46784.6, 5.0))),
                      2: dict(ionization=47.4453, levels=((0.0, 2.0), (174.4, 4.0), (57190.0, 12.0))),
                      3: dict(ionization=77.4735, levels=((0.0, 1.0), (67272.0, 9.0))),
                      4: dict(ionization=97.8901, ground_weight=2.0)}),
    # Sn III 4d10 5s2 1S0; Sn IV 5s 2S1/2 (5p at 8.6 eV); Sn V 4d10 1S0.
    'Sn': dict(mass=118.710, stages={2: dict(ionization=30.506, ground_weight=1.0),
                                     3: dict(ionization=40.74, ground_weight=2.0),
                                     4: dict(ionization=77.03, ground_weight=1.0)}),
    'Y': dict(mass=88.906, stages={2: dict(ionization=20.52441, levels=((0.0, 4.0), (724.15, 6.0), (7467.1, 2.0))),
                                   3: dict(ionization=60.6072, ground_weight=1.0),
                                   4: dict(ionization=75.35, ground_weight=4.0)}),
}


def air_to_vacuum(air):
    vacuum = np.asarray(air, dtype=float).copy()
    for _ in range(3):
        s2 = (1e4 / vacuum) ** 2
        vacuum = air * (1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2))
    return vacuum


def kurucz_partition(code):
    """log10 T and U(T) (first potential-lowering column, 500 cm-1)."""
    log_t, values = [], []
    for line in (KURUCZ / f'partfn{code}.dat').read_text().splitlines()[3:]:
        parts = line.split()
        if len(parts) >= 4:
            log_t.append(float(parts[1])); values.append(float(parts[3]))
    return np.asarray(log_t), np.asarray(values)


def kurucz_lines(code, charge, filename=None, upper_label=None):
    """Kurucz .pos/.all lines (gfall fixed format) inside DISPLAY_RANGE with log gf above the floor.

    ``upper_label`` keeps only lines whose upper-level label contains it (used
    to add only the transitions a Stout atom lacks).
    """
    lines = []
    for row in (KURUCZ / (filename or f'gf{code}.pos')).read_text().splitlines():
        if len(row) < 98:
            continue
        nm, log_gf = float(row[0:11]), float(row[11:18])
        e1, j1, e2, j2 = abs(float(row[24:36])), float(row[36:41]), abs(float(row[52:64])), float(row[64:69])
        gamma_rad, gamma_stark = float(row[80:86]), float(row[86:92])
        wavelength = nm * 10.0
        if wavelength > 2000.0:  # Kurucz: air above 200 nm
            wavelength = float(air_to_vacuum(wavelength))
        if not DISPLAY_RANGE[0] <= wavelength <= DISPLAY_RANGE[1] or log_gf < MINIMUM_LOG_GF:
            continue
        (e_low, j_low, label_low), (e_up, _, label_up) = sorted(((e1, j1, row[42:52]), (e2, j2, row[70:80])))
        if upper_label is not None and upper_label not in label_up:
            continue
        lines.append(dict(charge=charge, wavelength=wavelength, log_gf=log_gf, lower_energy=e_low,
                          lower_weight=2 * j_low + 1, upper_energy=e_up,
                          log_gamma_rad=gamma_rad if gamma_rad != 0 else None,
                          log_gamma_stark=gamma_stark if gamma_stark != 0 else None, source=f'Kurucz gf{code}.pos'))
    return lines


# Lines without Kurucz data, from the sources Dorsch et al. (2020, Table 4) used.
# Each entry: element, charge, vacuum A, log gf, lower level energy (cm-1), lower g, upper energy, source.
_NASLIM2011 = 'Naslim et al. (2011, MNRAS 412, 363) Table 3'
_BISWAS2018 = 'Biswas et al. (2018, arXiv:1802.08893) Table 3, f(RCC); 6s at 174138 cm-1 from NIST 5p-6s'
# Laboratory air wavelength -> observed air wavelength in LS IV-14 116 and
# Feige 46 (Dorsch et al. 2020, Table 5): positions only, not strengths.
POSITION_UPDATES = {('Sr', 2): {3976.706: 3976.033, 3991.587: 3992.272}, ('Ge', 2): {4178.960: 4179.078},
                    ('Y', 2): {4039.602: 4039.576}, ('Zr', 3): {5462.333: 5462.380, 5779.843: 5779.880},
                    ('Sn', 3): {3862.051: 3861.207, 4217.184: 4216.192}}


def updated_vacuum(element, charge, vacuum):
    air = vacuum / (1 + 0.0000834254 + 0.02406147 / (130 - (1e4 / vacuum) ** 2)
                    + 0.00015998 / (38.9 - (1e4 / vacuum) ** 2))
    for lab, observed in POSITION_UPDATES.get((element, charge), {}).items():
        if abs(air - lab) < 0.05:
            return float(air_to_vacuum(observed))
    return vacuum
_EV_TO_CM = 8065.543937
LITERATURE_LINES: list[dict] = [
    dict(element=element, charge=charge, wavelength=float(air_to_vacuum(air)), log_gf=log_gf,
         lower_energy=lower_ev * _EV_TO_CM, lower_weight=lower_weight,
         upper_energy=lower_ev * _EV_TO_CM + 1e8 / float(air_to_vacuum(air)),
         log_gamma_rad=None, log_gamma_stark=None, source=_NASLIM2011)
    for element, charge, air, log_gf, lower_ev, lower_weight in (
        ('Ge', 2, 4178.96, 0.341, 19.66, 3.0),    # 4s5s 3S1 - 4s5p 3P2
        ('Ge', 2, 4260.85, 0.108, 19.66, 3.0),    # - 3P1
        ('Ge', 2, 4291.71, -0.368, 19.66, 3.0),   # - 3P0
        ('Y', 2, 4039.602, 1.005, 12.53, 8.0),    # 4f 2F7/2 - 5g 2G9/2
        ('Y', 2, 4039.602, -0.538, 12.53, 8.0),   # 4f 2F7/2 - 5g 2G7/2
        ('Y', 2, 4040.112, 0.892, 12.53, 6.0),    # 4f 2F5/2 - 5g 2G7/2
    )] + [
    dict(element='Sn', charge=3, wavelength=float(air_to_vacuum(air)), log_gf=float(np.log10(2 * f)),
         lower_energy=174138.0, lower_weight=2.0, upper_energy=174138.0 + 1e8 / float(air_to_vacuum(air)),
         log_gamma_rad=None, log_gamma_stark=None, source=_BISWAS2018)
    for air, f in ((3862.051, 0.960), (4217.184, 0.445))]  # 6s 2S1/2 - 6p 2P3/2, 2P1/2


@dataclass
class HeavyLines:
    atmosphere: object
    abundances: dict  # element -> log10 N/N(H)

    def __post_init__(self):
        unknown = set(self.abundances) - set(ELEMENTS)
        if unknown:
            raise ValueError(f'no heavy-element data for {sorted(unknown)}; available {sorted(ELEMENTS)}')
        atm = self.atmosphere
        temperature, electrons = np.asarray(atm.temperature), np.asarray(atm.electron_density)
        nh = atm.hydrogen_lte_state.hydrogen_nuclei_density
        self.lines, self.populations = [], {}
        translation = (2 * np.pi * ELECTRON_MASS * BOLTZMANN * temperature / PLANCK ** 2) ** 1.5
        for element, abundance in self.abundances.items():
            spec = ELEMENTS[element]
            charges = sorted(spec['stages'])
            partitions = {}
            for q in charges:
                stage = spec['stages'][q]
                if 'kurucz' in stage:
                    log_t, values = kurucz_partition(stage['kurucz'])
                    partitions[q] = np.interp(np.log10(temperature), log_t, values)
                elif 'levels' in stage:
                    partitions[q] = sum(g * np.exp(-e * WAVENUMBER_TO_ERG / (BOLTZMANN * temperature))
                                        for e, g in stage['levels'])
                else:
                    partitions[q] = np.full_like(temperature, stage['ground_weight'])
            logs = [np.zeros_like(temperature)]
            for lower, upper in zip(charges[:-1], charges[1:]):
                chi = spec['stages'][lower]['ionization'] * EV
                logs.append(logs[-1] + np.log(2 * partitions[upper] / partitions[lower] * translation / electrons)
                            - chi / (BOLTZMANN * temperature))
            logs = np.asarray(logs)
            fractions = np.exp(logs - logs.max(axis=0)); fractions /= fractions.sum(axis=0)
            total = nh * 10.0 ** abundance
            for q, fraction in zip(charges, fractions):
                self.populations[(element, q)] = (total * fraction, partitions[q])
            for q in charges:
                if 'kurucz' in spec['stages'][q]:
                    for line in kurucz_lines(spec['stages'][q]['kurucz'], q):
                        self.lines.append(dict(line, element=element, mass=spec['mass'],
                                               ionization=spec['stages'][q]['ionization']))
            if 'supplementary' in spec:
                code, filename, q, label = spec['supplementary']
                for line in kurucz_lines(code, q, filename=filename, upper_label=label):
                    self.lines.append(dict(line, element=element, mass=spec['mass'],
                                           ionization=spec['stages'][q]['ionization']))
            for line in LITERATURE_LINES:
                if line['element'] == element:
                    self.lines.append(dict(line, mass=spec['mass'], ionization=spec['stages'][line['charge']]['ionization']))
        for line in self.lines:
            line['wavelength'] = updated_vacuum(line['element'], line['charge'], line['wavelength'])

    def coefficients(self, wavelength, half_window=3.0):
        """(absorption, emissivity) per unit mass on (wavelength, depth), LTE source function."""
        atm = self.atmosphere
        wave = np.asarray(wavelength, dtype=float)
        temperature, electrons = np.asarray(atm.temperature), np.asarray(atm.electron_density)
        density = np.asarray(atm.mass_density)
        absorption = np.zeros((wave.size, temperature.size))
        for line in self.lines:
            lo, hi = np.searchsorted(wave, [line['wavelength'] - half_window, line['wavelength'] + half_window])
            if hi <= lo:
                continue
            ions, partition = self.populations[(line['element'], line['charge'])]
            lower = ions * line['lower_weight'] * np.exp(-line['lower_energy'] * WAVENUMBER_TO_ERG
                                                         / (BOLTZMANN * temperature)) / partition
            nu0 = LIGHT_SPEED / (line['wavelength'] * 1e-8)
            stimulated = 1 - np.exp(-PLANCK * nu0 / (BOLTZMANN * temperature))
            doppler = nu0 / LIGHT_SPEED * np.sqrt(2 * BOLTZMANN * temperature / (line['mass'] * ATOMIC_MASS_UNIT))
            gamma_rad = (10 ** line['log_gamma_rad'] if line.get('log_gamma_rad') is not None
                         else 0.2223e16 / line['wavelength'] ** 2)
            if line.get('log_gamma_stark') is not None:
                gamma_stark = 10 ** line['log_gamma_stark'] * electrons * (temperature / 1e4) ** (1 / 6)
            else:
                gamma_stark = electrons * _classical_electron_stark_rate_per_electron(
                    line['charge'], line['ionization'], line['upper_energy'])
            damping = (gamma_rad + gamma_stark) / (4 * np.pi * doppler)
            nu = LIGHT_SPEED / (wave[lo:hi] * 1e-8)
            v = (nu[:, None] - nu0) / doppler[None, :]
            profile = wofz(v + 1j * damping[None, :]).real / (np.sqrt(np.pi) * doppler[None, :])
            absorption[lo:hi] += (CLASSICAL_CROSS_SECTION * 10 ** line['log_gf'] / line['lower_weight']
                                  * (lower * stimulated / density)[None, :] * profile)
        wave_cm = wave[:, None] * 1e-8
        # The exponent is capped: B is then ~1e-300 relative, i.e. zero, without overflow.
        exponent = np.minimum(PLANCK * LIGHT_SPEED / (wave_cm * BOLTZMANN * temperature[None, :]), 700.0)
        planck = 2 * PLANCK * LIGHT_SPEED ** 2 / wave_cm ** 5 * 1e-8 / np.expm1(exponent)
        return absorption, absorption * planck
