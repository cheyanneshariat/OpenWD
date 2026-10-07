"""LTE populations and line opacity for trace metals in white dwarfs.

Atomic levels and transition probabilities are read from the open Stout
database distributed by the Atomic Line List project.  The thermodynamic
ionization balance is solved simultaneously with charge neutrality.  For
cool helium-rich atmospheres, the first ionization potential can optionally
include the density-dependent depression calculated by Blouin et al. (2018).

Ground-state photoionization uses the analytic fits of Verner et al. (1996).
Both bound-bound and bound-free opacity can be included consistently in the
non-gray helium structure iteration as well as in the final formal solution.
The public Mg I--He red-wing table and provenance-explicit figure-profile
bridges cover the first strong unified profiles.  Ca II H/K can additionally
interpolate the published 4000--10,000-K figure curves at one density; the
missing density grid and other dense-helium resonance profiles remain
important future work.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
import re
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._compat import trapezoid

try:  # Optional element-independent profile kernel.
    from . import _rt
except ImportError:  # pragma: no cover - exercised when the extension is absent
    _rt = None

from .atmosphere import Atmosphere
from .constants import (
    BOLTZMANN,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE_ESU,
    HELIUM_MASS,
    HYDROGEN_MASS,
    LIGHT_SPEED,
    PI,
    PLANCK,
)
from .eos import (
    HeliumLTEState,
    HydrogenLTEState,
    charged_particle_hydrogen_occupation_probability,
    hooper_microfield_cumulative_probability,
    hydrogenic_critical_microfield_beta,
    hydrogen_level_distribution,
    hydrogen_saha_constant,
    molecular_hydrogen_dissociation_constant,
    molecular_hydrogen_ion_dissociation_constant,
    negative_hydrogen_ionization_constant,
    trihydrogen_ion_dissociation_constant,
)


FloatArray = NDArray[np.float64]
EV_TO_ERG = 1.602_176_634e-12
WAVENUMBER_TO_ERG = PLANCK * LIGHT_SPEED
EV_TO_WAVENUMBER = EV_TO_ERG / WAVENUMBER_TO_ERG
BOHR_RADIUS_CM = 5.291_772_105_44e-9
STOUT_ARCHIVE_URL = "https://linelist.pa.uky.edu/newpage/downloads/stout.tar.gz"
STOUT_ARCHIVE_SHA256 = "6be3a8ee145aa9364e06eec25348a3ac65cdcaf68951c52203b9bd7a4413341e"
MG_HE_RED_WING_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/619/A152/fig5.dat"
MG_HE_RED_WING_SHA256 = "b79879240f935aadb7afbf68a590a26a2a43c572315323c8785fc750bb0f9aa7"
VERNER_PHOTOIONIZATION_URL = "https://www.pa.uky.edu/~verner/dima/photo/photo.dat"
VERNER_PHOTOIONIZATION_SHA256 = "a53e73b0af4cc6b801aef67b84bd964e3ed8c9c4992f214e4a69dc7e4443f389"
VERNER_PHFIT2_URL = "https://www.pa.uky.edu/~verner/dima/photo/phfit2.f"
VERNER_PHFIT2_SHA256 = "8df982535a4a2b952e33ca2ffb6d81600e156d7373fbf547a0c0170c1cecc296"
BARKLEM_NEUTRAL_H_BROADENING_URL = (
    "https://raw.githubusercontent.com/barklem/public-data/"
    "master/broadening-neutrals/hlist"
)
BARKLEM_NEUTRAL_H_BROADENING_SHA256 = (
    "4c2476f94f63970d01986c0e68ab0b0a1148b37e9a4311ab9e115f9f75a992ad"
)
# Ordinary line profiles are evaluated over at least this half-window even
# when their thermal and Lorentz widths are much narrower.  Structure meshes
# must sample the same support or trapezoidal frequency integration assigns
# broad wavelength intervals to unresolved wing values.
METAL_LINE_MINIMUM_HALF_WINDOW_ANGSTROM = 0.25
# Maximum profile half-window of the hot NLTE metal lines (light_metal_nlte),
# as a fraction of a line's central wavelength.  The impact (Lorentz)
# approximation fails at detunings approaching the line frequency itself.
# Without this bound the Stark-broadened high-Rydberg Delta-n=1 lines of
# O IV-O VI (FWHM up to ~1e4 A at 2-6 micron) extend their Lorentz wings across
# the entire spectrum and act as a spurious UV/optical pseudo-continuum (the
# PG 1159-035 audit of 2026-09-24).  The LTE metal-line path is not bounded:
# pressure-broadened wings in cool dense atmospheres (DQ) are genuine opacity.
# Must match LINE_WINDOW_MAX_FRACTION in csrc/rt_core.c.
LINE_WINDOW_MAX_FRACTION = 0.1
KURUCZ_GF100_FILES = MappingProxyType(
    {
        "gf0300.100": (
            "http://kurucz.harvard.edu/linelists/GF100/gf0300.100",
            "3c6f9a5116d015799a409ebbd2cbc39577b9d59100bad80bf4f9ccef7c7a7e97",
        ),
        "gf0400.100": (
            "http://kurucz.harvard.edu/linelists/GF100/gf0400.100",
            "b17ac909eac49230e060d3147a6fbb0591fed4276206925342e37e4cdde6e513",
        ),
        "gf0500.100": (
            "http://kurucz.harvard.edu/linelists/GF100/gf0500.100",
            "8b9d35221a97c054db1f87d4a7672b786acda40349aabd21f3f65697546947ea",
        ),
        "gf0600.100": (
            "http://kurucz.harvard.edu/linelists/GF100/gf0600.100",
            "0a6e23c7b5fc93b0e44ca42905b0bc15f427e22354fe12a1ddd0b574a3cba9c1",
        ),
        "gf0800.100": (
            "http://kurucz.harvard.edu/linelists/GF100/gf0800.100",
            "0147500330686cfd0ca8339aaf21c60275eaafcc645eb979205e009ba6c1f788",
        ),
    }
)
KURUCZ_CURRENT_STRONG_ION_FILES = MappingProxyType(
    {
        "gf0800.all": (
            "http://kurucz.harvard.edu/atoms/0800/gf0800.all",
            "aaf1d1e7b332f44158c69f0a1341f97fba466197d95c5a9fd1a819f439cbfb73",
        ),
        "gf1100.all": (
            "http://kurucz.harvard.edu/atoms/1100/gf1100.all",
            "df3516b4c94fd98c81462019d914d8cd7bfa2a292fe569d9b47a4f9a9c031094",
        ),
        "gf1200.all": (
            "http://kurucz.harvard.edu/atoms/1200/gf1200.all",
            "8efc11d5545d2a2164983396adcc6e657606fa1fd606103f3f73d71e61401b34",
        ),
        "gf1201.all": (
            "http://kurucz.harvard.edu/atoms/1201/gf1201.all",
            "61feb0e686c25f354fd7b8fa680471c5163695eef56767d8c43502b97a3c2e5c",
        ),
        "gf1300.all": (
            "http://kurucz.harvard.edu/atoms/1300/gf1300.all",
            "c22d80418db369134bc5f56bcdc721f24a2e57a0bdfbeedaa98381d962973cef",
        ),
        "gf1401.all": (
            "http://kurucz.harvard.edu/atoms/1401/gf1401.all",
            "d18ba25a8496d777d4017039c1e0567533b250e4d14c2e2eb0c24e5295c12c19",
        ),
        "gf2000.all": (
            "http://kurucz.harvard.edu/atoms/2000/gf2000.all",
            "4090001902987d979098a7af2025d17d2ff3d90e57937cf6cde8dce513d934d9",
        ),
        "gf2001.all": (
            "http://kurucz.harvard.edu/atoms/2001/gf2001.all",
            "b43a8b2a4fa1434d1d77cde8c11698becfa8a7fe25c88e3f958106939be13afd",
        ),
        "gf2201.all": (
            "http://kurucz.harvard.edu/atoms/2201/gf2201.all",
            "f5e6866e03f1f935276365392b2642695b33e8b9a230bc469093ccd15a30298d",
        ),
        "gf2401.all": (
            "http://kurucz.harvard.edu/atoms/2401/gf2401.all",
            "6f0c4d0e01421fb0549ddbcf5649af391ca9b930e4e83e693e662d5874505205",
        ),
        "gf2601.all": (
            "http://kurucz.harvard.edu/atoms/2601/gf2601.all",
            "5a7d9d26eef58a2a32da523f8221f50cb08d3316cff20b45e22c8913e5390892",
        ),
    }
)

# Version-5.12 NIST ASD tab-delimited queries restricted to transitions with
# published transition probabilities between 300 and 900 nm.  These files are
# deliberately small: they are used to replace exact-level matches for the
# strongest optical diagnostics, not as a second complete line list.  NIST
# recommends citing Kramida et al. (2024), DOI 10.18434/T4W30F.
_NIST_ASD_LINES_QUERY = (
    "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?"
    "spectra={spectrum}&output_type=0&low_w=300&upp_w=900&unit=1&"
    "submit=Retrieve%20Data&de=0&plot_out=0&I_scale_type=1&format=3&"
    "line_out=1&en_unit=0&output=0&bibrefs=1&page_size=2000&"
    "show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&"
    "show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&"
    "loggf_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&"
    "min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on"
)
_NIST_ASD_UV_LINES_QUERY = _NIST_ASD_LINES_QUERY.replace("low_w=300&upp_w=900", "low_w=115&upp_w=300")
NIST_ASD_STRONG_ION_FILES = MappingProxyType(
    {
        "nist-asd-ca1.tsv": (
            _NIST_ASD_LINES_QUERY.format(spectrum="Ca%20I"),
            "691775b88d62b0d311df18f099c863cfda2dbb687e915d9acb868064c0916ec7",
            "Ca", 0,
        ),
        "nist-asd-ca2.tsv": (
            _NIST_ASD_LINES_QUERY.format(spectrum="Ca%20II"),
            "ed7a0f6304ed4d300533813cdd1f4b8b78f504bd4f0d46f022639421549b1cb4",
            "Ca", 1,
        ),
        "nist-asd-mg1.tsv": (
            _NIST_ASD_LINES_QUERY.format(spectrum="Mg%20I"),
            "5941149b7b19f035d502572cf3f237e5359e9f683ac0614e3e54db26c4fca5fc",
            "Mg", 0,
        ),
        "nist-asd-mg2.tsv": (
            _NIST_ASD_LINES_QUERY.format(spectrum="Mg%20II"),
            "57b4242d5d856ff70baa24b5c70a3efbe869dc56b9c579dfd22eb10055843b0f",
            "Mg", 1,
        ),
        "nist-asd-na1.tsv": (
            _NIST_ASD_LINES_QUERY.format(spectrum="Na%20I"),
            "c8d25e0912cfe46377d25ba8b26d4cbca462c36c86b9df48fb29a26a0525bbcb",
            "Na", 0,
        ),
        "nist-asd-o1.tsv": (
            _NIST_ASD_LINES_QUERY.format(spectrum="O%20I"),
            "053df0e3e62bdd6e7566547d4f03de3d3385a2af37c8fe5eb33973070b8e2bc9",
            "O", 0,
        ),
        "nist-asd-si2.tsv": (
            _NIST_ASD_LINES_QUERY.format(spectrum="Si%20II"),
            "1f13984a0cb4a1d2e727797b923a594f8638e050c532a719212d51f5d1872c88",
            "Si", 1,
        ),
        # 115--300 nm: UV resonance lines (Si II 1808, Mg I 2852, Fe II, ...).
        "nist-asd-ca1-uv.tsv": (
            _NIST_ASD_UV_LINES_QUERY.format(spectrum="Ca%20I"),
            "5edbec7445698b37e9d05ecd7a89effbffc8a166125ee9da84645f0fd1ffa451",
            "Ca", 0,
        ),
        "nist-asd-ca2-uv.tsv": (
            _NIST_ASD_UV_LINES_QUERY.format(spectrum="Ca%20II"),
            "dd2c28e06e40fe674f9a5ba63aa5cef7a659eb5035326319c00b3326dd1ece48",
            "Ca", 1,
        ),
        "nist-asd-mg1-uv.tsv": (
            _NIST_ASD_UV_LINES_QUERY.format(spectrum="Mg%20I"),
            "f2cf06e853dd5f13c13bd25c04460dfdbd1629acdb8147a970d4e6db4f3bc82b",
            "Mg", 0,
        ),
        "nist-asd-mg2-uv.tsv": (
            _NIST_ASD_UV_LINES_QUERY.format(spectrum="Mg%20II"),
            "99ed7d2b7c61f2e0d1b04025c9e489b1a2c34c99d22aa3b4cd047651f7cf48ef",
            "Mg", 1,
        ),
        "nist-asd-na1-uv.tsv": (
            _NIST_ASD_UV_LINES_QUERY.format(spectrum="Na%20I"),
            "b6ee800bd7e13f6ceeba42eaf6011779a61544b1b50dafafebf929e945fe2cfe",
            "Na", 0,
        ),
        "nist-asd-o1-uv.tsv": (
            _NIST_ASD_UV_LINES_QUERY.format(spectrum="O%20I"),
            "e9dcb4164f1ec87eb770e1299af6ec0bc1fafe66fe18ca79b3174ee860217d04",
            "O", 0,
        ),
        "nist-asd-si2-uv.tsv": (
            _NIST_ASD_UV_LINES_QUERY.format(spectrum="Si%20II"),
            "2a871f1f24c835a89637f074fc6203c7e4a082e72082bb49be1ba023a96e7eb2",
            "Si", 1,
        ),
        "nist-asd-si1-uv.tsv": (
            _NIST_ASD_UV_LINES_QUERY.format(spectrum="Si%20I"),
            "2c6a44698c699988c290e3c0eb061f4648b9d6bcffe5249ddfe455f5cd22ae8e",
            "Si", 0,
        ),
        "nist-asd-fe1-uv.tsv": (
            _NIST_ASD_UV_LINES_QUERY.format(spectrum="Fe%20I"),
            "2e816e95c2eec8c7daeefbbbe2546a198db52d1ac2b02785dd2cf6f56105b8f3",
            "Fe", 0,
        ),
        "nist-asd-fe2-uv.tsv": (
            _NIST_ASD_UV_LINES_QUERY.format(spectrum="Fe%20II"),
            "fa91ab12a934ea6a06b327d903fa72580e654f037d4fd850299b37d5fbd3259a",
            "Fe", 1,
        ),
        "nist-asd-c2-uv.tsv": (
            _NIST_ASD_UV_LINES_QUERY.format(spectrum="C%20II"),
            "09a0f2c4a08b9c13b47b92329db89ef03758526c666e22a3a90960cf8c46829b",
            "C", 1,
        ),
        "nist-asd-al2-uv.tsv": (
            _NIST_ASD_UV_LINES_QUERY.format(spectrum="Al%20II"),
            "5c41e58c8d0edd109b2f84de2ab5cabb2eb7df3ddae66a75b8dcc41f1a8f9b59",
            "Al", 1,
        ),
    }
)

# Hammond (1975, ApJ 196, 291), equations 6--7, measured the Lorentz
# wavenumber HWHM per neutral-He perturber at 5200 K for Ca II K and H.
# His gamma is the angular-frequency damping constant.  The Voigt convention
# below therefore needs Gamma/N = 4 pi c (delta_wavenumber/N).
#
# His two-shot temperature exponent (0.23, from 4400 and 6080 K) is, in his
# words, "far more uncertain because of poor statistics", and he discourages
# its use.  The same paper fits Lennard-Jones (12,6) potentials to the
# measured widths and shifts (eqs. 22--31): a = 4.09 (K) and 1.63 (H) at the
# mean relative speed 5.5e5 cm/s.  Lindholm--Foley/Hindmarsh impact theory
# then gives gamma/N proportional to v^(3/5) B(a) with a proportional to
# v^(6/5), which reproduces the 5200-K measurement exactly and carries the
# temperature dependence of the fitted He repulsion.
_CA_II_HE_WAVENUMBER_HWHM_PER_DENSITY_5200 = MappingProxyType(
    {"K": 1.71e-20, "H": 1.28e-20}
)
_CA_II_HE_REFERENCE_TEMPERATURE = 5200.0
_CA_II_HE_LENNARD_JONES_A_5200 = MappingProxyType({"K": 4.09, "H": 1.63})
_CA_II_HE_REDUCED_MASS_U = 40.078 * 4.002602 / (40.078 + 4.002602)

# Aguilera, Aragon & Manrique (2014, MNRAS 444, 1854), Table 2, measured
# electron-impact FWHM values at ne=1e17 cm^-3 and T=14000 K.  Their reported
# H and K widths agree with the earlier semiclassical calculations tabulated
# by Dimitrijevic & Sahal-Brechot.  Table 2 also contains the subordinate
# 4p--4d and 4p--5s lines that are prominent in the near-UV spectra of warm
# DZ/DBZ stars.  Centers here are vacuum wavelengths (the paper lists air
# wavelengths for the optical transitions).  The T^-1/2 scaling is the
# correction used in that comparison for the resonance lines.
_CA_II_ELECTRON_STARK_FWHM_14000 = MappingProxyType(
    {
        "3159": 0.53,
        "3180": 0.49,
        "3707": 0.66,
        "3738": 0.67,
        "K": 0.17,
        "H": 0.16,
    }
)
_CA_II_ELECTRON_STARK_REFERENCE_DENSITY = 1.0e17
_CA_II_ELECTRON_STARK_REFERENCE_TEMPERATURE = 14_000.0

# Dimitrijevic & Sahal-Brechot (1995), as tabulated and adopted for white
# dwarf synthesis by Vennes et al. (2011), give a 2.50-A electron-impact FWHM
# for the unresolved Mg II 4481-A multiplet at this plasma condition.
_MG_II_4481_ELECTRON_STARK_FWHM = 2.50
_MG_II_4481_ELECTRON_STARK_REFERENCE_DENSITY = 1.35e17
_MG_II_4481_ELECTRON_STARK_REFERENCE_TEMPERATURE = 16_900.0

# Kurucz's current per-ion Mg II list (atoms/1201/gf1201.all, SHA256
# 61feb0e686c25f354fd7b8fa680471c5163695eef56767d8c43502b97a3c2e5c)
# gives log10(Gamma_e/ne)=-2.59 for every fine-structure component of the
# 4f 2Fo--8g 2G multiplet at 4852.4 A.  The file uses the standard Kurucz
# 10,000-K damping convention.  This line-specific value is about 82 times
# the capped generic SYNSPEC fallback and is required by both J1109 and J1637;
# it is deliberately restricted to this identified multiplet rather than
# importing unrelated Mg II estimates wholesale.
_MG_II_4852_ELECTRON_STARK_RATE_COEFFICIENT_10000 = 10.0**-2.59
_MG_II_4852_ELECTRON_STARK_REFERENCE_TEMPERATURE = 10_000.0

# Cvejic et al. (2013, Spectrochim. Acta B 85, 20), Table 4, measured
# 1.2--1.7-A electron-impact FWHM values for the Mg I 3p 3Po--3d 3D
# multiplet near 3835 A at ne=(0.67--1.09)e17 cm^-3 and T=6287--6464 K.
# The density-normalized measurements cluster around 1.55 A at ne=1e17.
# All fine-structure components had the same width within the experimental
# uncertainty.  This is more than ten times the generic SYNSPEC classical
# estimate; the independent Dimitrijevic--Sahal-Brechot calculation is also
# close (about 1.9 A at the same density and temperature).
_MG_I_3835_ELECTRON_STARK_FWHM = 1.55
_MG_I_3835_ELECTRON_STARK_REFERENCE_DENSITY = 1.0e17
_MG_I_3835_ELECTRON_STARK_REFERENCE_TEMPERATURE = 6_370.0

# Dimitrijevic & Sahal-Brechot (1996, A&AS 117, 127), CDS table 1,
# give semiclassical electron-impact widths for the Mg I 3p 1Po--5d 1D
# line at 4704.3 A and the 3p 3Po--4s 3S b triplet at 5179.6 A.  The
# tabulated widths are term-averaged FWHM values at ne=1e11 cm^-3.  These
# two multiplets are deliberately kept separate: their widths differ by
# more than an order of magnitude, which the generic n_eff^5 fallback does
# not reproduce accurately.
_MG_I_OPTICAL_STARK_TEMPERATURE_K = np.asarray(
    [2_500.0, 5_000.0, 10_000.0, 20_000.0, 30_000.0, 50_000.0]
)
_MG_I_OPTICAL_STARK_FWHM_ANGSTROM = MappingProxyType(
    {
        "3p1P-5d1D": np.asarray(
            [6.17e-6, 7.07e-6, 7.89e-6, 8.85e-6, 9.64e-6, 1.07e-5]
        ),
        "3p3P-4s3S": np.asarray(
            [3.88e-7, 4.60e-7, 5.32e-7, 5.82e-7, 6.10e-7, 6.49e-7]
        ),
    }
)
_MG_I_OPTICAL_STARK_REFERENCE_DENSITY = 1.0e11

# Dimitrijevic & Sahal-Brechot (1990, Bull. Obs. Astron. Belgrade 142,
# 59), Tables 1--6, give semiclassical electron-impact FWHM values for
# Na I.  The entries below are from their ne=1e16 cm-3 table, where the
# impact widths still scale linearly with density.  The wavelengths in the
# paper are term averages: 5686.4 A represents the 3p--4d multiplet near
# 5684/5690 A in vacuum, and 5891.8 A represents the Na D doublet.  Using
# the tabulated temperature dependence avoids both a separate fitted scale
# for each observed feature and the generic capped-n_eff fallback.
_NA_I_OPTICAL_STARK_TEMPERATURE_K = np.asarray(
    [2_500.0, 5_000.0, 10_000.0, 20_000.0, 30_000.0, 80_000.0]
)
_NA_I_OPTICAL_STARK_FWHM_ANGSTROM = MappingProxyType(
    {
        "3p-4d": np.asarray([2.25, 2.22, 2.11, 1.97, 1.88, 1.65]),
        "3s-3p": np.asarray(
            [0.0191, 0.0211, 0.0249, 0.0322, 0.0381, 0.0551]
        ),
    }
)
_NA_I_OPTICAL_STARK_REFERENCE_DENSITY = 1.0e16

# Kachru, Mossberg & Hartmann (1980, J. Phys. B 13, L363;
# doi:10.1088/0022-3700/13/12/002), Table 2, measured the Na D FWHM
# damping coefficients in neutral neon at 450 K.  The values are angular-
# frequency FWHM per perturber in cm3 s-1, matching the damping-rate
# convention used below.  This direct Na--Ne result supersedes the generic
# H-rate/polarizability rescaling for these two resonance lines only.  The
# ordinary impact T^0.3 scaling is retained outside the laboratory point;
# this supplies the line core, not a unified far-wing profile.
_NA_I_D_NEON_REFERENCE_TEMPERATURE_K = 450.0
_NA_I_D_NEON_DAMPING_RATE_CM3_S = MappingProxyType(
    {
        "D2": 2.37e-8,
        "D1": 2.27e-8,
    }
)

# Dimitrijevic & Sahal-Brechot (2025, Galaxies 13, 116), their
# Table 2, give a semiclassical electron-impact FWHM of 0.192 A for the
# O I 3p 5P--4d 5Do multiplet at ne=1e16 cm^-3 and T=10000 K.  This later,
# line-by-line calculation supersedes the anomalous 16.7-A value printed in
# their earlier CAOSP table.  That older number disagrees by nearly an order
# of magnitude with both other calculations collected in the later paper
# and with the ordinary SYNSPEC classical estimate.
_O_I_3P5P_4D5D_STARK_FWHM_ANGSTROM = 0.192
_O_I_3P5P_4D5D_STARK_REFERENCE_WAVELENGTH = 6159.0
_O_I_3P5P_4D5D_STARK_REFERENCE_DENSITY = 1.0e16
_O_I_3P5P_4D5D_STARK_REFERENCE_TEMPERATURE = 10_000.0
_O_I_3P5P_4D5D_REFERENCE_EFFECTIVE_N = 3.9675

# Electron-impact FWHM values at ne=1e16 cm^-3.  The 4369-A values are from
# Dimitrijevic & Sahal-Brechot (2025, Galaxies 13, 116), Table 1.  Their
# 7774-A values are not used: at 10 kK the printed 0.00528-A FWHM is about an
# order of magnitude below Griem's width and the resolved measurements of
# Gosse et al. (2025, Spectrochim. Acta B 230, 107222).  The 7774-A table
# below follows the temperature dependence implied by Griem's 2500/40000-K
# endpoints and is normalized to the Gosse et al. 10--13.5-kK measurements.
#
# The printed 8449-A table has the same factor-of-about-ten inconsistency with
# the independent Griem curves reproduced by Werbowy, Pranszke & Windholz
# (2025, Phys. Rev. E 111, 025209, Fig. 9).  The corrected values below use
# that factor-ten normalization while retaining the calculated temperature
# dependence.  These two corrections matter for saturated O-rich-remnant
# lines even when their line-forming ne is below the laboratory reference
# density.
_O_I_OPTICAL_STARK_TEMPERATURE_K = np.asarray(
    [2_500.0, 5_000.0, 10_000.0, 20_000.0, 40_000.0, 80_000.0]
)
_O_I_OPTICAL_STARK_FWHM_ANGSTROM = MappingProxyType(
    {
        "3s3S-4p3P": np.asarray(
            [0.00926, 0.0107, 0.0130, 0.0161, 0.0200, 0.0236]
        ),
        "3s5S-3p5P": np.asarray(
            [0.0417153, 0.0539325, 0.0697278, 0.0901491, 0.116551, 0.150686]
        ),
        "3s3S-3p3P": np.asarray(
            [0.0686, 0.0721, 0.0822, 0.107, 0.145, 0.187]
        ),
    }
)
_O_I_OPTICAL_STARK_REFERENCE_DENSITY = 1.0e16

_METAL_HOLTSMARK_LAGUERRE_ABSCISSA, _METAL_HOLTSMARK_LAGUERRE_WEIGHT = (
    np.polynomial.laguerre.laggauss(96)
)

# Static dipole polarizabilities in A^3 (NIST CCCBDB values).  Their ratio
# rescales Unsold's H-perturber C6 to neutral helium; only the ratio enters.
_HYDROGEN_STATIC_POLARIZABILITY_A3 = 0.666_793
_HELIUM_STATIC_POLARIZABILITY_A3 = 0.204_956
_HYDROGEN_ATOMIC_MASS_U = 1.007_84
_HELIUM_ATOMIC_MASS_U = 4.002_602
# Experimental static dipole polarizabilities from the NIST CCCBDB (A^3).
# These are the dominant neutral perturbers in warm C/O and O/Ne remnant
# atmospheres.  Trace neutral metals are deliberately omitted until equally
# reliable values and a demonstrated spectroscopic need are available.  The
# Ne value is 2.66080 a0^3 from Lesiuk, Przybytek & Jeziorski (2020), converted
# with a0=0.529177210903 A; it agrees with the best experimental value.
_BULK_METAL_STATIC_POLARIZABILITY_A3 = MappingProxyType(
    {"C": 1.760, "O": 0.802, "Ne": 0.394_299}
)

# Negative-ion continuum data used for cool O/Ne-dominated remnants.  John
# (1975a,b) gives the long-wavelength free-free coefficient in the form
# k_lambda=A(T) lambda[A]^2 n(X I) P_e.  The two temperatures below are the
# directly relevant entries of his Table I; interpolation is a power law in
# temperature.  The O- electron affinity is the NIST value.  The compact
# photodetachment envelope follows the Wigner threshold law and is normalized
# to the 5.9--6.3e-18 cm2 laboratory measurements at 662/532 nm; its 1.2e-17
# cm2 asymptote is the close-coupling result of Robinson & Geltman (1967).
_JOHN_NEGATIVE_ION_FREE_FREE_A_TIMES_1E34 = MappingProxyType(
    {"Ne": (0.0410, 0.0328), "O": (0.429, 0.290), "Na": (2.7, 1.3)}
)
_O_MINUS_ELECTRON_AFFINITY_EV = 1.461_113_6
_O_MINUS_ASYMPTOTIC_CROSS_SECTION_CM2 = 1.2e-17
_RYDBERG_ENERGY_EV = 13.605_693_122_994

# The Stout O I file omits eight fine-structure components connecting the
# autoionizing 2p3(2D)3p 3F and 2p3(2D)4d 3F/3G levels at 6258--6271 A even
# though all seven participating levels are present.  These oscillator
# strengths and the common upper-level
# radiative damping rate are from Kurucz's public GF100 line lists.  Keeping
# this as a compact level-to-level supplement avoids replacing the generally
# better-performing Stout O I transition set wholesale.
_O_I_6258_6271_MULTIPLET = (
    (113714.444, 129693.488, 4.413239438549339e-05),
    (113721.413, 129693.488, 2.9778441187104125e-03),
    (113727.165, 129693.488, 4.786631512810775e-02),
    (113714.444, 129680.522, 4.516036990724586e-02),
    (113714.444, 129679.841, 2.316100981219210e-03),
    (113721.413, 129679.841, 4.507149461963752e-02),
    (113714.444, 129666.907, 1.0858191343953452e-02),
    (113721.413, 129666.907, 9.438477828679939e-04),
)
_O_I_6258_6271_RADIATIVE_DAMPING_RATE_S = 43_651_583.22401656


# CI-chondrite number ratios relative to calcium from columns 5--6 of
# Lodders (2003), ApJ 591, 1220, Table 1.  Blouin, Dufour & Allard (2018)
# scale otherwise unconstrained elements from C through Cu this way in their
# Ross 640 and LP 658-2 validation models.  We retain the subset currently
# covered by the LTE/Stout implementation; individually measured abundances
# should be supplied as overrides.
CHONDRITIC_LOG_NUMBER_RATIO_TO_CA = MappingProxyType(
    {
        "C": np.log10(7.724e5 / 5.968e4),
        "O": np.log10(7.552e6 / 5.968e4),
        "Na": np.log10(5.747e4 / 5.968e4),
        "Mg": np.log10(1.040e6 / 5.968e4),
        "Al": np.log10(8.308e4 / 5.968e4),
        "Si": np.log10(1.000e6 / 5.968e4),
        "Ca": 0.0,
        "Ti": np.log10(2.422e3 / 5.968e4),
        "Cr": np.log10(1.313e4 / 5.968e4),
        "Fe": np.log10(8.632e5 / 5.968e4),
        "Ni": np.log10(4.780e4 / 5.968e4),
    }
)


# Atomic weights are used only for thermal Doppler widths.  Ionization
# energies are NIST ASD 5.12 ground-state values in eV; entry k ionizes
# charge k to charge k+1.  The PG1159 trace mixture (F, Mg, Si, P, S, Ar,
# Ca, and Fe) now has a complete ladder through its bare nucleus so its LTE
# line blanketing cannot be spuriously trapped in the highest low ion loaded
# for the earlier cool-DZ milestone.
ATOMIC_MASS_U = MappingProxyType(
    {
        "C": 12.011,
        "F": 18.998_403_163,
        "N": 14.007,
        "Ne": 20.1797,
        "O": 15.999,
        "Na": 22.98976928,
        "Mg": 24.305,
        "Al": 26.9815385,
        "Si": 28.085,
        "P": 30.973761998,
        "S": 32.06,
        "Ar": 39.948,
        "Ca": 40.078,
        "Sc": 44.955908,
        "Ti": 47.867,
        "V": 50.9415,
        "Cr": 51.9961,
        "Mn": 54.938044,
        "Fe": 55.845,
        "Co": 58.933194,
        "Ni": 58.6934,
        "Cu": 63.546,
        "Zn": 65.38,
    }
)
ATOMIC_NUMBER = MappingProxyType(
    {
        "H": 1, "He": 2, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
        "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Ca": 20,
        "Ar": 18,
        "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
        "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    }
)
IONIZATION_ENERGY_EV = MappingProxyType(
    {
        "C": (
            11.2602880,
            24.383143,
            47.88778,
            64.49352,
            392.09056,
            489.99320779,
        ),
        "F": (
            17.42282,
            34.97081,
            62.70798,
            87.175,
            114.249,
            157.16311,
            185.1868,
            953.8983,
            1103.1175302,
        ),
        "N": (
            14.53413,
            29.60125,
            47.4453,
            77.4735,
            97.8901,
            552.06741,
            667.0461377,
        ),
        "Ne": (
            21.564541,
            40.96297,
            63.4233,
            97.1900,
            126.247,
            157.934,
            207.271,
            239.0970,
            1195.8082,
            1362.199256,
        ),
        "O": (
            13.618055,
            35.12112,
            54.93554,
            77.41350,
            113.8990,
            138.1189,
            739.32697,
            871.4099138,
        ),
        "Na": (5.13907696, 47.28636, 71.6200),
        "Mg": (
            7.646236,
            15.035271,
            80.1436,
            109.2654,
            141.33,
            186.76,
            225.02,
            265.924,
            327.99,
            367.489,
            1761.8049,
            1962.663889,
        ),
        "Al": (5.985769, 18.82855, 28.447642),
        "Si": (
            8.15168,
            16.34585,
            33.49300,
            45.14179,
            166.767,
            205.279,
            246.57,
            303.59,
            351.28,
            401.38,
            476.273,
            523.415,
            2437.65805,
            2673.177958,
        ),
        "P": (
            10.486686,
            19.76949,
            30.20264,
            51.44387,
            65.02511,
            220.430,
            263.57,
            309.60,
            372.31,
            424.40,
            479.44,
            560.62,
            611.741,
            2816.90868,
            3069.842145,
        ),
        "S": (
            10.3600167,
            23.33788,
            34.86,
            47.222,
            72.5945,
            88.0529,
            280.954,
            328.794,
            379.84,
            447.7,
            504.55,
            564.41,
            651.96,
            706.994,
            3223.78057,
            3494.188518,
        ),
        "Ar": (
            15.7596119,
            27.62967,
            40.735,
            59.58,
            74.84,
            91.290,
            124.41,
            143.4567,
            422.60,
            479.76,
            540.4,
            619.0,
            685.5,
            755.13,
            855.5,
            918.375,
            4120.66559,
            4426.22407,
        ),
        "Ca": (
            6.1131549210,
            11.871719,
            50.91316,
            67.2732,
            84.34,
            108.78,
            127.21,
            147.24,
            188.54,
            211.275,
            591.60,
            658.2,
            728.6,
            817.2,
            894.0,
            973.7,
            1086.8,
            1157.726,
            5128.8576,
            5469.86358,
        ),
        "Sc": (6.56149, 12.79977, 24.75684),
        "Ti": (6.828120, 13.5755, 27.49171),
        "V": (6.746187, 14.634, 29.311),
        "Cr": (6.76651, 16.486305, 30.959),
        "Mn": (7.4340380, 15.6400, 33.668),
        "Fe": (
            7.9024681,
            16.19921,
            30.651,
            54.91,
            75.00,
            98.985,
            124.9671,
            151.060,
            233.6,
            262.10,
            290.9,
            330.8,
            361.0,
            392.2,
            456.2,
            489.312,
            1262.7,
            1357.8,
            1460.0,
            1575.6,
            1687.0,
            1798.4,
            1950.4,
            2045.759,
            8828.1864,
            9277.6886,
        ),
        "Co": (7.88101, 17.0844, 33.50),
        "Ni": (7.639878, 18.168838, 35.187),
        # NIST ASD 5.12 ground-state ionization energies.  The local Stout
        # cache contains the first three stages of both species, which is
        # sufficient for their trace optical lines in cool O/Ne remnants.
        "Cu": (7.726380, 20.29239, 36.841),
        "Zn": (9.394197, 17.96439, 39.72330),
    }
)

# Blouin, Dufour & Allard (2018), Table 2.  Their b coefficients are printed
# in units of 1e-4 eV g^-1 K^-1 cm^3.  Delta I is in eV for rho in g cm^-3.
_DENSE_HE_IONIZATION_FIT = MappingProxyType(
    {
        "C": (1.91782, -3.24813e-4, -1.19948),
        "Ca": (-2.20703, -0.14431e-4, 0.57494),
        "Fe": (-2.23142, 0.48427e-4, 0.21301),
        "Mg": (0.45809, -0.85522e-4, -1.01958),
        "Na": (-0.52305, -0.62471e-4, 0.04833),
    }
)


@dataclass(frozen=True)
class AtomicLevel:
    """One fine-structure level from a Stout ``.nrg`` file."""

    index: int
    energy_wavenumber: float
    statistical_weight: float
    label: str


@dataclass(frozen=True)
class AtomicTransition:
    """One spontaneous transition from a Stout ``.tp`` file."""

    lower_index: int
    upper_index: int
    einstein_a: float
    transition_type: str
    wavelength_vacuum_angstrom: float
    absorption_oscillator_strength: float
    profile_formula: int | None = None
    profile_parameters: tuple[float, ...] = ()
    radiative_damping_rate_s: float | None = None
    electron_stark_rate_coefficient_cm3_s: float | None = None
    neutral_h_vdw_rate_coefficient_cm3_s: float | None = None
    neutral_h_vdw_temperature_exponent: float | None = None


@dataclass(frozen=True)
class AtomicIon:
    """Level and transition data for one element and charge state."""

    element: str
    charge: int
    atomic_mass_u: float
    ionization_energy_ev: float | None
    levels: tuple[AtomicLevel, ...]
    transitions: tuple[AtomicTransition, ...]
    source: str = "Stout Atomic Line List"

    def partition_function(
        self,
        temperature: ArrayLike,
        *,
        cutoff_below_ionization_ev: float = 0.1,
    ) -> FloatArray:
        """Return the ideal internal partition function.

        Following Koester's atmosphere-code description, nominally bound
        levels end 0.1 eV below the next ionization threshold.  The dense-He
        correction changes the Saha energy but not this spectroscopic cutoff.
        """

        temperature_array = np.asarray(temperature, dtype=np.float64)
        if np.any(~np.isfinite(temperature_array)) or np.any(temperature_array <= 0.0):
            raise ValueError("temperature must be finite and positive")
        energy = np.asarray([level.energy_wavenumber for level in self.levels])
        weight = np.asarray([level.statistical_weight for level in self.levels])
        selected = np.isfinite(energy) & (energy >= 0.0) & (weight > 0.0)
        if self.ionization_energy_ev is not None:
            cutoff = max(self.ionization_energy_ev - cutoff_below_ionization_ev, 0.0)
            selected &= energy <= cutoff * EV_TO_WAVENUMBER
        if not np.any(selected):
            raise ValueError(f"{self.element} {self.charge:+d} has no usable levels")
        exponent = (
            -WAVENUMBER_TO_ERG
            * energy[selected, np.newaxis]
            / (BOLTZMANN * temperature_array.reshape(1, -1))
        )
        result = np.sum(weight[selected, np.newaxis] * np.exp(exponent), axis=0)
        return result.reshape(temperature_array.shape)

    def occupation_weighted_partition_function(
        self,
        temperature: ArrayLike,
        electron_density: ArrayLike,
        *,
        cutoff_below_ionization_ev: float = 0.1,
        neutral_perturber_number_density: Mapping[str, ArrayLike] | None = None,
        neutral_perturber_radius_cm: Mapping[str, float] | None = None,
    ) -> FloatArray:
        """Return a Q-MHD occupation-probability partition function.

        Koester's metal EOS replaces the nominal fixed spectroscopic cutoff
        with a density-dependent non-ideal cutoff. Summing the same bound
        levels with their charged-microfield survival probabilities is the
        smooth Hummer--Mihalas analogue and uses the same level dissolution
        later applied to the line opacity.  Optional neutral perturbers (for
        example ground-state He in a DZ atmosphere) multiply in the HM88
        excluded-volume probability, as in
        :func:`metal_rydberg_level_occupation_probability`.
        """

        temperature_array, electron_density_array = np.broadcast_arrays(
            np.asarray(temperature, dtype=np.float64),
            np.asarray(electron_density, dtype=np.float64),
        )
        if (
            np.any(~np.isfinite(temperature_array))
            or np.any(temperature_array <= 0.0)
            or np.any(~np.isfinite(electron_density_array))
            or np.any(electron_density_array < 0.0)
        ):
            raise ValueError("temperature and electron density must be physical")
        energy = np.asarray([level.energy_wavenumber for level in self.levels])
        weight = np.asarray([level.statistical_weight for level in self.levels])
        selected = np.isfinite(energy) & (energy >= 0.0) & (weight > 0.0)
        if self.ionization_energy_ev is not None:
            cutoff = max(self.ionization_energy_ev - cutoff_below_ionization_ev, 0.0)
            selected &= energy <= cutoff * EV_TO_WAVENUMBER
        if not np.any(selected):
            raise ValueError(f"{self.element} {self.charge:+d} has no usable levels")
        selected_energy = energy[selected]
        if self.ionization_energy_ev is None:
            survival = np.ones(
                (selected_energy.size, temperature_array.size),
                dtype=np.float64,
            )
        else:
            binding_energy_ev = (
                self.ionization_energy_ev
                - selected_energy / EV_TO_WAVENUMBER
            )
            effective_n = np.sqrt(
                _RYDBERG_ENERGY_EV * (self.charge + 1.0) ** 2
                / binding_energy_ev
            )
            survival = charged_particle_hydrogen_occupation_probability(
                electron_density_array.reshape(1, -1),
                np.maximum(effective_n, 1.0).reshape(-1, 1),
                temperature_array.reshape(1, -1),
                ionic_charge=float(self.charge + 1),
            )
        exponent = (
            -WAVENUMBER_TO_ERG
            * selected_energy[:, np.newaxis]
            / (BOLTZMANN * temperature_array.reshape(1, -1))
        )
        if neutral_perturber_number_density is not None:
            if neutral_perturber_radius_cm is None:
                raise ValueError("neutral perturber radii are required with their densities")
            selected_levels = [
                level for level, keep in zip(self.levels, selected) if keep
            ]
            radii = np.asarray([
                hydrogenic_metal_level_mean_radius_cm(self, level)
                if self.ionization_energy_ev is not None
                and self.ionization_energy_ev * EV_TO_WAVENUMBER
                > level.energy_wavenumber
                else 0.0
                for level in selected_levels
            ])
            survival = survival * metal_neutral_hard_sphere_occupation_probability(
                radii[:, np.newaxis],
                {
                    name: np.asarray(value, dtype=np.float64).reshape(1, -1)
                    for name, value in neutral_perturber_number_density.items()
                },
                neutral_perturber_radius_cm,
            )
        result = np.sum(
            weight[selected, np.newaxis]
            * np.exp(exponent)
            * survival,
            axis=0,
        )
        return result.reshape(temperature_array.shape)


def strong_uv_resonance_minimum_half_window_angstrom(
    ion: AtomicIon,
    transition: AtomicTransition,
    lower_level: AtomicLevel,
) -> float:
    """Return extra formal-profile support for an optically thick resonance.

    Truncating a Voigt profile at 100 Lorentz HWHM discards less than one
    percent of its normalized area, but that is not sufficient for a strong
    ground-term resonance line: its far wing can remain optically important
    even when its fractional area is tiny.  Retain a fixed frequency-scale
    interval for strong UV resonance lines.  The wavelength-squared scaling
    follows ``d lambda = lambda**2 d nu / c`` and is normalized to the O VI
    1032/1038-A doublet validated against Werner et al. (2015).

    The restriction to wavelengths below 2000 A avoids changing optical
    resonance lines whose neutral-perturber or static-Stark profiles require
    dedicated physics (the Ca/Mg unified-profile paths are handled
    separately).  Fine-structure members within 500 cm-1 of the lowest level
    are treated as one ground term.
    """

    center = float(transition.wavelength_vacuum_angstrom)
    if not (0.0 < center <= 2_000.0):
        return 0.0
    if transition.absorption_oscillator_strength < 0.05:
        return 0.0
    ground_energy = min(level.energy_wavenumber for level in ion.levels)
    if lower_level.energy_wavenumber - ground_energy > 500.0:
        return 0.0
    return float(min(20.0, 5.0 * (center / 1_033.8) ** 2))


@dataclass(frozen=True)
class AtomicDatabase:
    """Atomic ions indexed by ``(element, charge)``."""

    ions: Mapping[tuple[str, int], AtomicIon]
    source: str = "Atomic Line List v3.00b4 Stout database (CC BY 4.0)"
    _line_selection_cache: dict[
        tuple[tuple[str, ...], float, float, float],
        tuple[tuple[AtomicIon, AtomicTransition, AtomicLevel], ...],
    ] = field(default_factory=dict, compare=False, repr=False)
    _unsold_hydrogen_coefficient_cache: dict[
        tuple[str, int, int, int], float | None
    ] = field(default_factory=dict, compare=False, repr=False)

    @property
    def elements(self) -> tuple[str, ...]:
        return tuple(sorted({key[0] for key in self.ions}))

    def ion_stages(self, element: str) -> tuple[AtomicIon, ...]:
        symbol = _canonical_element(element)
        stages = tuple(
            ion for (candidate, _), ion in sorted(
                self.ions.items(), key=lambda item: item[0][1]
            ) if candidate == symbol
        )
        if not stages or tuple(ion.charge for ion in stages) != tuple(range(len(stages))):
            raise ValueError(f"{symbol} requires consecutive ion stages beginning at neutral")
        return stages


@dataclass(frozen=True)
class VernerPhotoionizationFit:
    """Verner et al. (1996) ground-state total photoionization fit."""

    element: str
    charge: int
    threshold_energy_ev: float
    maximum_energy_ev: float
    energy_scale_ev: float
    cross_section_scale_megabar: float
    shape_a: float
    shape_p: float
    shape_w: float
    shape_0: float
    shape_1: float

    def cross_section(self, photon_energy_ev: ArrayLike) -> FloatArray:
        """Return the fitted cross section in cm2 on its published domain."""

        energy = np.asarray(photon_energy_ev, dtype=np.float64)
        if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
            raise ValueError("photon_energy_ev must be finite and positive")
        result = np.zeros_like(energy)
        valid = (
            (energy >= self.threshold_energy_ev)
            & (energy <= self.maximum_energy_ev)
        )
        if np.any(valid):
            x = energy[valid] / self.energy_scale_ev - self.shape_0
            y = np.sqrt(x**2 + self.shape_1**2)
            shape = (
                ((x - 1.0) ** 2 + self.shape_w**2)
                * y ** (0.5 * self.shape_p - 5.5)
                * (1.0 + np.sqrt(y / self.shape_a)) ** (-self.shape_p)
            )
            result[valid] = self.cross_section_scale_megabar * shape * 1.0e-18
        return result


@dataclass(frozen=True)
class VernerPhfit2Fit:
    """Total ground-state photoionization from Verner's ``phfit2`` fits.

    Sums the partial cross sections of every shell, reproducing the shell
    logic of D. A. Verner's ``phfit2.f`` (version 2, 1996): Verner, Ferland,
    Korista & Yakovlev (1996) outer-shell fits for Opacity-Project elements,
    and Verner & Yakovlev (1995) fits for inner shells and for the outer
    shells of the other elements (P, Cl, K and Sc--Zn except Fe).
    """

    element: str
    charge: int
    atomic_number: int
    electron_count: int
    shell_parameters: tuple[tuple[float, ...] | None, ...]
    outer_parameters: tuple[float, ...] | None
    outer_shell_count: int
    inner_shell_count: int

    @property
    def threshold_energy_ev(self) -> float:
        """Ground-state ionization threshold (the outermost occupied shell)."""
        return float(self.shell_parameters[self.outer_shell_count - 1][0])

    def cross_section(self, photon_energy_ev: ArrayLike) -> FloatArray:
        energy = np.asarray(photon_energy_ev, dtype=np.float64)
        if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
            raise ValueError("photon_energy_ev must be finite and positive")
        nz, ne = self.atomic_number, self.electron_count
        orbital_l = (0, 0, 1, 0, 1, 2, 0)
        non_op = nz in (15, 17, 19) or (nz > 20 and nz != 26)
        if non_op:
            inner_edge = 0.0
        elif ne < 3:
            inner_edge = np.inf
        else:
            inner_edge = self.shell_parameters[self.inner_shell_count - 1][0]
        total = np.zeros_like(energy)
        for shell in range(1, self.outer_shell_count + 1):
            params = self.shell_parameters[shell - 1]
            if params is None:
                continue
            above = energy >= params[0]
            if not np.any(above):
                continue
            e = energy[above]
            partial = np.zeros_like(e)
            use_inner_fit = (shell <= self.inner_shell_count) | (e >= inner_edge)
            skip = (
                (shell < self.outer_shell_count)
                & (shell > self.inner_shell_count)
                & (e < inner_edge)
            )
            fit_one = use_inner_fit & ~skip
            if np.any(fit_one):
                _, e0, sigma0, ya, p, yw = params
                y = e[fit_one] / e0
                q = 0.5 * p - orbital_l[shell - 1] - 5.5
                partial[fit_one] = (
                    sigma0 * ((y - 1.0) ** 2 + yw**2) * y**q
                    * (np.sqrt(y / ya) + 1.0) ** (-p)
                )
            fit_two = ~use_inner_fit & ~skip
            if np.any(fit_two) and self.outer_parameters is not None:
                e0, sigma0, ya, p, yw, y0, y1 = self.outer_parameters
                x = e[fit_two] / e0 - y0
                z = np.sqrt(x * x + y1**2)
                partial[fit_two] = (
                    sigma0 * ((x - 1.0) ** 2 + yw**2) * z ** (0.5 * p - 5.5)
                    * (1.0 + np.sqrt(z / ya)) ** (-p)
                )
            total[above] += partial
        return total * 1.0e-18


@dataclass(frozen=True)
class VernerPhotoionizationDatabase:
    """Ground-state fits indexed by ``(element, charge)``."""

    fits: Mapping[tuple[str, int], VernerPhotoionizationFit]
    source: str = "Verner et al. 1996, ApJ 465, 487"


@dataclass(frozen=True)
class MetalLTEState:
    """Trace-metal populations coupled through charge neutrality."""

    reference_species: Literal["H", "He", "metal"]
    log_number_abundance: Mapping[str, float]
    element_number_density: Mapping[str, FloatArray]
    ion_number_density: Mapping[str, FloatArray]
    partition_function: Mapping[tuple[str, int], FloatArray]
    electron_density: FloatArray
    metal_electron_density: FloatArray
    host_ion_number_density: FloatArray | None = None
    host_hydrogen_state: HydrogenLTEState | None = None
    nonideal_ionization: bool = False
    metal_level_dissolution: bool = False
    log_hydrogen_abundance: float | None = None
    trace_hydrogen_state: HydrogenLTEState | None = None
    composition_mode: Literal["trace", "bulk"] = "trace"
    mass_fraction: Mapping[str, float] | None = None
    total_mass_density: FloatArray | None = None
    # Neutral perturbers used in occupation-probability partitions; line
    # opacity applies the same level survival so populations stay normalized.
    neutral_perturber_number_density: Mapping[str, FloatArray] | None = None
    neutral_perturber_radius_cm: Mapping[str, float] | None = None


@dataclass(frozen=True)
class MgHeRedWingTable:
    """Allard et al. Mg I 2852-A red-wing cross sections at n_He=1e21.

    The CDS documentation authorizes linear density scaling only *below*
    ``1e21 cm^-3``. Above that point multiple-perturber terms change the
    profile nonlinearly. In the absence of the authors' full opacity grid we
    hold the tabulated cross section at its validity boundary; this avoids an
    unsupported linear extrapolation through dense DZ photospheres.
    """

    wavelength_by_temperature: Mapping[float, FloatArray]
    cross_section_by_temperature: Mapping[float, FloatArray]
    helium_density_cm3: float = 1.0e21
    wavelength_by_density: Mapping[float, FloatArray] | None = None
    cross_section_by_density: Mapping[float, FloatArray] | None = None
    density_profile_temperature_kelvin: float = 6000.0
    source: str = "Allard et al. 2018, A&A 619 A152, CDS J/A+A/619/A152"

    @property
    def temperatures(self) -> FloatArray:
        return np.asarray(sorted(self.wavelength_by_temperature), dtype=np.float64)

    @property
    def densities(self) -> FloatArray:
        if self.wavelength_by_density is None:
            return np.empty(0, dtype=np.float64)
        return np.asarray(sorted(self.wavelength_by_density), dtype=np.float64)

    def cross_section(
        self,
        wavelength_angstrom: ArrayLike,
        temperature: ArrayLike,
        helium_density: ArrayLike,
    ) -> FloatArray:
        """Interpolate the tabulated absolute cross section in cm^2.

        The CDS archive contains only the far red wing at four temperatures.
        Temperature interpolation is logarithmic in sigma.  If the optional
        6000-K Fig. 6 density sequence is absent, documented linear density
        scaling is applied below 1e21 cm^-3 and capped above it.  When that
        vector-recovered sequence is supplied, the profile is interpolated in
        log density from 1e21 to 1e22 cm^-3 and capped at its upper boundary.
        The CDS temperature dependence is retained as a wavelength-dependent
        ratio to its 6000-K curve.
        """

        wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
        temp = np.asarray(temperature, dtype=np.float64)
        density = np.asarray(helium_density, dtype=np.float64)
        if wavelength.ndim != 1:
            raise ValueError("wavelength_angstrom must be one dimensional")
        temp, density = np.broadcast_arrays(temp, density)
        if (
            np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0)
            or np.any(~np.isfinite(temp)) or np.any(temp <= 0.0)
            or np.any(~np.isfinite(density)) or np.any(density < 0.0)
        ):
            raise ValueError("wavelength, temperature, and density must be physical")

        grid_temperature = self.temperatures
        clipped_temperature = np.clip(temp, grid_temperature[0], grid_temperature[-1])
        result = np.zeros((wavelength.size,) + temp.shape, dtype=np.float64)
        tiny = np.finfo(np.float64).tiny
        flattened_temperature = clipped_temperature.ravel()
        for depth, local_temperature in enumerate(flattened_temperature):
            upper = int(np.searchsorted(grid_temperature, local_temperature, side="right"))
            upper = min(max(upper, 1), grid_temperature.size - 1)
            lower = upper - 1
            t_lower = float(grid_temperature[lower])
            t_upper = float(grid_temperature[upper])
            sigma_pair = []
            for table_temperature in (t_lower, t_upper):
                table_wavelength = self.wavelength_by_temperature[table_temperature]
                table_sigma = self.cross_section_by_temperature[table_temperature]
                sigma_pair.append(
                    np.interp(
                        wavelength,
                        table_wavelength,
                        table_sigma,
                        left=0.0,
                        right=0.0,
                    )
                )
            fraction = (
                (np.log(local_temperature) - np.log(t_lower))
                / (np.log(t_upper) - np.log(t_lower))
            )
            both_positive = (sigma_pair[0] > 0.0) & (sigma_pair[1] > 0.0)
            interpolated = np.where(
                both_positive,
                np.exp(
                    (1.0 - fraction) * np.log(np.maximum(sigma_pair[0], tiny))
                    + fraction * np.log(np.maximum(sigma_pair[1], tiny))
                ),
                (1.0 - fraction) * sigma_pair[0] + fraction * sigma_pair[1],
            )
            result.reshape(wavelength.size, -1)[:, depth] = interpolated
        density_scale = np.minimum(density / self.helium_density_cm3, 1.0)
        if (
            self.wavelength_by_density is None
            or self.cross_section_by_density is None
        ):
            result *= density_scale[np.newaxis, ...]
            return result

        grid_density = self.densities
        if grid_density.size < 2:
            raise ValueError("Mg I density profile needs at least two densities")
        reference_temperature = float(self.density_profile_temperature_kelvin)
        # Evaluate the CDS 6000-K reference through a table without the
        # density sequence; this avoids a second copy of the temperature
        # interpolation code.
        reference_table = replace(
            self,
            wavelength_by_density=None,
            cross_section_by_density=None,
        )
        reference_sigma = reference_table.cross_section(
            wavelength,
            np.full(temp.shape, reference_temperature),
            np.full(temp.shape, self.helium_density_cm3),
        )
        flattened_density = density.ravel()
        flat_result = result.reshape(wavelength.size, -1)
        flat_reference = reference_sigma.reshape(wavelength.size, -1)
        for depth, local_density in enumerate(flattened_density):
            binary_density_scale = (
                local_density / grid_density[0]
                if local_density < grid_density[0]
                else 1.0
            )
            clipped_density = float(np.clip(
                local_density, grid_density[0], grid_density[-1]
            ))
            upper = int(np.searchsorted(grid_density, clipped_density, side="right"))
            upper = min(max(upper, 1), grid_density.size - 1)
            lower = upper - 1
            d_lower = float(grid_density[lower])
            d_upper = float(grid_density[upper])
            sigma_pair = []
            for table_density in (d_lower, d_upper):
                table_wavelength = self.wavelength_by_density[table_density]
                table_sigma = self.cross_section_by_density[table_density]
                sigma_pair.append(np.interp(
                    wavelength, table_wavelength, table_sigma,
                    left=0.0, right=0.0,
                ))
            fraction = (
                (np.log(clipped_density) - np.log(d_lower))
                / (np.log(d_upper) - np.log(d_lower))
            )
            both_positive = (sigma_pair[0] > 0.0) & (sigma_pair[1] > 0.0)
            density_profile = np.where(
                both_positive,
                np.exp(
                    (1.0 - fraction) * np.log(np.maximum(sigma_pair[0], tiny))
                    + fraction * np.log(np.maximum(sigma_pair[1], tiny))
                ),
                (1.0 - fraction) * sigma_pair[0] + fraction * sigma_pair[1],
            )
            reference = flat_reference[:, depth]
            temperature_ratio = np.ones_like(reference)
            valid_ratio = reference > 0.0
            temperature_ratio[valid_ratio] = (
                flat_result[valid_ratio, depth] / reference[valid_ratio]
            )
            flat_result[:, depth] = (
                density_profile * temperature_ratio * binary_density_scale
            )
        return result


@dataclass(frozen=True)
class MgIIHeProfileTable:
    """Absolute cross section for the summed Mg II h/k lines in helium.

    Each instance represents one published temperature/density condition.
    The impact-limit density scaling is linear below that condition and held
    at the tabulated value above it, where a linear extrapolation would omit
    multiple-perturber terms. Temperature is not extrapolated, making these
    optional, provenance-explicit bridges until the authors' multidimensional
    profile grid is publicly available.
    """

    wavelength_angstrom: FloatArray
    cross_section_cm2: FloatArray
    temperature_kelvin: float = 8000.0
    helium_density_cm3: float = 2.0e21
    source: str = (
        "Allard et al. 2018 A&A 619 A152 Fig. 7 vector data; "
        "Mg II calculation from Allard et al. 2016 A&A 593 A13"
    )
    continue_fitted_log_linear_tails: bool = False

    def cross_section(
        self,
        wavelength_angstrom: ArrayLike,
        helium_density: ArrayLike,
    ) -> FloatArray:
        wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
        density = np.asarray(helium_density, dtype=np.float64)
        if wavelength.ndim != 1:
            raise ValueError("wavelength_angstrom must be one dimensional")
        if (
            np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0)
            or np.any(~np.isfinite(density)) or np.any(density < 0.0)
        ):
            raise ValueError("wavelength and helium density must be physical")
        sigma = np.interp(
            wavelength, self.wavelength_angstrom, self.cross_section_cm2,
            left=0.0, right=0.0,
        )
        if self.continue_fitted_log_linear_tails:
            # The figure is clipped while both wings are still nonzero.
            # Continue its local logarithmic slope, rather than treating the
            # panel limits as opacity edges or imposing an unrelated power
            # law. The final ordinary profile supplies the asymptotic floor.
            count = min(12, self.wavelength_angstrom.size)
            blue_slope = float(np.polyfit(
                self.wavelength_angstrom[:count],
                np.log(np.maximum(
                    self.cross_section_cm2[:count],
                    np.finfo(np.float64).tiny,
                )),
                1,
            )[0])
            red_slope = float(np.polyfit(
                self.wavelength_angstrom[-count:],
                np.log(np.maximum(
                    self.cross_section_cm2[-count:],
                    np.finfo(np.float64).tiny,
                )),
                1,
            )[0])
            blue = wavelength < self.wavelength_angstrom[0]
            red = wavelength > self.wavelength_angstrom[-1]
            if np.any(blue):
                sigma[blue] = self.cross_section_cm2[0] * (
                    np.exp(np.clip(
                        max(blue_slope, 0.0)
                        * (wavelength[blue] - self.wavelength_angstrom[0]),
                        -745.0,
                        0.0,
                    ))
                )
            if np.any(red):
                sigma[red] = self.cross_section_cm2[-1] * (
                    np.exp(np.clip(
                        min(red_slope, 0.0)
                        * (wavelength[red] - self.wavelength_angstrom[-1]),
                        -745.0,
                        0.0,
                    ))
                )
        density_scale = np.minimum(
            density[np.newaxis, ...] / self.helium_density_cm3,
            1.0,
        )
        return sigma.reshape((-1,) + (1,) * density.ndim) * density_scale


@dataclass(frozen=True)
class CaIIHeProfileTable(MgIIHeProfileTable):
    """Summed Ca II H&K profile at the condition plotted by Blouin et al."""

    temperature_kelvin: float = 6000.0
    helium_density_cm3: float = 1.0e22
    source: str = "Blouin, Dufour & Allard 2018, ApJ 863, 184, Fig. 1 vector data"
    wavelength_by_temperature: Mapping[float, FloatArray] | None = None
    cross_section_by_temperature: Mapping[float, FloatArray] | None = None
    wavelength_by_condition: Mapping[tuple[float, float], FloatArray] | None = None
    cross_section_by_condition: Mapping[tuple[float, float], FloatArray] | None = None

    def cross_section(
        self,
        wavelength_angstrom: ArrayLike,
        helium_density: ArrayLike,
        temperature: ArrayLike | None = None,
    ) -> FloatArray:
        """Interpolate the vector curve with continuous fitted tails.

        The plotted panel ends at 3700 and 4300 A while the red curve remains
        nonzero at both boundaries.  Those are plot limits, not opacity edges.
        Continue the local logarithmic slope at each endpoint so the optional
        figure bridge cannot introduce flux steps. The ordinary impact
        profile supplies the asymptotic opacity outside this bridge.
        """

        wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
        density = np.asarray(helium_density, dtype=np.float64)
        local_temperature = (
            None if temperature is None
            else np.asarray(temperature, dtype=np.float64)
        )
        if wavelength.ndim != 1:
            raise ValueError("wavelength_angstrom must be one dimensional")
        if (
            np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0)
            or np.any(~np.isfinite(density)) or np.any(density < 0.0)
            or (
                local_temperature is not None
                and (
                    np.any(~np.isfinite(local_temperature))
                    or np.any(local_temperature <= 0.0)
                )
            )
        ):
            raise ValueError(
                "wavelength, helium density, and temperature must be physical"
            )

        if (
            self.wavelength_by_condition is not None
            and self.cross_section_by_condition is not None
            and local_temperature is not None
        ):
            local_temperature, density = np.broadcast_arrays(
                local_temperature, density
            )
            conditions = tuple(self.wavelength_by_condition)
            grid_temperature = np.asarray(
                sorted({condition[0] for condition in conditions}),
                dtype=np.float64,
            )
            grid_density = np.asarray(
                sorted({condition[1] for condition in conditions}),
                dtype=np.float64,
            )
            if grid_temperature.size < 2 or grid_density.size < 2:
                raise ValueError(
                    "Ca II--He condition grid needs at least two temperatures "
                    "and two densities"
                )
            expected = {
                (float(grid_t), float(grid_n))
                for grid_t in grid_temperature
                for grid_n in grid_density
            }
            if set(conditions) != expected:
                raise ValueError("Ca II--He condition grid must be rectangular")
            clipped_temperature = np.clip(
                local_temperature, grid_temperature[0], grid_temperature[-1]
            )
            clipped_density = np.clip(
                density, grid_density[0], grid_density[-1]
            )
            result = np.zeros(
                (wavelength.size,) + local_temperature.shape,
                dtype=np.float64,
            )
            flat_result = result.reshape(wavelength.size, -1)
            tiny = np.finfo(np.float64).tiny

            def bracket(grid: FloatArray, value: float) -> tuple[int, int, float]:
                upper = int(np.searchsorted(grid, value, side="right"))
                upper = min(max(upper, 1), grid.size - 1)
                lower = upper - 1
                fraction = (
                    (np.log(value) - np.log(grid[lower]))
                    / (np.log(grid[upper]) - np.log(grid[lower]))
                )
                return lower, upper, float(fraction)

            for depth, (depth_temperature, depth_density) in enumerate(zip(
                clipped_temperature.ravel(), clipped_density.ravel()
            )):
                t0, t1, ft = bracket(grid_temperature, float(depth_temperature))
                n0, n1, fn = bracket(grid_density, float(depth_density))
                corner_sigma = []
                corner_weight = []
                for ti, tw in ((t0, 1.0 - ft), (t1, ft)):
                    for ni, nw in ((n0, 1.0 - fn), (n1, fn)):
                        condition = (
                            float(grid_temperature[ti]),
                            float(grid_density[ni]),
                        )
                        corner_sigma.append(np.interp(
                            wavelength,
                            self.wavelength_by_condition[condition],
                            self.cross_section_by_condition[condition],
                            left=0.0,
                            right=0.0,
                        ))
                        corner_weight.append(tw * nw)
                corner = np.asarray(corner_sigma)
                weight = np.asarray(corner_weight)[:, np.newaxis]
                all_positive = np.all(corner > 0.0, axis=0)
                flat_result[:, depth] = np.where(
                    all_positive,
                    np.exp(np.sum(weight * np.log(np.maximum(corner, tiny)), axis=0)),
                    np.sum(weight * corner, axis=0),
                )
            # Below the least-dense calculation, retain the binary-collision
            # limit. Above the largest published density, clamp rather than
            # extrapolating into an uncalculated multiple-perturber regime.
            low_density_scale = np.minimum(density / grid_density[0], 1.0)
            return result * low_density_scale[np.newaxis, ...]

        if (
            self.wavelength_by_temperature is not None
            and self.cross_section_by_temperature is not None
            and local_temperature is not None
        ):
            local_temperature, density = np.broadcast_arrays(
                local_temperature, density
            )
            grid_temperature = np.asarray(
                sorted(self.wavelength_by_temperature), dtype=np.float64
            )
            if grid_temperature.size < 2:
                raise ValueError(
                    "Ca II--He temperature grid needs at least two temperatures"
                )
            clipped_temperature = np.clip(
                local_temperature,
                grid_temperature[0],
                grid_temperature[-1],
            )
            result = np.zeros(
                (wavelength.size,) + local_temperature.shape,
                dtype=np.float64,
            )
            flat_result = result.reshape(wavelength.size, -1)
            tiny = np.finfo(np.float64).tiny
            for depth, depth_temperature in enumerate(
                clipped_temperature.ravel()
            ):
                upper = int(np.searchsorted(
                    grid_temperature, depth_temperature, side="right"
                ))
                upper = min(max(upper, 1), grid_temperature.size - 1)
                lower = upper - 1
                lower_temperature = float(grid_temperature[lower])
                upper_temperature = float(grid_temperature[upper])
                sigma_pair = []
                for table_temperature in (
                    lower_temperature, upper_temperature
                ):
                    sigma_pair.append(np.interp(
                        wavelength,
                        self.wavelength_by_temperature[table_temperature],
                        self.cross_section_by_temperature[table_temperature],
                        left=0.0,
                        right=0.0,
                    ))
                fraction = (
                    (np.log(depth_temperature) - np.log(lower_temperature))
                    / (
                        np.log(upper_temperature)
                        - np.log(lower_temperature)
                    )
                )
                both_positive = (
                    (sigma_pair[0] > 0.0) & (sigma_pair[1] > 0.0)
                )
                flat_result[:, depth] = np.where(
                    both_positive,
                    np.exp(
                        (1.0 - fraction)
                        * np.log(np.maximum(sigma_pair[0], tiny))
                        + fraction
                        * np.log(np.maximum(sigma_pair[1], tiny))
                    ),
                    (1.0 - fraction) * sigma_pair[0]
                    + fraction * sigma_pair[1],
                )
            density_scale = np.minimum(
                density / self.helium_density_cm3, 1.0
            )
            return result * density_scale[np.newaxis, ...]

        sigma = np.interp(
            wavelength, self.wavelength_angstrom, self.cross_section_cm2
        )
        count = min(12, self.wavelength_angstrom.size)
        blue_slope = float(np.polyfit(
            self.wavelength_angstrom[:count],
            np.log(np.maximum(
                self.cross_section_cm2[:count], np.finfo(np.float64).tiny
            )),
            1,
        )[0])
        red_slope = float(np.polyfit(
            self.wavelength_angstrom[-count:],
            np.log(np.maximum(
                self.cross_section_cm2[-count:], np.finfo(np.float64).tiny
            )),
            1,
        )[0])
        blue = wavelength < self.wavelength_angstrom[0]
        red = wavelength > self.wavelength_angstrom[-1]
        if np.any(blue):
            sigma[blue] = self.cross_section_cm2[0] * (
                np.exp(np.clip(
                    max(blue_slope, 0.0)
                    * (wavelength[blue] - self.wavelength_angstrom[0]),
                    -745.0,
                    0.0,
                ))
            )
        if np.any(red):
            sigma[red] = self.cross_section_cm2[-1] * (
                np.exp(np.clip(
                    min(red_slope, 0.0)
                    * (wavelength[red] - self.wavelength_angstrom[-1]),
                    -745.0,
                    0.0,
                ))
            )
        density_scale = np.minimum(
            density[np.newaxis, ...] / self.helium_density_cm3,
            1.0,
        )
        return sigma.reshape((-1,) + (1,) * density.ndim) * density_scale


@dataclass(frozen=True)
class CaIHeProfileTable:
    """Published unified Ca I 4227-A profiles in dense helium.

    The public article source contains original vector curves for five helium
    densities at 4000 K and four temperatures at ``n_He=5e21 cm^-3``. Each
    curve is an absolute cross section including the oscillator strength. A
    local temperature-density profile is reconstructed from those orthogonal
    published sequences: the 4000-K density profile is multiplied by the
    wavelength-dependent temperature ratio at the common reference density.
    This factorized interpolation exactly recovers every input slice and
    introduces no fitted scale; it is not a rectangular ab-initio grid.

    Below the least-dense calculation the binary-collision linear scaling is
    used. Temperature and density are clamped at the published boundaries
    rather than extrapolated into uncomputed multiple-perturber conditions.
    """

    wavelength_by_density: Mapping[float, FloatArray]
    cross_section_by_density: Mapping[float, FloatArray]
    temperature_kelvin: float = 4000.0
    wavelength_by_temperature: Mapping[float, FloatArray] | None = None
    cross_section_by_temperature: Mapping[float, FloatArray] | None = None
    temperature_profile_density_cm3: float = 5.0e21
    source: str = "Blouin et al. 2019 Ca I--He Figs. 5--6 vector data"

    @property
    def densities(self) -> FloatArray:
        return np.asarray(sorted(self.wavelength_by_density), dtype=np.float64)

    @property
    def temperatures(self) -> FloatArray:
        if self.wavelength_by_temperature is None:
            return np.empty(0, dtype=np.float64)
        return np.asarray(
            sorted(self.wavelength_by_temperature), dtype=np.float64
        )

    def cross_section(
        self,
        wavelength_angstrom: ArrayLike,
        helium_density: ArrayLike,
        temperature: ArrayLike | None = None,
    ) -> FloatArray:
        wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
        temperature = np.asarray(
            self.temperature_kelvin if temperature is None else temperature,
            dtype=np.float64,
        )
        density = np.asarray(helium_density, dtype=np.float64)
        if wavelength.ndim != 1:
            raise ValueError("wavelength_angstrom must be one dimensional")
        if (
            np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0)
            or np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0)
            or np.any(~np.isfinite(density)) or np.any(density < 0.0)
        ):
            raise ValueError(
                "wavelength, temperature, and helium density must be physical"
            )
        temperature, density = np.broadcast_arrays(temperature, density)

        grid = self.densities

        def profile_at_density(table_density: float) -> FloatArray:
            table_wavelength = self.wavelength_by_density[table_density]
            table_cross_section = self.cross_section_by_density[table_density]
            sigma = np.interp(wavelength, table_wavelength, table_cross_section)
            center = float(table_wavelength[np.argmax(table_cross_section)])
            blue = wavelength < table_wavelength[0]
            red = wavelength > table_wavelength[-1]
            if np.any(blue):
                separation = max(center - table_wavelength[0], 1.0)
                sigma[blue] = table_cross_section[0] * (
                    separation
                    / np.maximum(center - wavelength[blue], separation)
                ) ** 2
            if np.any(red):
                separation = max(table_wavelength[-1] - center, 1.0)
                sigma[red] = table_cross_section[-1] * (
                    separation
                    / np.maximum(wavelength[red] - center, separation)
                ) ** 2
            return sigma

        result = np.zeros((wavelength.size,) + density.shape, dtype=np.float64)
        flattened_result = result.reshape(wavelength.size, -1)
        for index, local_density in enumerate(density.ravel()):
            if local_density == 0.0:
                continue
            upper = int(np.searchsorted(grid, local_density, side="right"))
            if upper == 0:
                table_density = float(grid[0])
                sigma = profile_at_density(table_density)
                flattened_result[:, index] = sigma * local_density / table_density
                continue
            if upper >= grid.size:
                table_density = float(grid[-1])
                flattened_result[:, index] = profile_at_density(table_density)
                continue
            lower_density = float(grid[upper - 1])
            upper_density = float(grid[upper])
            lower_sigma = profile_at_density(lower_density)
            upper_sigma = profile_at_density(upper_density)
            fraction = (
                np.log(local_density) - np.log(lower_density)
            ) / (np.log(upper_density) - np.log(lower_density))
            overlap = (lower_sigma > 0.0) & (upper_sigma > 0.0)
            interpolated = (
                (1.0 - fraction) * lower_sigma + fraction * upper_sigma
            )
            interpolated[overlap] = np.exp(
                (1.0 - fraction) * np.log(lower_sigma[overlap])
                + fraction * np.log(upper_sigma[overlap])
            )
            flattened_result[:, index] = interpolated

        if (
            self.wavelength_by_temperature is None
            or self.cross_section_by_temperature is None
        ):
            return result

        grid_temperature = self.temperatures
        if grid_temperature.size < 2:
            raise ValueError("Ca I--He temperature sequence needs two temperatures")
        if self.temperature_kelvin not in self.wavelength_by_temperature:
            raise ValueError(
                "Ca I--He temperature sequence must include the density-grid "
                "reference temperature"
            )
        tiny = np.finfo(np.float64).tiny

        def temperature_profile(table_temperature: float) -> FloatArray:
            local_wavelength = self.wavelength_by_temperature[table_temperature]
            local_cross_section = self.cross_section_by_temperature[
                table_temperature
            ]
            # Clamp the *temperature ratio* at the plotted wavelength bounds.
            # The density profile still supplies the decaying absolute wings;
            # this merely avoids turning a figure boundary into an opacity edge.
            return np.interp(
                wavelength,
                local_wavelength,
                local_cross_section,
                left=local_cross_section[0],
                right=local_cross_section[-1],
            )

        reference_profile = temperature_profile(self.temperature_kelvin)
        flat_temperature = np.clip(
            temperature, grid_temperature[0], grid_temperature[-1]
        ).ravel()
        for index, local_temperature in enumerate(flat_temperature):
            if np.isclose(local_temperature, self.temperature_kelvin):
                continue
            upper = int(np.searchsorted(
                grid_temperature, local_temperature, side="right"
            ))
            upper = min(max(upper, 1), grid_temperature.size - 1)
            lower = upper - 1
            t_lower = float(grid_temperature[lower])
            t_upper = float(grid_temperature[upper])
            lower_sigma = temperature_profile(t_lower)
            upper_sigma = temperature_profile(t_upper)
            fraction = (
                (np.log(local_temperature) - np.log(t_lower))
                / (np.log(t_upper) - np.log(t_lower))
            )
            overlap = (lower_sigma > 0.0) & (upper_sigma > 0.0)
            interpolated = (
                (1.0 - fraction) * lower_sigma + fraction * upper_sigma
            )
            interpolated[overlap] = np.exp(
                (1.0 - fraction) * np.log(np.maximum(lower_sigma[overlap], tiny))
                + fraction * np.log(np.maximum(upper_sigma[overlap], tiny))
            )
            valid_ratio = reference_profile > 0.0
            temperature_ratio = np.ones_like(reference_profile)
            temperature_ratio[valid_ratio] = (
                interpolated[valid_ratio] / reference_profile[valid_ratio]
            )
            flattened_result[:, index] *= temperature_ratio
        return result


def _canonical_element(element: str) -> str:
    symbol = element.strip().capitalize()
    if symbol not in ATOMIC_MASS_U:
        raise ValueError(f"unsupported element {element!r}; choose {tuple(ATOMIC_MASS_U)}")
    return symbol


def chondritic_metal_abundances(
    log_ca_over_host: float,
    *,
    overrides: Mapping[str, float] | None = None,
    elements: Iterable[str] | None = None,
) -> dict[str, float]:
    """Return a supported CI-chondrite mixture on a Ca/host scale.

    Parameters are logarithmic number ratios relative to the atmosphere's
    host species (normally He). ``overrides`` is intended for elements whose
    photospheric abundances were measured independently. This reproduces the
    composition convention used for otherwise unconstrained elements in the
    Blouin et al. (2018) Ross 640 and LP 658-2 models, rather than silently
    treating omitted metals as absent.
    """

    if not np.isfinite(log_ca_over_host):
        raise ValueError("log_ca_over_host must be finite")
    requested = (
        tuple(CHONDRITIC_LOG_NUMBER_RATIO_TO_CA)
        if elements is None
        else tuple(_canonical_element(value) for value in elements)
    )
    unsupported = sorted(
        set(requested) - set(CHONDRITIC_LOG_NUMBER_RATIO_TO_CA)
    )
    if unsupported:
        raise ValueError(
            "no Lodders CI-chondrite ratio is available for "
            + ", ".join(unsupported)
        )
    result = {
        element: float(
            log_ca_over_host + CHONDRITIC_LOG_NUMBER_RATIO_TO_CA[element]
        )
        for element in requested
    }
    for element, abundance in (overrides or {}).items():
        symbol = _canonical_element(element)
        if symbol not in CHONDRITIC_LOG_NUMBER_RATIO_TO_CA:
            raise ValueError(f"unsupported chondritic abundance override: {symbol}")
        if not np.isfinite(abundance):
            raise ValueError(f"abundance override for {symbol} must be finite")
        if symbol not in result:
            raise ValueError(
                f"abundance override for {symbol} is not among the requested elements"
            )
        result[symbol] = float(abundance)
    return result


def _level_term_label(label: str) -> str:
    """Return a Stout level label with its ``<J>`` value removed."""

    return re.sub(r"<[^>]+>\)", ")", label)


def ground_term_levels(ion: AtomicIon) -> tuple[AtomicLevel, ...]:
    """Return every fine-structure level of an ion's LS ground term.

    Levels are grouped by their Stout label with the ``<J>`` value removed.
    A ground label without a parseable ``J`` returns only the lowest level.
    """

    levels = [
        level for level in ion.levels
        if np.isfinite(level.energy_wavenumber) and level.statistical_weight > 0.0
    ]
    if not levels:
        return ()
    ground = min(levels, key=lambda level: level.energy_wavenumber)
    if re.search(r"<[^>]+>\)", ground.label) is None:
        return (ground,)
    term = _level_term_label(ground.label)
    return tuple(
        level for level in levels if _level_term_label(level.label) == term
    )


def dense_helium_ionization_potential_shift_ev(
    element: str,
    mass_density: ArrayLike,
    temperature: ArrayLike,
) -> FloatArray:
    r"""Return Blouin et al.'s first-ionization ``Delta I`` in eV.

    The published fit covers 0--1.5 g cm^-3 and 4000--8000 K and was tested
    by its authors over 2000--10000 K.  Outside that tested domain the fit is
    held at the nearest tested temperature and at the largest fitted density
    rather than extrapolated.  Holding the edge value keeps the ionization
    balance continuous; switching the shift off at 10000 K made ion fractions
    jump by up to 0.7 dex at fixed density.  ``Delta I`` still vanishes as the
    density goes to zero.  Positive shifts are capped at zero exactly as in
    the paper.
    """

    symbol = _canonical_element(element)
    density, temp = np.broadcast_arrays(
        np.asarray(mass_density, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
    )
    if np.any(~np.isfinite(density)) or np.any(density < 0.0) or np.any(~np.isfinite(temp)):
        raise ValueError("mass_density must be non-negative and inputs finite")
    if symbol not in _DENSE_HE_IONIZATION_FIT:
        return np.zeros_like(density)
    a_coefficient, b_coefficient, c_coefficient = _DENSE_HE_IONIZATION_FIT[symbol]
    temp = np.clip(temp, 2000.0, 10_000.0)
    density = np.minimum(density, 1.5)
    return np.minimum(
        0.0,
        (a_coefficient + b_coefficient * temp) * density
        + c_coefficient * density**2,
    )


def _resolve_stout_ion_directory(root: Path, element: str, stage: int) -> Path:
    lower = element.lower()
    name = f"{lower}_{stage}"
    candidates = (
        root / "stout" / lower / name,
        root / lower / name,
        root / name,
    )
    for candidate in candidates:
        if (candidate / f"{name}.nrg").is_file() and (candidate / f"{name}.tp").is_file():
            return candidate
    raise FileNotFoundError(f"could not find Stout ion {name!r} below {root}")


def read_stout_atomic_ion(root: str | Path, element: str, charge: int) -> AtomicIon:
    """Read one Stout ion; ``charge=0`` corresponds to the ``*_1`` files."""

    symbol = _canonical_element(element)
    if not isinstance(charge, (int, np.integer)) or charge < 0:
        raise ValueError("charge must be a non-negative integer")
    atomic_number = ATOMIC_NUMBER[symbol]
    if charge > atomic_number:
        raise ValueError(f"charge cannot exceed Z={atomic_number} for {symbol}")
    if charge == atomic_number:
        return AtomicIon(
            element=symbol,
            charge=int(charge),
            atomic_mass_u=ATOMIC_MASS_U[symbol],
            ionization_energy_ev=None,
            levels=(AtomicLevel(1, 0.0, 1.0, "bare nucleus"),),
            transitions=(),
            source="NIST ASD ionization ladder; synthetic bare-nucleus closure",
        )
    stage = int(charge) + 1
    directory = _resolve_stout_ion_directory(Path(root), symbol, stage)
    stem = f"{symbol.lower()}_{stage}"

    levels: list[AtomicLevel] = []
    with (directory / f"{stem}.nrg").open(encoding="utf-8") as stream:
        next(stream, None)
        for line in stream:
            fields = line.strip().split(maxsplit=3)
            if len(fields) < 3:
                continue
            try:
                index = int(fields[0])
                energy = float(fields[1])
                weight = float(fields[2])
            except ValueError:
                continue
            label = fields[3].strip().strip('"') if len(fields) == 4 else ""
            levels.append(AtomicLevel(index, energy, weight, label))
    if not levels:
        raise ValueError(f"no levels parsed from {directory / f'{stem}.nrg'}")
    level_by_index = {level.index: level for level in levels}

    transitions: list[AtomicTransition] = []
    with (directory / f"{stem}.tp").open(encoding="utf-8") as stream:
        next(stream, None)
        for line in stream:
            fields = line.split()
            if len(fields) < 5 or fields[0] != "A":
                continue
            try:
                lower_index = int(fields[1])
                upper_index = int(fields[2])
                einstein_a = float(fields[3])
                lower = level_by_index[lower_index]
                upper = level_by_index[upper_index]
            except (ValueError, KeyError):
                continue
            delta_wavenumber = upper.energy_wavenumber - lower.energy_wavenumber
            if delta_wavenumber <= 0.0 or einstein_a <= 0.0:
                continue
            wavelength_cm = 1.0 / delta_wavenumber
            oscillator_strength = (
                ELECTRON_MASS * LIGHT_SPEED * wavelength_cm**2
                / (8.0 * PI**2 * ELEMENTARY_CHARGE_ESU**2)
                * upper.statistical_weight / lower.statistical_weight
                * einstein_a
            )
            transitions.append(
                AtomicTransition(
                    lower_index,
                    upper_index,
                    einstein_a,
                    fields[4],
                    wavelength_cm * 1.0e8,
                    oscillator_strength,
                )
            )
    ionization = IONIZATION_ENERGY_EV[symbol]
    return AtomicIon(
        element=symbol,
        charge=int(charge),
        atomic_mass_u=ATOMIC_MASS_U[symbol],
        ionization_energy_ev=(ionization[charge] if charge < len(ionization) else None),
        levels=tuple(levels),
        transitions=tuple(transitions),
    )


def read_stout_atomic_database(
    root: str | Path,
    *,
    elements: Iterable[str] = tuple(ATOMIC_MASS_U),
    maximum_charge: int | Mapping[str, int] = 2,
) -> AtomicDatabase:
    """Read consecutive ion stages for the requested elements.

    ``maximum_charge`` may be one common charge or an element-to-charge
    mapping.  Requesting charge ``Z`` appends a one-level bare nucleus, which
    closes the Saha ladder without requiring a non-existent Stout file.
    """

    if isinstance(maximum_charge, Mapping):
        maximum_by_element = {
            _canonical_element(element): int(charge)
            for element, charge in maximum_charge.items()
        }
    else:
        if maximum_charge < 1:
            raise ValueError("maximum_charge must be at least one")
        maximum_by_element = {}
    ions: dict[tuple[str, int], AtomicIon] = {}
    for element in elements:
        symbol = _canonical_element(element)
        if isinstance(maximum_charge, Mapping):
            if symbol not in maximum_by_element:
                raise ValueError(f"maximum_charge mapping is missing {symbol}")
            local_maximum = maximum_by_element[symbol]
        else:
            local_maximum = int(maximum_charge)
        if local_maximum < 1 or local_maximum > ATOMIC_NUMBER[symbol]:
            raise ValueError(
                f"maximum charge for {symbol} must lie between 1 and "
                f"{ATOMIC_NUMBER[symbol]}"
            )
        for charge in range(local_maximum + 1):
            ion = read_stout_atomic_ion(root, symbol, charge)
            ions[(symbol, charge)] = ion
    return AtomicDatabase(MappingProxyType(ions))


def augment_oxygen_i_6258_6271_multiplet(
    database: AtomicDatabase,
) -> AtomicDatabase:
    """Add the complete Kurucz O I 6258--6271-A multiplet to Stout data.

    Stout contains the participating O I levels but no transitions between
    them.  This function adds only those eight missing fine-structure
    components and is idempotent.  It deliberately does not import the full
    Kurucz O I replacement, which degraded independent blue-optical regions
    in the D6 validation.
    """

    key = ("O", 0)
    if key not in database.ions:
        return database
    ion = database.ions[key]

    def level_at(energy: float) -> AtomicLevel:
        matches = [
            level
            for level in ion.levels
            if abs(level.energy_wavenumber - energy) <= 0.02
        ]
        if len(matches) != 1:
            raise ValueError(
                "the O I 6258--6271-A supplement requires one Stout level "
                f"at {energy:.3f} cm^-1; found {len(matches)}"
            )
        return matches[0]

    existing_pairs = {
        (transition.lower_index, transition.upper_index)
        for transition in ion.transitions
    }
    added: list[AtomicTransition] = []
    for lower_energy, upper_energy, oscillator_strength in (
        _O_I_6258_6271_MULTIPLET
    ):
        lower = level_at(lower_energy)
        upper = level_at(upper_energy)
        pair = (lower.index, upper.index)
        if pair in existing_pairs:
            continue
        wavelength_cm = 1.0 / (upper_energy - lower_energy)
        einstein_a = (
            8.0 * PI**2 * ELEMENTARY_CHARGE_ESU**2
            / (ELECTRON_MASS * LIGHT_SPEED * wavelength_cm**2)
            * lower.statistical_weight / upper.statistical_weight
            * oscillator_strength
        )
        added.append(
            AtomicTransition(
                lower_index=lower.index,
                upper_index=upper.index,
                einstein_a=einstein_a,
                transition_type="E1",
                wavelength_vacuum_angstrom=wavelength_cm * 1.0e8,
                absorption_oscillator_strength=oscillator_strength,
                radiative_damping_rate_s=(
                    _O_I_6258_6271_RADIATIVE_DAMPING_RATE_S
                ),
            )
        )
        existing_pairs.add(pair)
    if not added:
        return database

    ions = dict(database.ions)
    ions[key] = replace(
        ion,
        transitions=ion.transitions + tuple(added),
        source=f"{ion.source}; Kurucz GF100 O I 6258--6271-A supplement",
    )
    return AtomicDatabase(
        MappingProxyType(ions),
        source=f"{database.source}; Kurucz GF100 O I 6258--6271-A supplement",
    )


def atomic_database_with_melendez_barbuy_feii_oscillator_strengths(
    database: AtomicDatabase,
) -> AtomicDatabase:
    """Apply the Melendez--Barbuy (2009) optical Fe II ``gf`` values.

    Their 142-line compilation combines laboratory normalization with
    theoretical relative strengths within each multiplet and was validated
    against very high signal-to-noise stellar spectra.  Klein et al. (2011)
    used the same values through VALD for PG 1225-079.  Only transitions that
    match both wavelength and lower-level excitation energy are replaced;
    all unmatched Stout transitions and all Stout levels are retained.  The
    Einstein A value is rescaled with the oscillator strength so natural
    damping remains internally consistent.  The accompanying published C6
    values are bundled for provenance but are not applied here because the
    conversion to a He-perturber impact width is a separate profile choice.
    """

    key = ("Fe", 1)
    if key not in database.ions:
        return database
    if "Melendez & Barbuy (2009)" in database.ions[key].source:
        return database
    resource = files("wd_spectra").joinpath(
        "data/atomic/feii_melendez_barbuy_2009.dat"
    )
    records: list[tuple[float, float, float]] = []
    with resource.open("r", encoding="ascii") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("malformed Melendez--Barbuy Fe II data row")
            air_wavelength, lower_excitation, log_gf = map(float, fields[:3])
            records.append((
                _physical_air_to_vacuum_scalar(air_wavelength),
                lower_excitation,
                log_gf,
            ))

    ion = database.ions[key]
    levels = {level.index: level for level in ion.levels}
    updated = list(ion.transitions)
    used_transition_indices: set[int] = set()
    matched_records = 0
    for wavelength, lower_excitation, log_gf in records:
        candidates: list[int] = []
        for index, transition in enumerate(ion.transitions):
            if index in used_transition_indices:
                continue
            lower = levels.get(transition.lower_index)
            if lower is None:
                continue
            lower_excitation_stout = (
                lower.energy_wavenumber / EV_TO_WAVENUMBER
            )
            if (
                abs(transition.wavelength_vacuum_angstrom - wavelength) <= 0.04
                and abs(lower_excitation_stout - lower_excitation) <= 0.015
            ):
                candidates.append(index)
        if len(candidates) != 1:
            continue
        index = candidates[0]
        transition = ion.transitions[index]
        lower_weight = levels[transition.lower_index].statistical_weight
        oscillator_strength = 10.0**log_gf / lower_weight
        scale = (
            oscillator_strength / transition.absorption_oscillator_strength
        )
        updated[index] = replace(
            transition,
            absorption_oscillator_strength=oscillator_strength,
            einstein_a=transition.einstein_a * scale,
        )
        used_transition_indices.add(index)
        matched_records += 1

    if matched_records == 0:
        return database
    ions = dict(database.ions)
    ions[key] = replace(
        ion,
        transitions=tuple(updated),
        source=(
            f"{ion.source}; {matched_records}/142 optical Fe II gf values "
            "from Melendez & Barbuy (2009)"
        ),
    )
    return AtomicDatabase(
        MappingProxyType(ions),
        source=(
            f"{database.source}; {matched_records}/142 optical Fe II gf "
            "values from Melendez & Barbuy (2009)"
        ),
    )


def _physical_air_to_vacuum_scalar(wavelength_air_angstrom: float) -> float:
    """Convert one ordinary optical air wavelength to vacuum."""

    vacuum = float(wavelength_air_angstrom)
    for _ in range(5):
        inverse_wavelength_squared = vacuum**-2
        refractive_index = (
            1.0
            + 2.735182e-4
            + 131.4182 * inverse_wavelength_squared
            + 2.76249e8 * inverse_wavelength_squared**2
        )
        vacuum = wavelength_air_angstrom * refractive_index
    return vacuum


def read_barklem_neutral_hydrogen_broadening(
    path: str | Path,
    base_database: AtomicDatabase,
    *,
    maximum_wavelength_difference_angstrom: float = 0.08,
    maximum_energy_difference_wavenumber: float = 2.0,
) -> AtomicDatabase:
    """Attach line-by-line ABO neutral-H damping data to a database.

    Barklem, Piskunov & O'Mara (2000) tabulate the Lorentz FWHM in angular
    frequency per neutral-H perturber at 10,000 K and a transition-specific
    temperature exponent. Records are matched by element, ion stage, vacuum
    wavelength, fine-structure ``J``, and, where present, both level energies.
    Oscillator strengths and level definitions remain those of
    ``base_database``. Unmatched lines retain the classical Unsold fallback.
    """

    if (
        not np.isfinite(maximum_wavelength_difference_angstrom)
        or maximum_wavelength_difference_angstrom <= 0.0
        or not np.isfinite(maximum_energy_difference_wavenumber)
        or maximum_energy_difference_wavenumber <= 0.0
    ):
        raise ValueError("Barklem matching tolerances must be finite and positive")
    number_to_element = {
        number: element for element, number in ATOMIC_NUMBER.items()
    }
    records: dict[
        tuple[str, int],
        list[tuple[float, float, float, float, float, float, float]],
    ] = {}
    source_path = Path(path)
    with source_path.open("r", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 16:
                raise ValueError(
                    f"{source_path}:{line_number}: expected 16 Barklem columns"
                )
            try:
                species = float(fields[0])
                atomic_number = int(np.floor(species + 1.0e-8))
                charge = int(round(100.0 * (species - atomic_number)))
                wavelength_vacuum = _physical_air_to_vacuum_scalar(
                    float(fields[1])
                )
                lower_j = float(fields[2])
                upper_j = float(fields[3])
                lower_energy = float(fields[6])
                upper_energy = float(fields[8])
                log_fwhm_per_perturber = float(fields[14])
                temperature_exponent = float(fields[15])
            except ValueError as error:
                raise ValueError(
                    f"{source_path}:{line_number}: invalid Barklem record"
                ) from error
            element = number_to_element.get(atomic_number)
            if element is None:
                continue
            if (
                not np.isfinite(log_fwhm_per_perturber)
                or not np.isfinite(temperature_exponent)
            ):
                raise ValueError(
                    f"{source_path}:{line_number}: non-finite Barklem width"
                )
            records.setdefault((element, charge), []).append(
                (
                    wavelength_vacuum,
                    lower_j,
                    upper_j,
                    lower_energy,
                    upper_energy,
                    10.0**log_fwhm_per_perturber,
                    temperature_exponent,
                )
            )

    merged_ions = dict(base_database.ions)
    matched = 0
    for key, ion in base_database.ions.items():
        local_records = records.get(key)
        if not local_records:
            continue
        levels = {level.index: level for level in ion.levels}
        updated: list[AtomicTransition] = []
        for transition in ion.transitions:
            if transition.transition_type != "E1":
                updated.append(transition)
                continue
            lower = levels.get(transition.lower_index)
            upper = levels.get(transition.upper_index)
            if lower is None or upper is None:
                updated.append(transition)
                continue
            lower_j = 0.5 * (lower.statistical_weight - 1.0)
            upper_j = 0.5 * (upper.statistical_weight - 1.0)
            candidates = []
            for record in local_records:
                (
                    wavelength,
                    record_lower_j,
                    record_upper_j,
                    lower_energy,
                    upper_energy,
                    _,
                    _,
                ) = record
                wavelength_difference = abs(
                    wavelength - transition.wavelength_vacuum_angstrom
                )
                if wavelength_difference > maximum_wavelength_difference_angstrom:
                    continue
                if (
                    abs(record_lower_j - lower_j) > 0.05
                    or abs(record_upper_j - upper_j) > 0.05
                ):
                    continue
                energy_score = 0.0
                if lower_energy >= 0.0 and upper_energy >= 0.0:
                    lower_difference = abs(
                        lower_energy - lower.energy_wavenumber
                    )
                    upper_difference = abs(
                        upper_energy - upper.energy_wavenumber
                    )
                    if (
                        lower_difference > maximum_energy_difference_wavenumber
                        or upper_difference > maximum_energy_difference_wavenumber
                    ):
                        continue
                    energy_score = (
                        lower_difference + upper_difference
                    ) / maximum_energy_difference_wavenumber
                candidates.append((
                    wavelength_difference
                    / maximum_wavelength_difference_angstrom
                    + energy_score,
                    record,
                ))
            if not candidates:
                updated.append(transition)
                continue
            _, best = min(candidates, key=lambda item: item[0])
            updated.append(replace(
                transition,
                neutral_h_vdw_rate_coefficient_cm3_s=best[5],
                neutral_h_vdw_temperature_exponent=best[6],
            ))
            matched += 1
        merged_ions[key] = replace(ion, transitions=tuple(updated))
    return AtomicDatabase(
        MappingProxyType(merged_ions),
        base_database.source
        + f"; Barklem-Piskunov-O'Mara neutral-H widths ({matched} matches)",
    )


def read_kurucz_gf100_atomic_database(
    paths: Iterable[str | Path],
    base_database: AtomicDatabase,
    *,
    elements: Iterable[str] | None = None,
    replace_transitions: bool = True,
    supplement_missing_transitions: bool = False,
    replace_matched_atomic_data: bool = False,
    minimum_wavelength_angstrom: float | None = None,
    maximum_wavelength_angstrom: float | None = None,
) -> AtomicDatabase:
    """Merge Kurucz GF100 optical lines and damping constants into a database.

    The public GF100 files use the documented fixed-width 160-column format.
    Their wavelengths are air wavelengths above 200 nm, but the level-energy
    difference supplies a consistent vacuum wavelength without an additional
    refractive-index convention. Within each supplied file's wavelength
    interval, Stout E1 transitions for the requested ions are replaced rather
    than double-counted. Stout levels are retained and Kurucz levels are
    matched by energy and statistical weight or appended when absent. Setting
    replace_transitions=False instead keeps the Stout oscillator strengths and
    only attaches Kurucz damping constants to wavelength-matched lines. Set
    ``replace_matched_atomic_data=True`` to replace oscillator strengths and
    Einstein coefficients as well, but only for transitions whose lower and
    upper energies and statistical weights match; unmatched Stout lines are
    retained. With
    ``supplement_missing_transitions=True``, existing Stout lines are retained
    and only Kurucz transitions with an unmatched pair of energy levels are
    appended. This recovers real complexes absent from Stout without replacing
    the better-performing Stout oscillator strengths wholesale.  Optional
    wavelength bounds permit a provenance-controlled replacement of one
    complete spectral interval from a current per-ion Kurucz file.
    """

    if sum((
        bool(replace_transitions),
        bool(supplement_missing_transitions),
        bool(replace_matched_atomic_data),
    )) > 1:
        raise ValueError(
            "Kurucz replacement, supplementation, and matched-data modes are exclusive"
        )
    if (
        minimum_wavelength_angstrom is not None
        and (
            not np.isfinite(minimum_wavelength_angstrom)
            or minimum_wavelength_angstrom <= 0.0
        )
    ):
        raise ValueError("the minimum Kurucz wavelength must be positive")
    if (
        maximum_wavelength_angstrom is not None
        and (
            not np.isfinite(maximum_wavelength_angstrom)
            or maximum_wavelength_angstrom <= 0.0
        )
    ):
        raise ValueError("the maximum Kurucz wavelength must be positive")
    if (
        minimum_wavelength_angstrom is not None
        and maximum_wavelength_angstrom is not None
        and minimum_wavelength_angstrom >= maximum_wavelength_angstrom
    ):
        raise ValueError("Kurucz wavelength bounds must be increasing")

    requested = (
        set(base_database.elements)
        if elements is None
        else {_canonical_element(element) for element in elements}
    )
    atomic_number_to_symbol = {
        number: symbol for symbol, number in ATOMIC_NUMBER.items()
    }
    records: dict[tuple[str, int], list[tuple[object, ...]]] = {}
    covered_intervals: list[tuple[float, float]] = []
    sources: list[str] = []
    for path in paths:
        source = Path(path)
        local_wavelengths: list[float] = []
        with source.open(encoding="latin1") as stream:
            for line in stream:
                if len(line) < 101 or not line.strip():
                    continue
                try:
                    wavelength_nm = float(line[0:11])
                    log_gf = float(line[11:18])
                    element_code = float(line[18:24])
                    first_energy = abs(float(line[24:36]))
                    first_j = float(line[36:41])
                    first_label = line[41:51].strip()
                    # Kurucz per-ion files have two independently expanded
                    # layouts in addition to the legacy GF100 columns.  A
                    # six-digit second energy widens that field by one
                    # column.  Some alkali labels such as ``2p6.10d 2D`` also
                    # overflow the nominal ten-column first-label field and
                    # shift the second energy one column to the right.  Try
                    # the documented layouts explicitly; silently dropping
                    # the latter otherwise removes Na I n>=10 optical lines.
                    second_layouts = (
                        (51, 63, 63, 68, 69, 79, 79),  # legacy GF100
                        (51, 64, 64, 69, 70, 80, 80),  # six-digit energy
                        (52, 64, 64, 69, 70, 80, 80),  # long first label
                        (52, 65, 65, 70, 71, 81, 81),  # both expansions
                    )
                    parsed_second = None
                    for (
                        energy_start,
                        energy_stop,
                        j_start,
                        j_stop,
                        label_start,
                        label_stop,
                        damping_start,
                    ) in second_layouts:
                        try:
                            candidate = (
                                abs(float(line[energy_start:energy_stop])),
                                float(line[j_start:j_stop]),
                                line[label_start:label_stop].strip(),
                                float(line[damping_start:damping_start + 6]),
                                float(line[damping_start + 6:damping_start + 12]),
                                float(line[damping_start + 12:damping_start + 18]),
                            )
                        except ValueError:
                            continue
                        parsed_second = candidate
                        break
                    if parsed_second is None:
                        raise ValueError("unrecognized Kurucz second-level layout")
                    (
                        second_energy,
                        second_j,
                        second_label,
                        log_radiative,
                        log_stark,
                        log_vdw,
                    ) = parsed_second
                except ValueError:
                    continue
                atomic_number = int(np.floor(element_code + 1.0e-6))
                charge = int(round(100.0 * (element_code - atomic_number)))
                symbol = atomic_number_to_symbol.get(atomic_number)
                if (
                    symbol not in requested
                    or (symbol, charge) not in base_database.ions
                    or first_energy == second_energy
                ):
                    continue
                if first_energy < second_energy:
                    lower_energy, lower_j, lower_label = (
                        first_energy, first_j, first_label
                    )
                    upper_energy, upper_j, upper_label = (
                        second_energy, second_j, second_label
                    )
                else:
                    lower_energy, lower_j, lower_label = (
                        second_energy, second_j, second_label
                    )
                    upper_energy, upper_j, upper_label = (
                        first_energy, first_j, first_label
                    )
                lower_weight = 2.0 * lower_j + 1.0
                upper_weight = 2.0 * upper_j + 1.0
                oscillator_strength = 10.0**log_gf / lower_weight
                if not np.isfinite(oscillator_strength) or oscillator_strength <= 0.0:
                    continue
                vacuum_wavelength = 1.0e8 / (upper_energy - lower_energy)
                if (
                    minimum_wavelength_angstrom is not None
                    and vacuum_wavelength < minimum_wavelength_angstrom
                ):
                    continue
                if (
                    maximum_wavelength_angstrom is not None
                    and vacuum_wavelength > maximum_wavelength_angstrom
                ):
                    continue
                radiative = 10.0**log_radiative if log_radiative > 0.0 else None
                stark = 10.0**log_stark if log_stark < 0.0 else None
                vdw = 10.0**log_vdw if log_vdw < 0.0 else None
                records.setdefault((symbol, charge), []).append(
                    (
                        vacuum_wavelength,
                        oscillator_strength,
                        lower_energy,
                        lower_weight,
                        lower_label,
                        upper_energy,
                        upper_weight,
                        upper_label,
                        radiative,
                        stark,
                        vdw,
                    )
                )
                local_wavelengths.append(wavelength_nm * 10.0)
        if local_wavelengths:
            covered_intervals.append(
                (min(local_wavelengths) - 5.0, max(local_wavelengths) + 5.0)
            )
        sources.append(source.name)

    if not records:
        raise ValueError("no requested Kurucz GF100 transitions were loaded")

    merged_ions = dict(base_database.ions)
    for key, ion in base_database.ions.items():
        ion_records = records.get(key)
        if not ion_records:
            continue
        if not replace_transitions and not supplement_missing_transitions:
            ordered_records = sorted(ion_records, key=lambda record: record[0])
            record_wavelength = np.asarray(
                [record[0] for record in ordered_records], dtype=np.float64
            )
            matched_transitions = []
            level_by_index = {level.index: level for level in ion.levels}
            for transition in ion.transitions:
                if transition.transition_type != "E1":
                    matched_transitions.append(transition)
                    continue
                start = int(np.searchsorted(
                    record_wavelength,
                    transition.wavelength_vacuum_angstrom - 0.08,
                ))
                stop = int(np.searchsorted(
                    record_wavelength,
                    transition.wavelength_vacuum_angstrom + 0.08,
                    side="right",
                ))
                if stop <= start:
                    matched_transitions.append(transition)
                    continue
                candidates = ordered_records[start:stop]
                if replace_matched_atomic_data:
                    lower_level = level_by_index.get(transition.lower_index)
                    upper_level = level_by_index.get(transition.upper_index)
                    if lower_level is None or upper_level is None:
                        matched_transitions.append(transition)
                        continue
                    candidates = [
                        record
                        for record in candidates
                        if abs(record[2] - lower_level.energy_wavenumber) <= 0.15
                        and abs(record[3] - lower_level.statistical_weight) <= 1.0e-6
                        and abs(record[5] - upper_level.energy_wavenumber) <= 0.15
                        and abs(record[6] - upper_level.statistical_weight) <= 1.0e-6
                    ]
                    if not candidates:
                        matched_transitions.append(transition)
                        continue
                candidate = min(
                    candidates,
                    key=lambda record: (
                        abs(record[0] - transition.wavelength_vacuum_angstrom)
                        / 0.02
                        + abs(
                            np.log10(record[1])
                            - np.log10(
                                transition.absorption_oscillator_strength
                            )
                        )
                    ),
                )
                replacement_values = {
                    "radiative_damping_rate_s": candidate[8],
                    "electron_stark_rate_coefficient_cm3_s": candidate[9],
                    "neutral_h_vdw_rate_coefficient_cm3_s": candidate[10],
                }
                if replace_matched_atomic_data:
                    wavelength_cm = transition.wavelength_vacuum_angstrom * 1.0e-8
                    replacement_values.update({
                        "absorption_oscillator_strength": candidate[1],
                        "einstein_a": (
                            8.0 * PI**2 * ELEMENTARY_CHARGE_ESU**2
                            / (ELECTRON_MASS * LIGHT_SPEED * wavelength_cm**2)
                            * candidate[3] / candidate[6] * candidate[1]
                        ),
                    })
                matched_transitions.append(replace(transition, **replacement_values))
            merged_ions[key] = replace(
                ion,
                transitions=tuple(matched_transitions),
                source=ion.source + (
                    "; Kurucz GF100 matched atomic data"
                    if replace_matched_atomic_data
                    else "; Kurucz GF100 matched damping constants"
                ),
            )
            continue
        levels = list(ion.levels)
        next_index = max(level.index for level in levels) + 1
        # Large Kurucz predicted-line slices can contain millions of records.
        # A linear scan through every accumulated level for every transition
        # makes ingestion quadratic.  Index levels in the same 0.1-cm^-1
        # energy / exact-weight tolerance used by the historical matcher.
        level_bins: dict[tuple[int, int], list[AtomicLevel]] = {}

        def add_level_to_bins(level: AtomicLevel) -> None:
            key = (
                int(round(level.energy_wavenumber * 10.0)),
                int(round(level.statistical_weight * 1.0e6)),
            )
            level_bins.setdefault(key, []).append(level)

        for existing_level in levels:
            add_level_to_bins(existing_level)

        def level_index(energy: float, weight: float, label: str) -> int:
            nonlocal next_index
            energy_bin = int(round(energy * 10.0))
            weight_bin = int(round(weight * 1.0e6))
            for nearby_energy_bin in (
                energy_bin - 1,
                energy_bin,
                energy_bin + 1,
            ):
                for level in level_bins.get(
                    (nearby_energy_bin, weight_bin), ()
                ):
                    if (
                        abs(level.energy_wavenumber - energy) <= 0.1
                        and abs(level.statistical_weight - weight) <= 1.0e-6
                    ):
                        return level.index
            index = next_index
            next_index += 1
            new_level = AtomicLevel(
                index, float(energy), float(weight), f"Kurucz {label}"
            )
            levels.append(new_level)
            add_level_to_bins(new_level)
            return index

        if supplement_missing_transitions:
            transitions = list(ion.transitions)
            existing_transition_wavelengths: dict[
                tuple[int, int], list[float]
            ] = {}
            for transition in transitions:
                if transition.transition_type == "E1":
                    existing_transition_wavelengths.setdefault(
                        (transition.lower_index, transition.upper_index), []
                    ).append(transition.wavelength_vacuum_angstrom)
        else:
            transitions = [
                transition
                for transition in ion.transitions
                if transition.transition_type != "E1"
                or not any(
                    lower <= transition.wavelength_vacuum_angstrom <= upper
                    for lower, upper in covered_intervals
                )
            ]
        seen: set[tuple[int, int, int]] = set()
        for (
            wavelength,
            oscillator_strength,
            lower_energy,
            lower_weight,
            lower_label,
            upper_energy,
            upper_weight,
            upper_label,
            radiative,
            stark,
            vdw,
        ) in ion_records:
            lower_index = level_index(lower_energy, lower_weight, lower_label)
            upper_index = level_index(upper_energy, upper_weight, upper_label)
            if supplement_missing_transitions:
                pair = (lower_index, upper_index)
                if any(
                    abs(existing_wavelength - wavelength) <= 0.08
                    for existing_wavelength in existing_transition_wavelengths.get(
                        pair, ()
                    )
                ):
                    continue
            identifier = (
                lower_index,
                upper_index,
                int(round(wavelength * 1.0e4)),
            )
            if identifier in seen:
                continue
            seen.add(identifier)
            wavelength_cm = wavelength * 1.0e-8
            einstein_a = (
                8.0 * PI**2 * ELEMENTARY_CHARGE_ESU**2
                / (ELECTRON_MASS * LIGHT_SPEED * wavelength_cm**2)
                * lower_weight / upper_weight
                * oscillator_strength
            )
            transitions.append(
                AtomicTransition(
                    lower_index,
                    upper_index,
                    float(einstein_a),
                    "E1",
                    float(wavelength),
                    float(oscillator_strength),
                    radiative_damping_rate_s=radiative,
                    electron_stark_rate_coefficient_cm3_s=stark,
                    neutral_h_vdw_rate_coefficient_cm3_s=vdw,
                )
            )
            if supplement_missing_transitions:
                existing_transition_wavelengths.setdefault(
                    (lower_index, upper_index), []
                ).append(float(wavelength))
        merged_ions[key] = replace(
            ion,
            levels=tuple(levels),
            transitions=tuple(transitions),
            source=(
                ion.source + "; Kurucz GF100 missing-transition supplement"
                if supplement_missing_transitions
                else ion.source + "; Kurucz GF100 optical replacement"
            ),
        )
    operation = (
        "missing-transition supplement: "
        if supplement_missing_transitions
        else "optical replacement: "
        if replace_transitions
        else "matched atomic data: "
        if replace_matched_atomic_data
        else "matched damping constants: "
    )
    return AtomicDatabase(
        MappingProxyType(merged_ions),
        base_database.source + "; Kurucz GF100 " + operation + ", ".join(sources),
    )


_NIST_ASD_ACCURACY_PERCENT = MappingProxyType(
    {
        "AAA": 0.3,
        "AA": 1.0,
        "A+": 2.0,
        "A": 3.0,
        "B+": 7.0,
        "B": 10.0,
        "C+": 18.0,
        "C": 25.0,
        "D+": 40.0,
        "D": 50.0,
        "E": np.inf,
    }
)


def _nist_asd_j_value(value: str) -> float:
    """Parse one ASD integer or fractional J value."""

    cleaned = value.strip().strip('"').strip("=")
    if "/" in cleaned:
        numerator, denominator = cleaned.split("/", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(cleaned)


def read_nist_asd_strong_atomic_database(
    paths: Iterable[str | Path],
    base_database: AtomicDatabase,
    *,
    maximum_accuracy_percent: float = 25.0,
    minimum_oscillator_strength: float = 0.01,
    energy_tolerance_wavenumber: float = 0.15,
) -> AtomicDatabase:
    """Replace exact-level strong-line data with evaluated NIST ASD values.

    The input files are tab-delimited NIST ASD line queries such as the
    checksum-pinned files in :data:`NIST_ASD_STRONG_ION_FILES`.  A record is
    accepted only when both level energies *and* statistical weights match a
    transition already present in ``base_database``.  Thus this function
    cannot create duplicate blends or attach a multiplet oscillator strength
    to the wrong fine-structure component.  The default accepts accuracy
    grades C or better and ``f >= 0.01``; weaker or less certain lines retain
    their original Stout values.
    """

    if (
        not np.isfinite(maximum_accuracy_percent)
        or maximum_accuracy_percent <= 0.0
    ):
        raise ValueError("maximum_accuracy_percent must be finite and positive")
    if (
        not np.isfinite(minimum_oscillator_strength)
        or minimum_oscillator_strength <= 0.0
    ):
        raise ValueError("minimum_oscillator_strength must be finite and positive")
    if (
        not np.isfinite(energy_tolerance_wavenumber)
        or energy_tolerance_wavenumber <= 0.0
    ):
        raise ValueError("energy_tolerance_wavenumber must be finite and positive")

    records: dict[
        tuple[str, int],
        list[tuple[float, float, float, float, float, float, str]],
    ] = {}
    sources: list[str] = []
    file_definition = {
        filename: (element, charge)
        for filename, (_, _, element, charge) in NIST_ASD_STRONG_ION_FILES.items()
    }
    for path in paths:
        source = Path(path)
        ion_key = file_definition.get(source.name)
        if ion_key is None:
            raise ValueError(
                f"unrecognized NIST ASD strong-line filename {source.name!r}"
            )
        if ion_key not in base_database.ions:
            continue
        with source.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                try:
                    accuracy = row["Acc"].strip().strip('"').strip("=")
                    uncertainty = _NIST_ASD_ACCURACY_PERCENT[accuracy]
                    oscillator_strength = float(
                        row["fik"].strip().strip('"').strip("=")
                    )
                    einstein_a = float(
                        row["Aki(s^-1)"].strip().strip('"').strip("=")
                    )
                    # Brackets mark NIST level energies derived from theory
                    # or interpolation; they still identify the Stout level.
                    lower_energy = float(
                        row["Ei(cm-1)"].strip().strip('"').strip("=").strip("[]")
                    )
                    upper_energy = float(
                        row["Ek(cm-1)"].strip().strip('"').strip("=").strip("[]")
                    )
                    lower_weight = 2.0 * _nist_asd_j_value(row["J_i"]) + 1.0
                    upper_weight = 2.0 * _nist_asd_j_value(row["J_k"]) + 1.0
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    uncertainty > maximum_accuracy_percent
                    or oscillator_strength < minimum_oscillator_strength
                    or oscillator_strength <= 0.0
                    or einstein_a <= 0.0
                    or upper_energy <= lower_energy
                ):
                    continue
                records.setdefault(ion_key, []).append(
                    (
                        lower_energy,
                        lower_weight,
                        upper_energy,
                        upper_weight,
                        oscillator_strength,
                        einstein_a,
                        accuracy,
                    )
                )
        sources.append(source.name)

    if not records:
        raise ValueError("no usable NIST ASD strong-line records were loaded")

    merged_ions = dict(base_database.ions)
    matched = 0
    matched_by_grade: dict[str, int] = {}
    for ion_key, ion in base_database.ions.items():
        candidates = records.get(ion_key)
        if not candidates:
            continue
        levels = {level.index: level for level in ion.levels}
        transitions: list[AtomicTransition] = []
        for transition in ion.transitions:
            if transition.transition_type != "E1":
                transitions.append(transition)
                continue
            lower = levels.get(transition.lower_index)
            upper = levels.get(transition.upper_index)
            if lower is None or upper is None:
                transitions.append(transition)
                continue
            exact = [
                record
                for record in candidates
                if (
                    abs(record[0] - lower.energy_wavenumber)
                    <= energy_tolerance_wavenumber
                    and abs(record[1] - lower.statistical_weight) <= 1.0e-8
                    and abs(record[2] - upper.energy_wavenumber)
                    <= energy_tolerance_wavenumber
                    and abs(record[3] - upper.statistical_weight) <= 1.0e-8
                )
            ]
            if not exact:
                transitions.append(transition)
                continue
            record = min(
                exact,
                key=lambda candidate: (
                    abs(candidate[0] - lower.energy_wavenumber)
                    + abs(candidate[2] - upper.energy_wavenumber)
                    + abs(
                        np.log(candidate[4])
                        - np.log(transition.absorption_oscillator_strength)
                    )
                ),
            )
            transitions.append(replace(
                transition,
                absorption_oscillator_strength=float(record[4]),
                einstein_a=float(record[5]),
            ))
            matched += 1
            matched_by_grade[record[6]] = matched_by_grade.get(record[6], 0) + 1
        merged_ions[ion_key] = replace(
            ion,
            transitions=tuple(transitions),
            source=ion.source + "; evaluated NIST ASD 5.12 strong-line data",
        )
    if matched == 0:
        raise ValueError("no NIST ASD records matched the base atomic levels")
    grade_summary = ",".join(
        f"{grade}:{count}" for grade, count in sorted(matched_by_grade.items())
    )
    return AtomicDatabase(
        MappingProxyType(merged_ions),
        base_database.source
        + "; NIST ASD 5.12 exact-level strong-line replacement "
        + f"({matched} matches; grades {grade_summary}; files {', '.join(sources)})",
    )


def read_pg1159_atomic_database(
    root: str | Path,
    *,
    elements: Iterable[str] = ("C", "O"),
) -> AtomicDatabase:
    """Read complete ion ladders for hot PG1159 atmospheres.

    Carbon and oxygen remain the default bulk mixture.  Validation against a
    particular TMAP/observed spectrum may request trace species such as N or
    Ne without needing a separate atomic-database assembly path.
    """

    selected = tuple(_canonical_element(element) for element in elements)

    return read_stout_atomic_database(
        root,
        elements=selected,
        maximum_charge={element: ATOMIC_NUMBER[element] for element in selected},
    )


def read_verner_phfit2_database(
    path: str | Path,
    *,
    elements: Iterable[str],
    maximum_charge: int | None = None,
    exclude: Iterable[tuple[str, int]] = (),
) -> dict[tuple[str, int], VernerPhfit2Fit]:
    """Parse the fit tables of Verner's public ``phfit2.f``.

    Returns a total ground-state fit for each requested ion, skipping
    ``exclude`` (ions already covered by ``photo.dat``).  This supplies the
    iron-group ions (Sc--Zn except Fe), P, Cl and K that ``photo.dat`` omits.
    """

    text = Path(path).read_text(encoding="ascii", errors="replace")
    lines = text.splitlines()
    joined = []
    for line in lines:
        if line.lstrip().startswith(("*", "c", "C")) and not line.lstrip().upper().startswith("COMMON"):
            if len(line) > 5 and line[5:6] not in (" ", "") and line[:5].strip() == "":
                pass
            elif line[:1] in ("*", "c", "C"):
                continue
        if len(line) > 5 and line[:5].strip() == "" and line[5] not in (" ", "0") and joined:
            joined[-1] += line[6:]
        else:
            joined.append(line)
    numbers = r"([-+0-9.EeDd,\s]+)"
    ph1: dict[tuple[int, int, int], tuple[float, ...]] = {}
    ph2: dict[tuple[int, int], tuple[float, ...]] = {}
    ninn: list[int] = []
    ntot: list[int] = []
    for line in joined:
        compact = line.strip()
        m = re.match(r"DATA\s*\(PH1\(I,\s*(\d+),\s*(\d+),\s*(\d+)\),I=1,6\)\s*/(.*)/", compact, re.I)
        if m:
            values = tuple(float(v.replace("D", "E")) for v in m.group(4).replace(" ", "").split(","))
            ph1[(int(m.group(1)), int(m.group(2)), int(m.group(3)))] = values
            continue
        m = re.match(r"DATA\s*\(PH2\(I,\s*(\d+),\s*(\d+)\),I=1,7\)\s*/(.*)/", compact, re.I)
        if m:
            values = tuple(float(v.replace("D", "E")) for v in m.group(3).replace(" ", "").split(","))
            ph2[(int(m.group(1)), int(m.group(2)))] = values
            continue
        m = re.match(r"DATA\s*\(NINN\(I\),I=1,30\)\s*/(.*)/", compact, re.I)
        if m:
            ninn = [int(v) for v in m.group(1).replace(" ", "").split(",")]
            continue
        m = re.match(r"DATA\s*\(NTOT\(I\),I=1,30\)\s*/(.*)/", compact, re.I)
        if m:
            ntot = [int(v) for v in m.group(1).replace(" ", "").split(",")]
    if len(ninn) != 30 or len(ntot) != 30 or not ph1:
        raise ValueError(f"{path} does not look like Verner's phfit2.f")
    excluded = {(_canonical_element(e), int(c)) for e, c in exclude}
    number_to_element = {z: e for e, z in ATOMIC_NUMBER.items()}
    result: dict[tuple[str, int], VernerPhfit2Fit] = {}
    for element in elements:
        symbol = _canonical_element(element)
        nz = ATOMIC_NUMBER[symbol]
        if nz > 30:
            continue
        for ne in range(nz, 0, -1):
            charge = nz - ne
            if maximum_charge is not None and charge > maximum_charge:
                break
            if (symbol, charge) in excluded:
                continue
            nout = ntot[ne - 1]
            if nz == ne and nz > 18:
                nout = 7
            if nz == ne + 1 and nz in (20, 21, 22, 25, 26):
                nout = 7
            shells = tuple(ph1.get((nz, ne, shell)) for shell in range(1, 8))
            if all(params is None or params[0] <= 0.0 for params in shells[:nout]):
                continue
            shells = tuple(
                None if params is None or params[0] <= 0.0 else params
                for params in shells
            )
            result[(symbol, charge)] = VernerPhfit2Fit(
                symbol, charge, nz, ne, shells, ph2.get((nz, ne)), nout, ninn[ne - 1]
            )
    return result


def read_verner_photoionization_database(
    path: str | Path,
    *,
    elements: Iterable[str] | None = None,
    maximum_charge: int | None = None,
    require_all_elements: bool = False,
) -> VernerPhotoionizationDatabase:
    """Read the public ASCII fit table distributed by Verner et al. (1996).

    The distributed table does not cover every element supported by the
    Stout line reader (notably Ti, Cr, and Ni).  Missing elements therefore
    contribute no bound-free opacity by default; set ``require_all_elements``
    when a validation workflow should fail rather than accept that omission.
    """

    if elements is None:
        selected_elements = set(ATOMIC_NUMBER)
    else:
        selected_elements = {element.strip().capitalize() for element in elements}
        unsupported = selected_elements - set(ATOMIC_NUMBER)
        if unsupported:
            raise ValueError(
                "unsupported photoionization elements: "
                + ", ".join(sorted(unsupported))
            )
    number_to_element = {
        atomic_number: element for element, atomic_number in ATOMIC_NUMBER.items()
        if element in selected_elements
    }
    fits: dict[tuple[str, int], VernerPhotoionizationFit] = {}
    with Path(path).open("r", encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) != 11:
                raise ValueError(
                    f"{path}:{line_number}: expected 11 Verner fit columns"
                )
            atomic_number, electron_count = map(int, fields[:2])
            element = number_to_element.get(atomic_number)
            if element is None:
                continue
            charge = atomic_number - electron_count
            if charge < 0 or (
                maximum_charge is not None and charge > maximum_charge
            ):
                continue
            values = tuple(float(value) for value in fields[2:])
            fit = VernerPhotoionizationFit(element, charge, *values)
            fits[(element, charge)] = fit
    missing = selected_elements - {element for element, _ in fits}
    if missing and require_all_elements:
        raise ValueError(
            "Verner table is missing requested elements: " + ", ".join(sorted(missing))
        )
    return VernerPhotoionizationDatabase(MappingProxyType(fits))


def read_mg_he_red_wing_table(
    path: str | Path,
    density_profile_path: str | Path | None = None,
) -> MgHeRedWingTable:
    """Read the CDS Mg I--He wing and an optional Fig. 6 density sequence."""

    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Mg-He table must contain temperature, wavelength, cross section")
    wavelength_by_temperature: dict[float, FloatArray] = {}
    cross_section_by_temperature: dict[float, FloatArray] = {}
    for temperature in np.unique(data[:, 0]):
        rows = data[data[:, 0] == temperature]
        order = np.argsort(rows[:, 1])
        wavelength_by_temperature[float(temperature)] = np.asarray(rows[order, 1])
        cross_section_by_temperature[float(temperature)] = np.asarray(rows[order, 2])
    if len(wavelength_by_temperature) < 2:
        raise ValueError("Mg-He table must contain at least two temperatures")
    wavelength_by_density = None
    cross_section_by_density = None
    if density_profile_path is not None:
        density_data = np.loadtxt(density_profile_path, comments="#")
        if (
            density_data.ndim != 2 or density_data.shape[1] != 3
            or density_data.shape[0] < 100
        ):
            raise ValueError(
                "Mg I--He density profile must have density, wavelength, "
                "cross-section columns"
            )
        density_wavelengths: dict[float, FloatArray] = {}
        density_cross_sections: dict[float, FloatArray] = {}
        for density in np.unique(density_data[:, 0]):
            rows = density_data[density_data[:, 0] == density]
            order = np.argsort(rows[:, 1])
            local_wavelength = np.asarray(rows[order, 1])
            local_cross_section = np.asarray(rows[order, 2])
            if (
                density <= 0.0 or np.any(local_wavelength <= 0.0)
                or np.any(np.diff(local_wavelength) <= 0.0)
                or np.any(~np.isfinite(local_cross_section))
                or np.any(local_cross_section < 0.0)
            ):
                raise ValueError(
                    "Mg I--He density profile values must be finite and physical"
                )
            density_wavelengths[float(density)] = local_wavelength
            density_cross_sections[float(density)] = local_cross_section
        if len(density_wavelengths) < 2:
            raise ValueError("Mg I--He density profile needs at least two densities")
        wavelength_by_density = MappingProxyType(density_wavelengths)
        cross_section_by_density = MappingProxyType(density_cross_sections)
    return MgHeRedWingTable(
        MappingProxyType(wavelength_by_temperature),
        MappingProxyType(cross_section_by_temperature),
        wavelength_by_density=wavelength_by_density,
        cross_section_by_density=cross_section_by_density,
    )


def read_mg_ii_he_profile_table(path: str | Path) -> MgIIHeProfileTable:
    """Read a two-column Mg II--He wavelength/cross-section table.

    ``T=... K`` and ``n_He=... cm^-3`` tokens in comment headers override
    the original 8000-K, 2e21-cm^-3 defaults.  Figure-panel tables whose
    curves remain nonzero at their plotting limits opt into continuous
    endpoint-log-slope bridge tails rather than acquiring artificial opacity
    edges at those limits.
    """

    table_path = Path(path)
    header = "\n".join(
        line[1:].strip()
        for line in table_path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#")
    )
    temperature_match = re.search(r"T\s*=\s*([0-9.eE+-]+)\s*K", header)
    density_match = re.search(
        r"n_(?:He|\\?mathrm\{He\})\s*=\s*([0-9.eE+-]+)\s*cm\^?-?3",
        header,
    )
    data = np.loadtxt(table_path, comments="#")
    if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] < 20:
        raise ValueError("Mg II--He profile must have two columns and at least 20 rows")
    wavelength = np.asarray(data[:, 0], dtype=np.float64)
    cross_section = np.asarray(data[:, 1], dtype=np.float64)
    if (
        np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
        or np.any(~np.isfinite(cross_section)) or np.any(cross_section < 0.0)
    ):
        raise ValueError("Mg II--He profile values must be finite and physical")
    temperature = (
        float(temperature_match.group(1)) if temperature_match else 8000.0
    )
    density = float(density_match.group(1)) if density_match else 2.0e21
    from_truncated_figure_panel = (
        "Blouin, Dufour & Allard 2018" in header and "Fig. 1" in header
    )
    source = next(
        (line.strip() for line in header.splitlines() if line.strip()),
        "user-supplied Mg II--He profile",
    )
    return MgIIHeProfileTable(
        wavelength,
        cross_section,
        temperature_kelvin=temperature,
        helium_density_cm3=density,
        source=source,
        continue_fitted_log_linear_tails=from_truncated_figure_panel,
    )


def read_ca_ii_he_profile_table(path: str | Path) -> CaIIHeProfileTable:
    """Read the two-column summed Ca II H&K--He figure profile."""

    table_path = Path(path)
    header = "\n".join(
        line[1:].strip()
        for line in table_path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#")
    )
    temperature_match = re.search(r"T\s*=\s*([0-9.eE+-]+)\s*K", header)
    density_match = re.search(
        r"n_(?:He|\\?mathrm\{He\})\s*=\s*([0-9.eE+-]+)\s*cm\^?-?3",
        header,
    )
    data = np.loadtxt(table_path, comments="#")
    if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] < 20:
        raise ValueError("Ca II--He profile must have two columns and at least 20 rows")
    wavelength = np.asarray(data[:, 0], dtype=np.float64)
    cross_section = np.asarray(data[:, 1], dtype=np.float64)
    if (
        np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
        or np.any(~np.isfinite(cross_section)) or np.any(cross_section < 0.0)
    ):
        raise ValueError("Ca II--He profile values must be finite and physical")
    temperature = float(temperature_match.group(1)) if temperature_match else 6000.0
    density = float(density_match.group(1)) if density_match else 1.0e22
    source = next(
        (line.strip() for line in header.splitlines() if line.strip()),
        "user-supplied Ca II--He profile",
    )
    return CaIIHeProfileTable(
        wavelength,
        cross_section,
        temperature_kelvin=temperature,
        helium_density_cm3=density,
        source=source,
    )


def read_ca_ii_he_profile_temperature_grid(
    paths: Iterable[str | Path],
) -> CaIIHeProfileTable:
    """Read multiple same-density Ca II--He curves as a temperature grid.

    The available curves were recovered independently from the vector paths
    of Allard & Alekseev's published figure.  They supply a real temperature
    dimension but only one helium density.  The returned object therefore
    interpolates in log temperature and retains the single-table rule of
    linear density scaling below the published density with no extrapolation
    above it.
    """

    tables = tuple(read_ca_ii_he_profile_table(path) for path in paths)
    if len(tables) < 2:
        raise ValueError("Ca II--He temperature grid needs at least two tables")
    densities = np.asarray(
        [table.helium_density_cm3 for table in tables], dtype=np.float64
    )
    if not np.allclose(densities, densities[0], rtol=1.0e-10, atol=0.0):
        raise ValueError("Ca II--He temperature tables must share one density")
    temperatures = np.asarray(
        [table.temperature_kelvin for table in tables], dtype=np.float64
    )
    if np.unique(temperatures).size != temperatures.size:
        raise ValueError("Ca II--He temperature tables must be unique")
    wavelength_by_temperature = {
        float(table.temperature_kelvin): table.wavelength_angstrom
        for table in tables
    }
    cross_section_by_temperature = {
        float(table.temperature_kelvin): table.cross_section_cm2
        for table in tables
    }
    union_wavelength = np.unique(np.concatenate(
        tuple(wavelength_by_temperature.values())
    ))
    reference = min(
        tables, key=lambda table: abs(table.temperature_kelvin - 6000.0)
    )
    reference_cross_section = np.interp(
        union_wavelength,
        reference.wavelength_angstrom,
        reference.cross_section_cm2,
        left=0.0,
        right=0.0,
    )
    return CaIIHeProfileTable(
        union_wavelength,
        reference_cross_section,
        temperature_kelvin=float(reference.temperature_kelvin),
        helium_density_cm3=float(densities[0]),
        source=(
            "temperature grid assembled from "
            + "; ".join(table.source for table in tables)
        ),
        wavelength_by_temperature=MappingProxyType(
            wavelength_by_temperature
        ),
        cross_section_by_temperature=MappingProxyType(
            cross_section_by_temperature
        ),
    )


def read_ca_ii_he_profile_grid(
    paths: Iterable[str | Path],
) -> CaIIHeProfileTable:
    """Read a rectangular temperature-density Ca II--He profile grid.

    Each two-column file must declare ``T=... K`` and ``n_He=... cm^-3`` in
    its comment header. Cross sections are interpolated logarithmically in
    temperature and density wherever all four bracketing values are positive.
    The density dependence is linear below the least-dense calculation and is
    clamped above the largest published density.
    """

    tables = tuple(read_ca_ii_he_profile_table(path) for path in paths)
    if len(tables) < 4:
        raise ValueError(
            "Ca II--He temperature-density grid needs at least four tables"
        )
    temperatures = sorted({float(table.temperature_kelvin) for table in tables})
    densities = sorted({float(table.helium_density_cm3) for table in tables})
    if len(temperatures) < 2 or len(densities) < 2:
        raise ValueError(
            "Ca II--He profile grid needs multiple temperatures and densities"
        )
    by_condition = {
        (float(table.temperature_kelvin), float(table.helium_density_cm3)): table
        for table in tables
    }
    expected = {
        (temperature, density)
        for temperature in temperatures
        for density in densities
    }
    if len(by_condition) != len(tables) or set(by_condition) != expected:
        raise ValueError(
            "Ca II--He profile tables must form a unique rectangular grid"
        )
    wavelength_by_condition = {
        condition: table.wavelength_angstrom
        for condition, table in by_condition.items()
    }
    cross_section_by_condition = {
        condition: table.cross_section_cm2
        for condition, table in by_condition.items()
    }
    reference_condition = min(
        expected,
        key=lambda condition: (
            abs(np.log(condition[0] / 6000.0))
            + abs(np.log(condition[1] / 1.0e22))
        ),
    )
    reference = by_condition[reference_condition]
    return CaIIHeProfileTable(
        reference.wavelength_angstrom,
        reference.cross_section_cm2,
        temperature_kelvin=reference.temperature_kelvin,
        helium_density_cm3=reference.helium_density_cm3,
        source=(
            "temperature-density grid assembled from "
            + "; ".join(table.source for table in tables)
        ),
        wavelength_by_condition=MappingProxyType(wavelength_by_condition),
        cross_section_by_condition=MappingProxyType(
            cross_section_by_condition
        ),
    )


def read_ca_i_he_profile_table(
    path: str | Path,
    temperature_profile_path: str | Path | None = None,
) -> CaIHeProfileTable:
    """Read the Ca I density sequence and optional temperature sequence."""

    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] != 3 or data.shape[0] < 100:
        raise ValueError("Ca I--He profile must have three columns and at least 100 rows")
    wavelength_by_density: dict[float, FloatArray] = {}
    cross_section_by_density: dict[float, FloatArray] = {}
    for density in np.unique(data[:, 0]):
        rows = data[data[:, 0] == density]
        order = np.argsort(rows[:, 1])
        wavelength = np.asarray(rows[order, 1], dtype=np.float64)
        cross_section = np.asarray(rows[order, 2], dtype=np.float64)
        if (
            density <= 0.0 or np.any(wavelength <= 0.0)
            or np.any(np.diff(wavelength) <= 0.0)
            or np.any(~np.isfinite(cross_section)) or np.any(cross_section < 0.0)
        ):
            raise ValueError("Ca I--He profile values must be finite and physical")
        wavelength_by_density[float(density)] = wavelength
        cross_section_by_density[float(density)] = cross_section
    if len(wavelength_by_density) < 2:
        raise ValueError("Ca I--He profile needs at least two helium densities")
    wavelength_by_temperature = None
    cross_section_by_temperature = None
    if temperature_profile_path is not None:
        temperature_data = np.loadtxt(temperature_profile_path, comments="#")
        if (
            temperature_data.ndim != 2 or temperature_data.shape[1] != 3
            or temperature_data.shape[0] < 100
        ):
            raise ValueError(
                "Ca I--He temperature profile must have temperature, "
                "wavelength, cross-section columns"
            )
        temperature_wavelengths: dict[float, FloatArray] = {}
        temperature_cross_sections: dict[float, FloatArray] = {}
        for temperature in np.unique(temperature_data[:, 0]):
            rows = temperature_data[temperature_data[:, 0] == temperature]
            order = np.argsort(rows[:, 1])
            local_wavelength = np.asarray(rows[order, 1], dtype=np.float64)
            local_cross_section = np.asarray(rows[order, 2], dtype=np.float64)
            if (
                temperature <= 0.0 or np.any(local_wavelength <= 0.0)
                or np.any(np.diff(local_wavelength) <= 0.0)
                or np.any(~np.isfinite(local_cross_section))
                or np.any(local_cross_section < 0.0)
            ):
                raise ValueError(
                    "Ca I--He temperature profile values must be finite and physical"
                )
            temperature_wavelengths[float(temperature)] = local_wavelength
            temperature_cross_sections[float(temperature)] = local_cross_section
        if len(temperature_wavelengths) < 2:
            raise ValueError("Ca I--He temperature profile needs two temperatures")
        wavelength_by_temperature = MappingProxyType(temperature_wavelengths)
        cross_section_by_temperature = MappingProxyType(
            temperature_cross_sections
        )
    return CaIHeProfileTable(
        MappingProxyType(wavelength_by_density),
        MappingProxyType(cross_section_by_density),
        wavelength_by_temperature=wavelength_by_temperature,
        cross_section_by_temperature=cross_section_by_temperature,
    )


def _ion_fractions(log_ratios: FloatArray) -> FloatArray:
    cumulative = np.concatenate(
        (np.zeros((1, log_ratios.shape[1])), np.cumsum(log_ratios, axis=0)),
        axis=0,
    )
    maximum = np.max(cumulative, axis=0, keepdims=True)
    weights = np.exp(np.clip(cumulative - maximum, -745.0, 0.0))
    return weights / np.sum(weights, axis=0, keepdims=True)


def _hydrogen_state_with_trace_metal_electrons(
    template: HydrogenLTEState,
    temperature: FloatArray,
    electron_density: FloatArray,
) -> tuple[HydrogenLTEState, FloatArray]:
    """Re-solve H chemistry at fixed nuclei density and total electron density.

    The pressure structure and total hydrogen-nuclei density come from the
    host HM atmosphere.  Charge donated by trace metals changes the Saha and
    molecular equilibria, so the H, H+, H2, H2+, H-, and H3+ populations are
    recomputed rather than simply adding metal electrons to a frozen pure-H
    state.  As in the helium-host trace-metal closure, the small metal
    contribution to the gas-pressure equation is deliberately omitted.
    """

    nuclei_density = np.asarray(template.hydrogen_nuclei_density)
    temperature, electron_density, nuclei_density = np.broadcast_arrays(
        np.asarray(temperature, dtype=np.float64),
        np.asarray(electron_density, dtype=np.float64),
        nuclei_density,
    )
    tiny = np.finfo(np.float64).tiny
    safe_electron = np.maximum(electron_density, tiny)
    include_molecules = template.molecular_hydrogen_density is not None
    include_negative = template.negative_hydrogen_density is not None
    h3_model = template.trihydrogen_ion_partition_model

    def species_for_partition(partition: FloatArray) -> tuple[FloatArray, ...]:
        saha = hydrogen_saha_constant(temperature) / np.maximum(partition, tiny)
        proton_ratio = saha / safe_electron
        if include_molecules:
            h2_inverse = 1.0 / np.maximum(
                molecular_hydrogen_dissociation_constant(
                    temperature,
                    atomic_internal_partition_function=partition,
                ),
                tiny,
            )
            h2plus_inverse = 1.0 / np.maximum(
                molecular_hydrogen_ion_dissociation_constant(
                    temperature,
                    atomic_internal_partition_function=partition,
                ),
                tiny,
            )
        else:
            h2_inverse = np.zeros_like(temperature)
            h2plus_inverse = np.zeros_like(temperature)
        hminus_ratio = (
            safe_electron
            / np.maximum(
                negative_hydrogen_ionization_constant(
                    temperature,
                    atomic_internal_partition_function=partition,
                ),
                tiny,
            )
            if include_negative
            else np.zeros_like(temperature)
        )
        h3_inverse = (
            1.0
            / np.maximum(
                trihydrogen_ion_dissociation_constant(
                    temperature, partition_model=h3_model
                ),
                tiny,
            )
            if h3_model is not None
            else np.zeros_like(temperature)
        )

        # At fixed n_e every hydrogen species is a first-, second-, or
        # third-order function of n(H).  The nuclei-conservation equation is
        # therefore a monotonic cubic, solved by bracketed Newton iteration.
        linear = 1.0 + proton_ratio + hminus_ratio
        quadratic = 2.0 * h2_inverse + 2.0 * proton_ratio * h2plus_inverse
        cubic = 3.0 * h2_inverse * proton_ratio * h3_inverse
        lower = np.zeros_like(nuclei_density)
        upper = nuclei_density.copy()
        neutral = np.minimum(nuclei_density / np.maximum(linear, tiny), upper)
        conservation_tolerance = (
            32.0
            * np.finfo(np.float64).eps
            * np.maximum(nuclei_density, 1.0)
        )
        # A Newton proposal can become unusable in molecular transition
        # layers, reducing this to bisection.  Retain enough iterations for
        # that worst case rather than relying on a platform-sensitive Newton
        # path reaching the root within a short fixed iteration count.
        for _ in range(64):
            residual = (
                linear * neutral
                + quadratic * neutral**2
                + cubic * neutral**3
                - nuclei_density
            )
            if np.all(np.abs(residual) <= conservation_tolerance):
                break
            upper = np.where(residual > 0.0, neutral, upper)
            lower = np.where(residual > 0.0, lower, neutral)
            derivative = (
                linear + 2.0 * quadratic * neutral + 3.0 * cubic * neutral**2
            )
            candidate = neutral - residual / np.maximum(derivative, tiny)
            midpoint = 0.5 * (lower + upper)
            usable = (
                np.isfinite(candidate)
                & (candidate > lower)
                & (candidate < upper)
            )
            neutral = np.where(usable, candidate, midpoint)
        proton = proton_ratio * neutral
        molecule = h2_inverse * neutral**2
        molecular_ion = h2plus_inverse * neutral * proton
        negative = hminus_ratio * neutral
        trihydrogen_ion = h3_inverse * molecule * proton
        return (
            neutral,
            proton,
            molecule,
            molecular_ion,
            negative,
            trihydrogen_ion,
        )

    partition = (
        np.asarray(template.internal_partition_function)
        if template.internal_partition_function is not None
        else np.ones_like(temperature)
    )
    species = species_for_partition(partition)
    neutral = species[0]

    neutral, proton, molecule, molecular_ion, negative, trihydrogen_ion = species
    positive_charge = proton + molecular_ion + trihydrogen_ion
    required_electron = positive_charge - negative
    neutral_scale = neutral / np.maximum(template.neutral_h_density, tiny)
    return (
        HydrogenLTEState(
            mass_density=np.asarray(HYDROGEN_MASS * nuclei_density),
            hydrogen_nuclei_density=np.asarray(nuclei_density),
            neutral_h_density=np.asarray(neutral),
            proton_density=np.asarray(proton),
            electron_density=np.asarray(electron_density),
            ionization_fraction=np.asarray(proton / np.maximum(nuclei_density, tiny)),
            internal_partition_function=partition,
            level_occupation_probability=template.level_occupation_probability,
            level_population_density=(
                None if template.level_population_density is None
                else template.level_population_density
                * neutral_scale[..., np.newaxis]
            ),
            microfield_model=template.microfield_model,
            molecular_hydrogen_density=(
                np.asarray(molecule) if include_molecules else None
            ),
            molecular_hydrogen_ion_density=(
                np.asarray(molecular_ion) if include_molecules else None
            ),
            negative_hydrogen_density=(
                np.asarray(negative) if include_negative else None
            ),
            trihydrogen_ion_density=(
                np.asarray(trihydrogen_ion) if h3_model is not None else None
            ),
            trihydrogen_ion_partition_model=h3_model,
            chemical_model=template.chemical_model + "+trace-metal-charge-neutral",
            neutral_radius_scale=template.neutral_radius_scale,
        ),
        np.asarray(required_electron),
    )


def metal_lte_state(
    atmosphere: Atmosphere,
    atomic_database: AtomicDatabase,
    log_number_abundance: Mapping[str, float],
    *,
    reference_species: Literal["auto", "H", "He"] = "auto",
    include_dense_helium_ionization: bool = True,
    log_hydrogen_abundance: float | None = None,
    trace_hydrogen_correlated_microfields: bool = True,
    occupation_probability_partitions: bool = False,
) -> MetalLTEState:
    """Solve trace-metal Saha populations and total electron density.

    Abundances are base-10 logarithmic number ratios relative to hydrogen or
    helium nuclei. In a hydrogen atmosphere, the H/H+/H2/H2+/H-/H3+ balance
    is re-solved at fixed H-nuclei density. In a helium atmosphere the host
    He I/II/III balance is re-solved at fixed density with the existing HM
    partition functions, so metal-donated electrons feed back consistently
    on charge neutrality.
    ``log_hydrogen_abundance`` adds atomic trace H to a helium atmosphere and
    solves its HM partition function and proton contribution in the same
    charge-neutrality equation. This approximation is intended for warm DZA
    validation stars; it deliberately does not claim a cool mixed H/He/H2
    nonideal EOS.

    ``occupation_probability_partitions`` (helium hosts) replaces the fixed
    spectroscopic partition cutoff by Hummer--Mihalas occupation
    probabilities for charged Q-MHD microfields and ground-state neutral He
    perturbers.  The partitions depend on the electron density, so they are
    iterated with charge neutrality to self-consistency.  The state is then
    marked for level dissolution, and line opacity applies the same survival
    probabilities.
    """

    if reference_species == "auto":
        reference_species = "He" if atmosphere.helium_lte_state is not None else "H"
    if reference_species == "He":
        host_state = atmosphere.helium_lte_state
        if host_state is None:
            raise ValueError("a helium reference requires atmosphere.helium_lte_state")
        reference_density = host_state.helium_nuclei_density
    elif reference_species == "H":
        host_state = atmosphere.hydrogen_lte_state
        if host_state is None:
            raise ValueError("a hydrogen reference requires atmosphere.hydrogen_lte_state")
        reference_density = host_state.hydrogen_nuclei_density
    else:
        raise ValueError("reference_species must be 'auto', 'H', or 'He'")
    if log_hydrogen_abundance is not None:
        if reference_species != "He":
            raise ValueError("trace hydrogen is currently supported only in helium atmospheres")
        log_hydrogen_abundance = float(log_hydrogen_abundance)
        if not np.isfinite(log_hydrogen_abundance):
            raise ValueError("log_hydrogen_abundance must be finite")

    abundances: dict[str, float] = {}
    element_density: dict[str, FloatArray] = {}
    ion_stages: dict[str, tuple[AtomicIon, ...]] = {}
    partitions: dict[tuple[str, int], FloatArray] = {}
    for element, abundance in log_number_abundance.items():
        symbol = _canonical_element(element)
        value = float(abundance)
        if not np.isfinite(value):
            raise ValueError("log abundances must be finite")
        stages = atomic_database.ion_stages(symbol)
        if len(stages) < 2:
            raise ValueError(f"at least two ion stages are required for {symbol}")
        abundances[symbol] = value
        element_density[symbol] = np.asarray(reference_density * 10.0**value)
        ion_stages[symbol] = stages
        for ion in stages:
            partitions[(symbol, ion.charge)] = ion.partition_function(atmosphere.temperature)
    if not abundances:
        raise ValueError("at least one metal abundance is required")

    temperature = atmosphere.temperature
    translational_log = 1.5 * np.log(
        2.0 * PI * ELECTRON_MASS * BOLTZMANN * temperature / PLANCK**2
    )
    helium_density = (
        atmosphere.helium_lte_state.helium_nuclei_density
        if atmosphere.helium_lte_state is not None else np.zeros_like(temperature)
    )
    helium_mass_density = HELIUM_MASS * helium_density
    trace_hydrogen_density = (
        np.zeros_like(temperature)
        if log_hydrogen_abundance is None
        else helium_density * 10.0**log_hydrogen_abundance
    )
    hydrogen_host_template = (
        host_state if isinstance(host_state, HydrogenLTEState) else None
    )

    def populations_at_electron_density(
        electron_density: FloatArray,
    ) -> tuple[
        dict[str, FloatArray],
        FloatArray,
        FloatArray | None,
        tuple[FloatArray, FloatArray, object] | None,
        HydrogenLTEState | None,
    ]:
        populations: dict[str, FloatArray] = {}
        metal_charge = np.zeros_like(temperature)
        for symbol, stages in ion_stages.items():
            ratios = []
            for lower, upper in zip(stages[:-1], stages[1:]):
                ionization_ev = lower.ionization_energy_ev
                if ionization_ev is None:
                    raise ValueError(f"missing ionization energy for {symbol} {lower.charge:+d}")
                shift = np.zeros_like(temperature)
                if (
                    reference_species == "He"
                    and include_dense_helium_ionization
                    and lower.charge == 0
                ):
                    shift = dense_helium_ionization_potential_shift_ev(
                        symbol, helium_mass_density, temperature
                    )
                effective_ionization = np.maximum(ionization_ev + shift, 0.05)
                lower_partition = partitions[(symbol, lower.charge)]
                upper_partition = partitions[(symbol, upper.charge)]
                ratios.append(
                    np.log(2.0)
                    + translational_log
                    + np.log(upper_partition)
                    - np.log(lower_partition)
                    - effective_ionization * EV_TO_ERG / (BOLTZMANN * temperature)
                    - np.log(electron_density)
                )
            fraction = _ion_fractions(np.stack(ratios))
            population = element_density[symbol][np.newaxis, :] * fraction
            populations[symbol] = population
            metal_charge += np.sum(
                np.arange(population.shape[0])[:, np.newaxis] * population,
                axis=0,
            )

        helium_fraction = None
        if reference_species == "He":
            assert atmosphere.helium_lte_state is not None
            helium = atmosphere.helium_lte_state
            first_ratio = (
                np.log(2.0) + translational_log
                + np.log(helium.singly_ionized_partition_function)
                - np.log(helium.neutral_partition_function)
                - 24.587_389_011 * EV_TO_ERG / (BOLTZMANN * temperature)
                - np.log(electron_density)
            )
            second_ratio = (
                np.log(2.0) + translational_log
                - np.log(helium.singly_ionized_partition_function)
                - 54.417_765_528_2 * EV_TO_ERG / (BOLTZMANN * temperature)
                - np.log(electron_density)
            )
            helium_fraction = _ion_fractions(np.stack((first_ratio, second_ratio)))

        trace_hydrogen = None
        hydrogen_charge = np.zeros_like(temperature)
        if log_hydrogen_abundance is not None:
            # Trace-H levels are not dissolved by neutral He.  The HM88
            # hard-sphere term with r_n = n^2 a_0 removes almost all n >= 6
            # population in DBA photospheres (N_He ~ 1e20 cm^-3), erasing the
            # observed Hdelta of WD J1013+0259; it was reverted on 2026-10-01.
            # At fixed total H nuclei and total electron density, iterate the
            # occupation-probability partition function and atomic Saha ratio.
            # H2 is negligible for the warm Ross 640 use case and is excluded
            # explicitly rather than folded into an uncontrolled closure.
            neutral_hydrogen = trace_hydrogen_density.copy()
            distribution_electron_density = np.maximum(electron_density, 1.0)
            distribution = hydrogen_level_distribution(
                neutral_hydrogen,
                distribution_electron_density,
                temperature,
                correlated_microfields=trace_hydrogen_correlated_microfields,
            )
            for _ in range(12):
                log_saha_ratio = (
                    np.log(hydrogen_saha_constant(temperature))
                    - np.log(np.maximum(
                        electron_density, np.finfo(np.float64).tiny
                    ))
                    - np.log(np.maximum(
                        distribution.internal_partition_function,
                        np.finfo(np.float64).tiny,
                    ))
                )
                # The charge-neutrality bracket deliberately visits almost
                # electron-free trial states.  Evaluate the neutral fraction
                # as 1/(1+exp(log Saha ratio)) in log space so those harmless
                # endpoints do not overflow before bisection rejects them.
                candidate_neutral = trace_hydrogen_density * np.exp(
                    -np.logaddexp(0.0, log_saha_ratio)
                )
                neutral_hydrogen = candidate_neutral
                distribution = hydrogen_level_distribution(
                    neutral_hydrogen,
                    distribution_electron_density,
                    temperature,
                    correlated_microfields=trace_hydrogen_correlated_microfields,
                )
            proton_hydrogen = trace_hydrogen_density - neutral_hydrogen
            hydrogen_charge = proton_hydrogen
            trace_hydrogen = (
                neutral_hydrogen,
                proton_hydrogen,
                distribution,
            )

        host_charge = None
        host_hydrogen_state = None
        if reference_species == "He":
            host_charge = helium_density[np.newaxis, :] * helium_fraction
            total_charge = (
                metal_charge + hydrogen_charge
                + host_charge[1] + 2.0 * host_charge[2]
            )
        else:
            assert hydrogen_host_template is not None
            host_hydrogen_state, hydrogen_charge = (
                _hydrogen_state_with_trace_metal_electrons(
                    hydrogen_host_template,
                    temperature,
                    electron_density,
                )
            )
            total_charge = metal_charge + hydrogen_charge
        return (
            populations,
            total_charge,
            host_charge,
            trace_hydrogen,
            host_hydrogen_state,
        )

    maximum_charge = sum(
        len(ion_stages[symbol]) - 1 for symbol in abundances
    )
    charge_ceiling = (
        (
            2.0 * helium_density
            if reference_species == "He"
            else reference_density
        )
        + maximum_charge * sum(element_density.values())
        + trace_hydrogen_density
    )
    # H occupation probabilities depend on the shared electron density. Use
    # the previous host partition as a cheap charge-root preconditioner before
    # the damped joint charge/partition completion below. This avoids
    # evaluating all bound levels at every wide-bracket bisection endpoint.
    lower = np.full_like(temperature, np.log(np.finfo(np.float64).tiny))
    upper = np.log(np.maximum(charge_ceiling * 1.01, 1.0))
    for _ in range(48):
        middle = 0.5 * (lower + upper)
        electron_density = np.exp(middle)
        _, required_charge, _, _, _ = populations_at_electron_density(
            electron_density
        )
        positive = electron_density > required_charge
        upper = np.where(positive, middle, upper)
        lower = np.where(positive, lower, middle)
    electron_density = np.exp(0.5 * (lower + upper))
    neutral_perturbers = None
    neutral_radii = None
    if occupation_probability_partitions:
        if reference_species != "He":
            raise ValueError("occupation-probability metal partitions require a helium host")
        assert atmosphere.helium_lte_state is not None
        neutral_perturbers = {
            "He": np.asarray(atmosphere.helium_lte_state.neutral_he_density)
        }
        neutral_radii = {
            "He": float(atmosphere.helium_lte_state.neutral_radius_scale)
            * BOHR_RADIUS_CM
        }
        for _ in range(40):
            updated = {
                (symbol, ion.charge): ion.occupation_weighted_partition_function(
                    temperature,
                    electron_density,
                    neutral_perturber_number_density=neutral_perturbers,
                    neutral_perturber_radius_cm=neutral_radii,
                )
                for symbol, stages in ion_stages.items()
                for ion in stages
            }
            change = max(
                float(np.max(np.abs(np.log(
                    updated[key] / np.maximum(partitions[key], np.finfo(np.float64).tiny)
                ))))
                for key in updated
            )
            partitions.update(updated)
            lower = np.log(electron_density) - 2.0
            upper = np.log(electron_density) + 2.0
            for _ in range(44):
                middle = 0.5 * (lower + upper)
                trial = np.exp(middle)
                _, required_charge, _, _, _ = populations_at_electron_density(trial)
                positive = trial > required_charge
                upper = np.where(positive, middle, upper)
                lower = np.where(positive, lower, middle)
            electron_density = np.exp(0.5 * (lower + upper))
            if change < 1.0e-10:
                break
    population_result = populations_at_electron_density(
        electron_density,
    )
    if reference_species == "H":
        # The final H-host closure has two unknowns at each depth: total n_e
        # and the HM internal partition U_H. Solving charge first and applying
        # occupation probabilities afterward is not adequate in cool dense
        # layers, where the partition map can have a very steep slope. Use a
        # small, depth-vectorized 2x2 Newton solve for log(n_e) and log(U_H).
        assert hydrogen_host_template is not None
        log_electron = np.log(electron_density)
        log_partition = np.log(np.maximum(
            hydrogen_host_template.internal_partition_function,
            np.finfo(np.float64).tiny,
        ))

        def coupled_hydrogen_residual(
            trial_log_electron: FloatArray,
            trial_log_partition: FloatArray,
        ):
            nonlocal hydrogen_host_template
            hydrogen_host_template = replace(
                hydrogen_host_template,
                internal_partition_function=np.exp(trial_log_partition),
            )
            trial_electron = np.exp(trial_log_electron)
            trial_result = populations_at_electron_density(
                trial_electron,
            )
            trial_hydrogen = trial_result[4]
            assert trial_hydrogen is not None
            trial_distribution = hydrogen_level_distribution(
                trial_hydrogen.neutral_h_density,
                trial_electron,
                temperature,
                maximum_level=(
                    trial_hydrogen.level_population_density.shape[-1]
                    if trial_hydrogen.level_population_density is not None
                    else 40
                ),
                neutral_radius_scale=trial_hydrogen.neutral_radius_scale,
                correlated_microfields=(
                    trial_hydrogen.microfield_model == "qmhd"
                ),
            )
            charge_residual = trial_log_electron - np.log(np.maximum(
                trial_result[1], np.finfo(np.float64).tiny
            ))
            partition_residual = trial_log_partition - np.log(np.maximum(
                trial_distribution.internal_partition_function,
                np.finfo(np.float64).tiny,
            ))
            return (
                charge_residual,
                partition_residual,
                trial_result,
                trial_distribution,
            )

        base = coupled_hydrogen_residual(log_electron, log_partition)
        finite_step = 2.0e-4
        for _ in range(30):
            charge_residual, partition_residual = base[:2]
            merit = float(np.max(np.hypot(
                charge_residual, partition_residual
            )))
            if merit < 2.0e-9:
                break
            electron_shift = coupled_hydrogen_residual(
                log_electron + finite_step, log_partition
            )
            partition_shift = coupled_hydrogen_residual(
                log_electron, log_partition + finite_step
            )
            a = (electron_shift[0] - charge_residual) / finite_step
            c = (electron_shift[1] - partition_residual) / finite_step
            b = (partition_shift[0] - charge_residual) / finite_step
            d = (partition_shift[1] - partition_residual) / finite_step
            determinant = a * d - b * c
            usable = np.isfinite(determinant) & (np.abs(determinant) > 1.0e-8)
            delta_electron = np.where(
                usable,
                (-d * charge_residual + b * partition_residual)
                / determinant,
                -0.4 * charge_residual,
            )
            delta_partition = np.where(
                usable,
                (c * charge_residual - a * partition_residual)
                / determinant,
                -0.2 * partition_residual,
            )
            delta_electron = np.clip(delta_electron, -0.25, 0.25)
            delta_partition = np.clip(delta_partition, -0.25, 0.25)
            accepted = False
            line_factor = 1.0
            for _ in range(8):
                trial = coupled_hydrogen_residual(
                    log_electron + line_factor * delta_electron,
                    log_partition + line_factor * delta_partition,
                )
                trial_merit = float(np.max(np.hypot(trial[0], trial[1])))
                if trial_merit < merit:
                    log_electron += line_factor * delta_electron
                    log_partition += line_factor * delta_partition
                    base = trial
                    accepted = True
                    break
                line_factor *= 0.5
            if not accepted:
                log_electron -= 0.1 * charge_residual
                log_partition -= 0.05 * partition_residual
                base = coupled_hydrogen_residual(
                    log_electron, log_partition
                )
        electron_density = np.exp(log_electron)
        population_result = base[2]
        final_hydrogen = population_result[4]
        assert final_hydrogen is not None
        final_distribution = base[3]
        final_hydrogen = replace(
            final_hydrogen,
            internal_partition_function=(
                final_distribution.internal_partition_function
            ),
            level_occupation_probability=(
                final_distribution.occupation_probability
            ),
            level_population_density=final_distribution.population_density,
        )
        population_result = (*population_result[:4], final_hydrogen)
    assert population_result is not None
    (
        populations,
        _,
        host_population,
        trace_hydrogen,
        host_hydrogen_state,
    ) = population_result
    metal_electrons = np.zeros_like(temperature)
    for population in populations.values():
        metal_electrons += np.sum(
            np.arange(population.shape[0])[:, np.newaxis] * population, axis=0
        )
    trace_hydrogen_state = None
    if trace_hydrogen is not None:
        neutral_hydrogen, proton_hydrogen, distribution = trace_hydrogen
        trace_hydrogen_state = HydrogenLTEState(
            mass_density=np.asarray(HYDROGEN_MASS * trace_hydrogen_density),
            hydrogen_nuclei_density=np.asarray(trace_hydrogen_density),
            neutral_h_density=np.asarray(neutral_hydrogen),
            proton_density=np.asarray(proton_hydrogen),
            electron_density=np.asarray(electron_density),
            ionization_fraction=np.divide(
                proton_hydrogen,
                trace_hydrogen_density,
                out=np.zeros_like(proton_hydrogen),
                where=trace_hydrogen_density > 0.0,
            ),
            internal_partition_function=distribution.internal_partition_function,
            level_occupation_probability=distribution.occupation_probability,
            level_population_density=distribution.population_density,
            microfield_model=(
                "qmhd" if trace_hydrogen_correlated_microfields else "holtsmark"
            ),
            chemical_model="trace-atomic-hm-in-helium",
        )
    return MetalLTEState(
        reference_species=reference_species,
        log_number_abundance=MappingProxyType(abundances),
        element_number_density=MappingProxyType(element_density),
        ion_number_density=MappingProxyType(populations),
        partition_function=MappingProxyType(partitions),
        electron_density=np.asarray(electron_density),
        metal_electron_density=np.asarray(metal_electrons),
        host_ion_number_density=host_population,
        host_hydrogen_state=host_hydrogen_state,
        nonideal_ionization=bool(
            reference_species == "He" and include_dense_helium_ionization
        ),
        log_hydrogen_abundance=log_hydrogen_abundance,
        trace_hydrogen_state=trace_hydrogen_state,
        metal_level_dissolution=bool(occupation_probability_partitions),
        neutral_perturber_number_density=(
            None if neutral_perturbers is None else MappingProxyType(neutral_perturbers)
        ),
        neutral_perturber_radius_cm=(
            None if neutral_radii is None else MappingProxyType(neutral_radii)
        ),
    )


def helium_metal_number_abundances_from_mass_fractions(
    mass_fraction: Mapping[str, float],
) -> dict[str, float]:
    """Convert a homogeneous He/metal mixture to log number ratios to He.

    The input is normalized internally, but it must contain helium and at
    least one supported metal.  This is the abundance convention needed by
    PG 1159 work, whose published He/C/O compositions are normally mass
    fractions rather than trace number abundances.
    """

    values = {key.strip().capitalize(): float(value) for key, value in mass_fraction.items()}
    if "He" not in values or values["He"] <= 0.0:
        raise ValueError("mass fractions require a positive He entry")
    unsupported = set(values) - ({"He"} | set(ATOMIC_MASS_U))
    if unsupported:
        raise ValueError(
            "unsupported mass-fraction elements: " + ", ".join(sorted(unsupported))
        )
    if len(values) < 2:
        raise ValueError("at least one metal mass fraction is required")
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("mass fractions must be finite and non-negative")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("mass fractions must have a positive sum")
    normalized = {element: value / total for element, value in values.items()}
    helium_number_per_mass = normalized["He"] / _HELIUM_ATOMIC_MASS_U
    return {
        element: float(np.log10(
            (normalized[element] / ATOMIC_MASS_U[element]) / helium_number_per_mass
        ))
        for element in normalized
        if element != "He" and normalized[element] > 0.0
    }


def helium_metal_lte_state_from_mass_fractions(
    atmosphere: Atmosphere,
    atomic_database: AtomicDatabase,
    mass_fraction: Mapping[str, float],
) -> MetalLTEState:
    """Solve an ideal homogeneous He/metal LTE mixture at fixed gas pressure.

    Unlike :func:`metal_lte_state`, metals are not assumed to be trace.  The
    common electron density, all ion stages, the helium nuclei density, and
    the total mass density are solved together using charge neutrality and
    the fixed-pressure particle equation.  The existing helium HM partition
    functions are retained as the thermodynamic reference; C/O partition
    functions come from the loaded Stout levels.

    This is the composition closure for the first PG 1159 milestone.  It is
    intentionally an LTE reference state: the later NLTE solver scales it by
    ion/level departure coefficients.
    """

    helium = atmosphere.helium_lte_state
    if helium is None:
        raise ValueError("a helium atmosphere is required for a bulk He/metal mixture")
    abundances = helium_metal_number_abundances_from_mass_fractions(mass_fraction)
    if not abundances:
        raise ValueError("at least one positive metal mass fraction is required")

    raw = {key.strip().capitalize(): float(value) for key, value in mass_fraction.items()}
    total_mass_fraction = sum(raw.values())
    normalized_mass_fraction = {
        element: value / total_mass_fraction for element, value in raw.items()
        if value > 0.0
    }
    number_ratio = {element: 10.0**value for element, value in abundances.items()}
    nuclei_per_helium = 1.0 + sum(number_ratio.values())
    temperature = atmosphere.temperature
    particle_density = atmosphere.gas_pressure / (BOLTZMANN * temperature)
    translational_log = 1.5 * np.log(
        2.0 * PI * ELECTRON_MASS * BOLTZMANN * temperature / PLANCK**2
    )

    ion_stages: dict[str, tuple[AtomicIon, ...]] = {}
    partitions: dict[tuple[str, int], FloatArray] = {}
    for element in abundances:
        stages = atomic_database.ion_stages(element)
        if len(stages) < 2:
            raise ValueError(f"at least two ion stages are required for {element}")
        ion_stages[element] = stages
        for ion in stages:
            partitions[(element, ion.charge)] = ion.partition_function(temperature)

    def fractions_and_charge(
        electron_density: FloatArray,
    ) -> tuple[FloatArray, dict[str, FloatArray], FloatArray]:
        first_ratio = (
            np.log(2.0)
            + translational_log
            + np.log(helium.singly_ionized_partition_function)
            - np.log(helium.neutral_partition_function)
            - 24.587_389_011 * EV_TO_ERG / (BOLTZMANN * temperature)
            - np.log(electron_density)
        )
        second_ratio = (
            np.log(2.0)
            + translational_log
            - np.log(helium.singly_ionized_partition_function)
            - 54.417_765_528_2 * EV_TO_ERG / (BOLTZMANN * temperature)
            - np.log(electron_density)
        )
        helium_fraction = _ion_fractions(np.stack((first_ratio, second_ratio)))
        charge_per_helium = helium_fraction[1] + 2.0 * helium_fraction[2]
        metal_fraction: dict[str, FloatArray] = {}
        for element, stages in ion_stages.items():
            ratios = []
            for lower, upper in zip(stages[:-1], stages[1:]):
                if lower.ionization_energy_ev is None:
                    raise ValueError(
                        f"missing ionization energy for {element} {lower.charge:+d}"
                    )
                ratios.append(
                    np.log(2.0)
                    + translational_log
                    + np.log(partitions[(element, upper.charge)])
                    - np.log(partitions[(element, lower.charge)])
                    - lower.ionization_energy_ev * EV_TO_ERG
                    / (BOLTZMANN * temperature)
                    - np.log(electron_density)
                )
            fraction = _ion_fractions(np.stack(ratios))
            metal_fraction[element] = fraction
            charge_per_helium += number_ratio[element] * np.sum(
                np.arange(fraction.shape[0])[:, np.newaxis] * fraction,
                axis=0,
            )
        return helium_fraction, metal_fraction, charge_per_helium

    lower = np.full_like(temperature, np.log(np.finfo(np.float64).tiny))
    upper = np.log(np.maximum(particle_density * (1.0 - 1.0e-12), 1.0))
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        electron_density = np.exp(middle)
        _, _, charge_per_helium = fractions_and_charge(electron_density)
        helium_density = particle_density / (
            nuclei_per_helium + charge_per_helium
        )
        required_electrons = helium_density * charge_per_helium
        positive = electron_density > required_electrons
        upper = np.where(positive, middle, upper)
        lower = np.where(positive, lower, middle)

    electron_density = np.exp(0.5 * (lower + upper))
    helium_fraction, metal_fraction, charge_per_helium = fractions_and_charge(
        electron_density
    )
    helium_density = particle_density / (nuclei_per_helium + charge_per_helium)
    host_population = helium_density[np.newaxis, :] * helium_fraction
    element_density = {
        element: np.asarray(helium_density * ratio)
        for element, ratio in number_ratio.items()
    }
    populations = {
        element: element_density[element][np.newaxis, :] * metal_fraction[element]
        for element in abundances
    }
    metal_electrons = np.zeros_like(temperature)
    for population in populations.values():
        metal_electrons += np.sum(
            np.arange(population.shape[0])[:, np.newaxis] * population,
            axis=0,
        )
    atomic_mass_unit = 1.660_539_068_92e-24
    total_mass_density = HELIUM_MASS * helium_density
    for element, density in element_density.items():
        total_mass_density += ATOMIC_MASS_U[element] * atomic_mass_unit * density

    return MetalLTEState(
        reference_species="He",
        log_number_abundance=MappingProxyType(dict(abundances)),
        element_number_density=MappingProxyType(element_density),
        ion_number_density=MappingProxyType(populations),
        partition_function=MappingProxyType(partitions),
        electron_density=np.asarray(electron_density),
        metal_electron_density=np.asarray(metal_electrons),
        host_ion_number_density=np.asarray(host_population),
        nonideal_ionization=False,
        composition_mode="bulk",
        mass_fraction=MappingProxyType(normalized_mass_fraction),
        total_mass_density=np.asarray(total_mass_density),
    )


def atmosphere_with_metal_electrons(
    atmosphere: Atmosphere,
    metal_state: MetalLTEState,
) -> Atmosphere:
    """Return a fixed-structure atmosphere with composition-consistent electrons."""

    if metal_state.electron_density.shape != atmosphere.temperature.shape:
        raise ValueError("metal_state and atmosphere depth grids do not match")
    helium_state = atmosphere.helium_lte_state
    hydrogen_state = atmosphere.hydrogen_lte_state
    if metal_state.reference_species == "He":
        if helium_state is None or metal_state.host_ion_number_density is None:
            raise ValueError("helium metal state is inconsistent with atmosphere")
        host = metal_state.host_ion_number_density
        neutral_scale = host[0] / np.maximum(
            helium_state.neutral_he_density, np.finfo(np.float64).tiny
        )
        helium_state = replace(
            helium_state,
            mass_density=np.asarray(HELIUM_MASS * np.sum(host, axis=0)),
            helium_nuclei_density=np.asarray(np.sum(host, axis=0)),
            neutral_he_density=np.asarray(host[0]),
            singly_ionized_he_density=np.asarray(host[1]),
            doubly_ionized_he_density=np.asarray(host[2]),
            electron_density=np.asarray(metal_state.electron_density),
            mean_ion_charge=np.asarray(
                (host[1] + 2.0 * host[2])
                / np.maximum(np.sum(host, axis=0), np.finfo(np.float64).tiny)
            ),
            neutral_level_population_density=np.asarray(
                helium_state.neutral_level_population_density * neutral_scale[:, np.newaxis]
            ),
        )
    if metal_state.reference_species == "H":
        if hydrogen_state is None or metal_state.host_hydrogen_state is None:
            raise ValueError("hydrogen metal state is inconsistent with atmosphere")
        hydrogen_state = metal_state.host_hydrogen_state
    elif metal_state.trace_hydrogen_state is not None:
        hydrogen_state = metal_state.trace_hydrogen_state
    metadata = dict(atmosphere.metadata)
    metadata.update(
        {
            "metal_abundances": dict(metal_state.log_number_abundance),
            "metal_electron_feedback": (
                "bulk fixed-pressure particle and charge closure"
                if metal_state.composition_mode == "bulk"
                else "fixed-host-density shared H/metal charge closure"
                if metal_state.reference_species == "H"
                else "fixed thermodynamic structure"
            ),
            "metal_nonideal_ionization": metal_state.nonideal_ionization,
            "log_hydrogen_abundance": metal_state.log_hydrogen_abundance,
            "metal_composition_mode": metal_state.composition_mode,
            "mass_fractions": (
                dict(metal_state.mass_fraction)
                if metal_state.mass_fraction is not None else None
            ),
        }
    )
    return replace(
        atmosphere,
        neutral_h_density=(
            atmosphere.neutral_h_density
            if hydrogen_state is None
            else hydrogen_state.neutral_h_density
        ),
        proton_density=(
            atmosphere.proton_density
            if hydrogen_state is None
            else hydrogen_state.proton_density
        ),
        electron_density=np.asarray(metal_state.electron_density),
        mass_density=(
            atmosphere.mass_density
            if metal_state.total_mass_density is None
            else np.asarray(metal_state.total_mass_density)
        ),
        hydrogen_lte_state=hydrogen_state,
        helium_lte_state=helium_state,
        metadata=metadata,
    )


def _pseudo_voigt_profile_per_angstrom(
    wavelength: FloatArray,
    center: float,
    gaussian_sigma: float,
    lorentz_hwhm: float,
) -> FloatArray:
    gaussian_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * max(gaussian_sigma, 1.0e-12)
    lorentz_fwhm = 2.0 * max(lorentz_hwhm, 0.0)
    width = (
        gaussian_fwhm**5 + 2.69269 * gaussian_fwhm**4 * lorentz_fwhm
        + 2.42843 * gaussian_fwhm**3 * lorentz_fwhm**2
        + 4.47163 * gaussian_fwhm**2 * lorentz_fwhm**3
        + 0.07842 * gaussian_fwhm * lorentz_fwhm**4 + lorentz_fwhm**5
    ) ** 0.2
    ratio = lorentz_fwhm / max(width, np.finfo(np.float64).tiny)
    mixing = np.clip(1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3, 0.0, 1.0)
    offset = wavelength - center
    gaussian = (
        2.0 * np.sqrt(np.log(2.0)) / (np.sqrt(np.pi) * width)
        * np.exp(-4.0 * np.log(2.0) * (offset / width) ** 2)
    )
    lorentz = 2.0 / (np.pi * width) / (1.0 + 4.0 * (offset / width) ** 2)
    return (1.0 - mixing) * gaussian + mixing * lorentz


def _humlicek_w4(x: FloatArray, y: FloatArray) -> FloatArray:
    """Return the real part of the Faddeeva function, ``Re w(x + i y)``.

    Humlicek (1982, JQSRT 27, 437) region algorithm W4, accurate to about
    1e-4 relative for ``y >= 0``.  The compiled line kernel uses the same
    four regions and coefficients.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.broadcast_to(np.asarray(y, dtype=np.float64), x.shape)
    t = y - 1j * x
    s = np.abs(x) + y
    w = np.empty(x.shape, dtype=np.complex128)
    region1 = s >= 15.0
    region2 = (s >= 5.5) & ~region1
    region3 = (s < 5.5) & (y >= 0.195 * np.abs(x) - 0.176)
    region4 = ~(region1 | region2 | region3)
    if np.any(region1):
        tt = t[region1]
        w[region1] = tt * 0.5641896 / (0.5 + tt * tt)
    if np.any(region2):
        tt = t[region2]
        u = tt * tt
        w[region2] = tt * (1.410474 + u * 0.5641896) / (0.75 + u * (3.0 + u))
    if np.any(region3):
        tt = t[region3]
        w[region3] = (
            16.4955 + tt * (20.20933 + tt * (11.96482 + tt * (3.778987 + tt * 0.5642236)))
        ) / (
            16.4955 + tt * (38.82363 + tt * (39.27121 + tt * (21.69274 + tt * (6.699398 + tt))))
        )
    if np.any(region4):
        tt = t[region4]
        u = tt * tt
        w[region4] = np.exp(u) - tt * (
            36183.31 - u * (3321.9905 - u * (1540.787 - u * (219.0313 - u * (
                35.76683 - u * (1.320522 - u * 0.56419)))))
        ) / (
            32066.6 - u * (24322.84 - u * (9022.228 - u * (2186.181 - u * (
                364.2191 - u * (61.57037 - u * (1.841439 - u))))))
        )
    return w.real


def _voigt_profile_per_angstrom(
    wavelength: FloatArray,
    center: float,
    gaussian_sigma: float,
    lorentz_hwhm: float,
) -> FloatArray:
    """Return an exact Voigt profile evaluated in frequency.

    Impact broadening gives a Lorentzian in frequency, and the thermal
    Doppler profile is Gaussian in frequency, so the Voigt function is
    evaluated at ``nu = c/lambda`` with the widths converted at line centre.
    The result is expressed per Angstrom at line centre, ``phi_nu c /
    lambda0^2``, so callers multiply by ``1e8 lambda0^2/c`` exactly as for a
    wavelength profile.  Unlike a symmetric wavelength Lorentzian, this keeps
    the physical ``(lambda/lambda0)^2`` asymmetry of distant impact wings,
    and unlike the Thompson--Cox--Hastings pseudo-Voigt it is accurate in
    the far wings of weakly damped lines.
    """

    wavelength = np.asarray(wavelength, dtype=np.float64)
    center_cm = center * 1.0e-8
    per_angstrom = LIGHT_SPEED / center_cm**2 * 1.0e-8
    sigma_nu = max(gaussian_sigma, 1.0e-12) * per_angstrom
    gamma_nu = max(lorentz_hwhm, 0.0) * per_angstrom
    detuning = LIGHT_SPEED / (wavelength * 1.0e-8) - LIGHT_SPEED / center_cm
    scale = 1.0 / (sigma_nu * np.sqrt(2.0))
    profile_nu = _humlicek_w4(detuning * scale, gamma_nu * scale) * scale / np.sqrt(PI)
    return profile_nu * per_angstrom


def _pseudo_voigt_profile_grid(
    wavelength: FloatArray,
    center: float,
    gaussian_sigma: FloatArray,
    lorentz_hwhm: FloatArray,
) -> FloatArray:
    """Depth-vectorized :func:`_pseudo_voigt_profile_per_angstrom`."""

    gaussian_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * np.maximum(gaussian_sigma, 1.0e-12)
    lorentz_fwhm = 2.0 * np.maximum(lorentz_hwhm, 0.0)
    width = (
        gaussian_fwhm**5 + 2.69269 * gaussian_fwhm**4 * lorentz_fwhm
        + 2.42843 * gaussian_fwhm**3 * lorentz_fwhm**2
        + 4.47163 * gaussian_fwhm**2 * lorentz_fwhm**3
        + 0.07842 * gaussian_fwhm * lorentz_fwhm**4 + lorentz_fwhm**5
    ) ** 0.2
    ratio = lorentz_fwhm / np.maximum(width, np.finfo(np.float64).tiny)
    mixing = np.clip(1.36603 * ratio - 0.47719 * ratio**2 + 0.11116 * ratio**3, 0.0, 1.0)
    offset = (wavelength - center)[:, np.newaxis] / width[np.newaxis, :]
    gaussian = (
        2.0 * np.sqrt(np.log(2.0)) / (np.sqrt(np.pi) * width[np.newaxis, :])
        * np.exp(-4.0 * np.log(2.0) * offset**2)
    )
    lorentz = 2.0 / (np.pi * width[np.newaxis, :]) / (1.0 + 4.0 * offset**2)
    return (1.0 - mixing)[np.newaxis, :] * gaussian + mixing[np.newaxis, :] * lorentz


def _quasistatic_line_profile(
    wavelength: FloatArray,
    center: float,
    gaussian_sigma: float,
    lorentz_hwhm: float,
    frequency_scale: float,
    mixing: float,
    support_half_width: float | None = None,
) -> FloatArray:
    """Impact profile blended with its quasi-static Holtsmark envelope.

    Returns ``(1 - mixing) V + mixing max(V, H)`` (unnormalized) on
    ``wavelength``.  Points within 25 impact widths of the centre are
    evaluated exactly; the smooth outer profile is evaluated on log-spaced
    offset nodes and interpolated, which keeps wide quasi-static windows on
    dense opacity grids affordable.
    """

    def exact(points: FloatArray) -> FloatArray:
        impact = _voigt_profile_per_angstrom(
            points, center, gaussian_sigma, lorentz_hwhm
        )
        if frequency_scale <= 0.0 or mixing <= 0.0:
            return impact
        points_cm = points * 1.0e-8
        beta = np.abs(LIGHT_SPEED / points_cm - LIGHT_SPEED / (center * 1.0e-8)) / frequency_scale
        static = (
            _tabulated_holtsmark_distribution(beta) / (2.0 * frequency_scale)
            * LIGHT_SPEED / points_cm**2 * 1.0e-8
        )
        return (1.0 - mixing) * impact + mixing * np.maximum(impact, static)

    offset = wavelength - center
    core = 25.0 * max(gaussian_sigma, lorentz_hwhm, 1.0e-4)
    outer = np.abs(offset) > core
    if support_half_width is None and np.count_nonzero(outer) < 800:
        return exact(wavelength)
    result = np.empty_like(wavelength)
    result[~outer] = exact(wavelength[~outer])
    extent = max(
        float(np.max(np.abs(offset))),
        float(support_half_width) if support_half_width is not None else 0.0,
    )
    nodes = np.geomspace(core, extent * (1.0 + 1.0e-12), 400)
    for sign in (-1.0, 1.0):
        side = outer & (np.sign(offset) == sign)
        if np.any(side):
            node_values = exact(center + sign * nodes)
            result[side] = np.exp(np.interp(
                np.log(np.abs(offset[side])), np.log(nodes),
                np.log(np.maximum(node_values, np.finfo(np.float64).tiny)),
            ))
    return result


def _quasistatic_line_profile_area(
    center: float,
    gaussian_sigma: float,
    lorentz_hwhm: float,
    frequency_scale: float,
    mixing: float,
    half_window: float,
) -> float:
    """Area of :func:`_quasistatic_line_profile` over its full support.

    Evaluated on an internal grid (linear core, logarithmic wings) spanning
    ``center +- half_window``, so the normalization does not depend on which
    wavelengths a caller requests.
    """

    width = max(gaussian_sigma, lorentz_hwhm, 1.0e-5)
    core = min(half_window, 25.0 * width)
    linear = np.linspace(-core, core, 1_001)
    if half_window > core:
        wing = np.geomspace(core, half_window, 1_500)[1:]
        offsets = np.concatenate((-wing[::-1], linear, wing))
    else:
        offsets = linear
    grid = center + offsets
    return float(trapezoid(
        _quasistatic_line_profile(
            grid, center, gaussian_sigma, lorentz_hwhm, frequency_scale, mixing
        ),
        grid,
    ))


_HOLTSMARK_TABLE_BETA = np.linspace(0.0, 8.0, 16_001)
_HOLTSMARK_TABLE_VALUE: FloatArray | None = None


def _tabulated_holtsmark_distribution(beta: ArrayLike) -> FloatArray:
    """Holtsmark distribution from a fine table (quadrature evaluated once).

    Below beta = 8 the Laguerre-quadrature values are tabulated at a spacing of
    5e-4 and interpolated linearly; the asymptotic series is used above.
    """

    global _HOLTSMARK_TABLE_VALUE
    if _HOLTSMARK_TABLE_VALUE is None:
        _HOLTSMARK_TABLE_VALUE = _metal_holtsmark_microfield_distribution(
            _HOLTSMARK_TABLE_BETA
        )
    value = np.asarray(beta, dtype=np.float64)
    result = np.interp(value, _HOLTSMARK_TABLE_BETA, _HOLTSMARK_TABLE_VALUE)
    wing = value > 8.0
    if np.any(wing):
        result[wing] = _metal_holtsmark_microfield_distribution(value[wing])
    return result


def optically_thick_line_minimum_half_window_angstrom(
    atmosphere: Atmosphere,
    center_angstrom: float,
    integrated_strength: float,
    gaussian_sigma_angstrom: ArrayLike,
    lorentz_hwhm_angstrom: ArrayLike,
    population_scale: ArrayLike,
    *,
    initial_half_window_angstrom: float = METAL_LINE_MINIMUM_HALF_WINDOW_ANGSTROM,
    maximum_rosseland_optical_depth: float = 2.0,
    edge_optical_depth: float = 1.0e-2,
    maximum_half_window_angstrom: float = 100.0,
) -> float:
    """Return support whose omitted wing is optically thin.

    A fixed number of Lorentz widths is not a safe profile cutoff for a
    saturated line: even a very small fraction of a profile can remain
    optically thick when the line-center optical depth is enormous.  Measure
    the vertical line optical depth at the proposed support edge, integrating
    through the layers relevant to continuum formation, and expand the
    support geometrically until that optical depth is small.

    This criterion controls the transfer error rather than the omitted
    normalized profile area.  It is intentionally independent of element or
    wavelength so the same physics can be used by every atmosphere class.
    """

    sigma = np.asarray(gaussian_sigma_angstrom, dtype=np.float64)
    gamma = np.asarray(lorentz_hwhm_angstrom, dtype=np.float64)
    scale = np.asarray(population_scale, dtype=np.float64)
    if sigma.shape != atmosphere.temperature.shape or (
        gamma.shape != sigma.shape or scale.shape != sigma.shape
    ):
        raise ValueError("line widths and population scale must match depth")
    if (
        not np.isfinite(center_angstrom) or center_angstrom <= 0.0
        or not np.isfinite(integrated_strength) or integrated_strength < 0.0
        or not np.isfinite(initial_half_window_angstrom)
        or initial_half_window_angstrom < 0.0
        or not np.isfinite(maximum_rosseland_optical_depth)
        or maximum_rosseland_optical_depth <= 0.0
        or not np.isfinite(edge_optical_depth) or edge_optical_depth <= 0.0
        or not np.isfinite(maximum_half_window_angstrom)
        or maximum_half_window_angstrom <= 0.0
    ):
        raise ValueError("optical-depth profile-support inputs must be physical")
    use = np.flatnonzero(
        atmosphere.rosseland_optical_depth <= maximum_rosseland_optical_depth
    )
    stop = int(use[-1]) + 1 if use.size else 1
    local_sigma = sigma[:stop]
    local_gamma = gamma[:stop]
    local_scale = scale[:stop]
    column_mass = atmosphere.column_mass[:stop]
    half_window = min(
        max(
            METAL_LINE_MINIMUM_HALF_WINDOW_ANGSTROM,
            initial_half_window_angstrom,
            10.0 * float(np.max(local_sigma)),
            100.0 * float(np.max(local_gamma)),
        ),
        maximum_half_window_angstrom,
    )
    center_cm = center_angstrom * 1.0e-8
    wavelength_factor = 1.0e8 * center_cm**2 / LIGHT_SPEED
    tiny = np.finfo(np.float64).tiny
    while True:
        profile = np.asarray([
            _voigt_profile_per_angstrom(
                np.asarray([center_angstrom + half_window]),
                center_angstrom,
                float(local_sigma[depth]),
                float(local_gamma[depth]),
            )[0]
            for depth in range(stop)
        ])
        opacity = np.maximum(
            integrated_strength * wavelength_factor * profile * local_scale,
            0.0,
        )
        optical_depth = float(opacity[0] * column_mass[0])
        if stop > 1:
            optical_depth += float(np.sum(
                0.5 * (opacity[1:] + opacity[:-1]) * np.diff(column_mass)
            ))
        if optical_depth <= edge_optical_depth + tiny or (
            half_window >= maximum_half_window_angstrom
        ):
            return float(half_window)
        half_window = min(2.0 * half_window, maximum_half_window_angstrom)


def _metal_holtsmark_microfield_distribution(beta: ArrayLike) -> FloatArray:
    """Return the normalized Holtsmark field-magnitude distribution."""

    value = np.asarray(beta, dtype=np.float64)
    flat = value.ravel()
    result = np.empty_like(flat)
    small = flat < 1.0e-3
    central = (flat >= 1.0e-3) & (flat <= 8.0)
    wing = flat > 8.0
    result[small] = 4.0 / (3.0 * PI) * flat[small] ** 2
    if np.any(central):
        y = _METAL_HOLTSMARK_LAGUERRE_ABSCISSA
        weight = _METAL_HOLTSMARK_LAGUERRE_WEIGHT
        argument = flat[central, np.newaxis] * y[np.newaxis, :] ** (2.0 / 3.0)
        result[central] = (
            4.0 * flat[central] / (3.0 * PI)
            * np.sum(
                weight[np.newaxis, :]
                * y[np.newaxis, :] ** (1.0 / 3.0)
                * np.sin(argument),
                axis=1,
            )
        )
    if np.any(wing):
        inverse = 1.0 / flat[wing]
        result[wing] = (
            1.496_033_551_505_373 * inverse**2.5
            + 7.639_437_268_410_976 * inverse**4.0
            + 21.598_984_399_858_832 * inverse**5.5
            - 447.503_958_034_575_65 * inverse**8.5
            - 3_208.563_652_732_61 * inverse**10.0
            - 12_222.451_853_819_328 * inverse**11.5
        )
    return np.maximum(result.reshape(value.shape), 0.0)


def _accumulate_lte_metal_line_profiles(
    wavelength: FloatArray,
    center: FloatArray,
    integrated_strength: FloatArray,
    gaussian_sigma: FloatArray,
    lorentz_hwhm: FloatArray,
    minimum_half_window: FloatArray,
    population_scale: FloatArray,
    absorption: FloatArray,
) -> None:
    """Accumulate an ordinary LTE line batch, using C when available.

    Atomic-data interpretation, level populations, dissolution, and all
    broadening widths remain in Python.  The compiled kernel only performs
    the repeated line/depth/profile arithmetic, so enabling it cannot change
    the physical model.  The reference loop intentionally mirrors the C
    implementation and remains available for source-only installations.
    """

    compiled = (
        None if _rt is None
        else getattr(_rt, "accumulate_lte_metal_line_profiles", None)
    )
    values = tuple(
        np.ascontiguousarray(value, dtype=np.float64)
        for value in (
            wavelength,
            center,
            integrated_strength,
            gaussian_sigma,
            lorentz_hwhm,
            minimum_half_window,
            population_scale,
        )
    )
    if compiled is not None:
        compiled(*values, absorption)
        return

    (
        wavelength,
        center,
        integrated_strength,
        gaussian_sigma,
        lorentz_hwhm,
        minimum_half_window,
        population_scale,
    ) = values
    for line_index, line_center in enumerate(center):
        center_cm = line_center * 1.0e-8
        for depth in range(absorption.shape[1]):
            half_window = max(
                METAL_LINE_MINIMUM_HALF_WINDOW_ANGSTROM,
                minimum_half_window[line_index],
                10.0 * gaussian_sigma[line_index, depth],
                100.0 * lorentz_hwhm[line_index, depth],
            )
            start = int(np.searchsorted(wavelength, line_center - half_window))
            stop = int(np.searchsorted(
                wavelength, line_center + half_window, side="right"
            ))
            if stop <= start:
                continue
            profile_lambda = _voigt_profile_per_angstrom(
                wavelength[start:stop],
                float(line_center),
                float(gaussian_sigma[line_index, depth]),
                float(lorentz_hwhm[line_index, depth]),
            )
            absorption[start:stop, depth] += (
                integrated_strength[line_index]
                * profile_lambda
                * 1.0e8 * center_cm**2 / LIGHT_SPEED
                * population_scale[line_index, depth]
            )


def selected_metal_lines(
    atomic_database: AtomicDatabase,
    elements: Iterable[str],
    wavelength_minimum: float,
    wavelength_maximum: float,
    minimum_oscillator_strength: float,
    maximum_lines: int | None,
    selection_temperature: float = 10_000.0,
    ion_stage_weight: Mapping[tuple[str, int], float] | None = None,
    *,
    flux_weighted: bool = False,
) -> list[tuple[AtomicIon, AtomicTransition]]:
    """Select likely LTE-strong dipole lines intersecting a wavelength band.

    A cap ranked by oscillator strength alone preferentially retains lines
    from highly excited, negligibly populated lower levels.  Rank instead by
    ``abundance * ion_fraction * g_lower * f * exp(-E_lower/kT)``.  When
    ``elements`` is an abundance mapping its values are interpreted as
    logarithmic number ratios; a plain iterable gives every element equal
    weight. ``ion_stage_weight`` can provide representative LTE ion fractions
    from an atmosphere.  This is important in warm polluted white dwarfs:
    ranking Fe I and Fe II as equally populated can fill a finite line budget
    with thousands of irrelevant Fe I transitions.  ``flux_weighted`` also
    multiplies the proxy by the Planck flux at each line wavelength.  It is
    intended for a finite atmospheric-blanketing budget, where blocking a
    photon near the SED peak matters more to radiative equilibrium than an
    otherwise identical far-infrared line; it is not used for dense formal
    spectra over a preselected wavelength interval.
    """

    if not np.isfinite(selection_temperature) or selection_temperature <= 0.0:
        raise ValueError("selection_temperature must be finite and positive")
    if isinstance(elements, Mapping):
        symbols = tuple(_canonical_element(element) for element in elements)
        abundance_weight = {
            _canonical_element(element): 10.0 ** float(abundance)
            for element, abundance in elements.items()
        }
    else:
        symbols = tuple(_canonical_element(element) for element in elements)
        abundance_weight = {
            _canonical_element(element): 1.0 for element in symbols
        }

    margin = max(100.0, 0.05 * (wavelength_maximum - wavelength_minimum))
    cache_key = (
        symbols,
        float(wavelength_minimum),
        float(wavelength_maximum),
        float(minimum_oscillator_strength),
    )
    candidates = atomic_database._line_selection_cache.get(cache_key)
    if candidates is None:
        mutable_candidates: list[
            tuple[AtomicIon, AtomicTransition, AtomicLevel]
        ] = []
        for element in symbols:
            for ion in atomic_database.ion_stages(element):
                level_lookup = {level.index: level for level in ion.levels}
                ground_energy = min(
                    (level.energy_wavenumber for level in ion.levels), default=0.0
                )
                for line in ion.transitions:
                    if (
                        line.transition_type != "E1"
                        or line.absorption_oscillator_strength
                        < minimum_oscillator_strength
                    ):
                        continue
                    center = line.wavelength_vacuum_angstrom
                    lower = level_lookup[line.lower_index]
                    # Strong ground-term resonance lines (Ca II H/K, Mg II h/k,
                    # Mg I 2852, Ca I 4227) keep optically thick wings hundreds
                    # of Angstroms away, so a narrow band must still include
                    # them.  Half the line wavelength bounds the detuning at
                    # which any impact or unified wing is meaningful.
                    line_margin = (
                        max(margin, 0.5 * center)
                        if line.absorption_oscillator_strength >= 0.1
                        and lower.energy_wavenumber - ground_energy <= 500.0
                        else margin
                    )
                    if (
                        wavelength_minimum - line_margin
                        <= center
                        <= wavelength_maximum + line_margin
                    ):
                        mutable_candidates.append((ion, line, lower))
        candidates = tuple(mutable_candidates)
        atomic_database._line_selection_cache[cache_key] = candidates
    thermal_energy = BOLTZMANN * selection_temperature

    def approximate_log_strength(
        item: tuple[AtomicIon, AtomicTransition, AtomicLevel],
    ) -> float:
        ion, line, lower = item
        log_strength = (
            np.log(abundance_weight[ion.element])
            + np.log(max(
                1.0 if ion_stage_weight is None else ion_stage_weight.get(
                    (ion.element, ion.charge), 0.0
                ),
                np.finfo(np.float64).tiny,
            ))
            + np.log(lower.statistical_weight)
            + np.log(line.absorption_oscillator_strength)
            - lower.energy_wavenumber * WAVENUMBER_TO_ERG / thermal_energy
        )
        if flux_weighted:
            wavelength_cm = line.wavelength_vacuum_angstrom * 1.0e-8
            exponent = (
                PLANCK * LIGHT_SPEED / (wavelength_cm * thermal_energy)
            )
            log_exponential_minus_one = (
                exponent + np.log1p(-np.exp(-exponent))
            )
            log_strength += (
                -5.0 * np.log(wavelength_cm)
                - log_exponential_minus_one
            )
        return log_strength

    ranked = sorted(candidates, key=approximate_log_strength, reverse=True)
    if maximum_lines is not None:
        ranked = ranked[:maximum_lines]
    return [(ion, line) for ion, line, _ in ranked]


_LJ_B_LOG_A = np.linspace(np.log(1.0e-7), np.log(1.0e3), 401)
_LJ_B_TABLE: FloatArray | None = None


def _lennard_jones_b_integral(a: float, maximum_phase: float = 4000.0) -> float:
    """Evaluate ``B(a)`` accurately in the phase variable ``u = x^-5``.

    ``B(a) = (1/5) int_0^inf u^(-7/5) [1 - cos(phi(u))] du`` with
    ``phi = a u^(11/5) - u`` (only ``|phi|`` matters for the cosine).  The
    integral is sampled uniformly in ``|phi|`` up to ``maximum_phase``, beyond
    which the cosine averages to zero and the tail is analytic.
    """

    u_grid = np.geomspace(1.0e-8, 1.0e6, 400_001)
    phase = a * u_grid ** 2.2 + u_grid  # |phi| grows monotonically for a >= 0
    end = int(np.searchsorted(phase, maximum_phase))
    u_end = float(u_grid[min(end, u_grid.size - 1)])
    targets = np.concatenate((
        np.geomspace(phase[0], 1.0, 4001)[:-1],
        np.arange(1.0, np.interp(u_end, u_grid, phase), 0.02),
    ))
    u = np.interp(targets, phase, u_grid)
    true_phase = a * u ** 2.2 - u
    body = trapezoid(u ** -1.4 * (1.0 - np.cos(true_phase)), u)
    tail = 2.5 * u[-1] ** -0.4
    return float((body + tail) / 5.0)


def lennard_jones_impact_broadening_integral(a: ArrayLike) -> FloatArray:
    r"""Return Hindmarsh, Petford & Smith's (1967) Lennard-Jones width integral.

    For the (12,6) phase shift ``eta = a x^-11 - x^-5`` along straight
    trajectories, ``B(a) = int_0^inf x [1 - cos(eta)] dx``.  ``B(0)`` recovers
    the classical pure van der Waals width, ``4 pi (3 pi/8)^(2/5) B(0) = 8.08``.
    (Hammond 1975 quotes the same integral with an extra factor 1/2.)
    The integral is tabulated once on a logarithmic grid in ``a`` and
    interpolated.
    """

    global _LJ_B_TABLE
    if _LJ_B_TABLE is None:
        _LJ_B_TABLE = np.asarray([
            _lennard_jones_b_integral(float(np.exp(log_a))) for log_a in _LJ_B_LOG_A
        ])
    value = np.asarray(a, dtype=np.float64)
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("Lennard-Jones parameter a must be finite and non-negative")
    if np.any(value > np.exp(_LJ_B_LOG_A[-1])):
        raise ValueError("Lennard-Jones parameter a is outside the tabulated range")
    log_value = np.log(np.maximum(value, np.exp(_LJ_B_LOG_A[0])))
    return np.interp(log_value, _LJ_B_LOG_A, _LJ_B_TABLE)


def mg_ii_helium_impact_rate_coefficient(
    wavelength_vacuum_angstrom: float,
    temperature_kelvin: ArrayLike,
) -> FloatArray:
    """Return the ab initio Mg II h/k damping rate per neutral-He atom.

    Allard, Guillon, Alekseev & Kielkopf (2016, A&A 593, A13), eqs. 11--12,
    give the impact half-widths from semi-classical unified theory with
    ab initio Mg+--He potentials: ``w = 0.055e-9 n_He T^0.4309`` (3p 2P1/2,
    2803 A) and ``w = 0.09e-9 n_He T^0.4031`` (3p 2P3/2, 2796 A) in rad/s,
    fitted over 400--12000 K.  These are 1.5--2.1 times the hydrogenic Unsold
    estimate, as Hammond (1975) measured for Ca II K.  The power law is
    continued above 12000 K.  The result is the angular-frequency FWHM, 2w,
    in cm3 s-1.
    """

    center = float(wavelength_vacuum_angstrom)
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    if 2794.0 < center < 2799.0:
        coefficient, exponent = 0.09e-9, 0.4031
    elif 2801.0 < center < 2806.0:
        coefficient, exponent = 0.055e-9, 0.4309
    else:
        raise ValueError("wavelength does not identify Mg II h or k")
    return 2.0 * coefficient * temperature**exponent


def ca_ii_helium_impact_rate_coefficient(
    wavelength_vacuum_angstrom: float,
    temperature_kelvin: ArrayLike,
) -> FloatArray:
    """Return the measured Ca II H/K damping rate per neutral-He atom.

    Hammond (1975) measured the Lorentz wavenumber HWHM of both Ca II
    resonance components in helium at 5200 K.  This converts those laboratory
    values to the angular-frequency damping-rate convention used by the Voigt
    opacity calculation.  The result has units cm3 s-1.  Its temperature
    dependence follows from impact theory with Hammond's own Lennard-Jones
    (12,6) fits to the measured widths and shifts, rather than from his
    two-point exponent, which he states is too uncertain to use.
    ``wavelength_vacuum_angstrom`` must identify either the 3934.8-A K line
    or the 3969.6-A H line.
    """

    center = float(wavelength_vacuum_angstrom)
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    if 3925.0 < center < 3950.0:
        component = "K"
    elif 3955.0 < center < 3985.0:
        component = "H"
    else:
        raise ValueError("wavelength does not identify Ca II H or K")
    coefficient_5200 = (
        4.0 * PI * LIGHT_SPEED
        * _CA_II_HE_WAVENUMBER_HWHM_PER_DENSITY_5200[component]
    )
    # Mean relative speed scales as sqrt(T) for the fixed Ca+--He pair.
    speed_ratio = np.sqrt(temperature / _CA_II_HE_REFERENCE_TEMPERATURE)
    reference_a = _CA_II_HE_LENNARD_JONES_A_5200[component]
    return coefficient_5200 * speed_ratio**0.6 * (
        lennard_jones_impact_broadening_integral(reference_a * speed_ratio**1.2)
        / lennard_jones_impact_broadening_integral(reference_a)
    )


def ca_ii_electron_stark_rate_coefficient(
    wavelength_vacuum_angstrom: float,
    temperature_kelvin: ArrayLike,
) -> FloatArray:
    """Return experimental Ca II electron damping for measured optical lines.

    The laboratory FWHM is converted to the angular-frequency damping-rate
    convention used by :func:`metal_line_mass_absorption_coefficient`.  The
    returned coefficient has units cm3 s-1 and is linear in electron density.
    """

    center = float(wavelength_vacuum_angstrom)
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    if 3153.0 < center < 3167.0:
        component = "3159"
    elif 3173.0 < center < 3187.0:
        component = "3180"
    elif 3700.0 < center < 3715.0:
        component = "3707"
    elif 3730.0 < center < 3745.0:
        component = "3738"
    elif 3925.0 < center < 3950.0:
        component = "K"
    elif 3955.0 < center < 3985.0:
        component = "H"
    else:
        raise ValueError("wavelength does not identify a measured Ca II line")
    fwhm_per_density = (
        _CA_II_ELECTRON_STARK_FWHM_14000[component]
        / _CA_II_ELECTRON_STARK_REFERENCE_DENSITY
        * (_CA_II_ELECTRON_STARK_REFERENCE_TEMPERATURE / temperature) ** 0.5
    )
    center_cm = center * 1.0e-8
    # FWHM=2*HWHM and HWHM_A=lambda_cm^2*Gamma/(4*pi*c)*1e8.
    return (
        2.0 * PI * LIGHT_SPEED * fwhm_per_density * 1.0e-8 / center_cm**2
    )


def mg_ii_4481_electron_stark_rate_coefficient(
    wavelength_vacuum_angstrom: float,
    temperature_kelvin: ArrayLike,
) -> FloatArray:
    """Return the theoretical Mg II 4481 electron damping per electron."""

    center = float(wavelength_vacuum_angstrom)
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    if not 4478.0 < center < 4487.0:
        raise ValueError("wavelength does not identify Mg II 4481")
    fwhm_per_density = (
        _MG_II_4481_ELECTRON_STARK_FWHM
        / _MG_II_4481_ELECTRON_STARK_REFERENCE_DENSITY
        * (_MG_II_4481_ELECTRON_STARK_REFERENCE_TEMPERATURE / temperature) ** 0.5
    )
    center_cm = center * 1.0e-8
    return (
        2.0 * PI * LIGHT_SPEED * fwhm_per_density * 1.0e-8 / center_cm**2
    )


_MG_II_KURUCZ_WIDTHS: dict[tuple[str, str], float] | None = None


def _mg_ii_kurucz_widths() -> dict[tuple[str, str], float]:
    global _MG_II_KURUCZ_WIDTHS
    if _MG_II_KURUCZ_WIDTHS is None:
        table: dict[tuple[str, str], float] = {}
        resource = files("wd_spectra").joinpath(
            "data/stark/kurucz_mg_ii_electron_widths.csv"
        )
        with resource.open("r", encoding="ascii") as stream:
            for raw in stream:
                if raw.startswith("#") or raw.startswith("lower,"):
                    continue
                lower, upper, value, _ = raw.strip().split(",")
                table[(lower, upper)] = float(value)
        _MG_II_KURUCZ_WIDTHS = table
    return _MG_II_KURUCZ_WIDTHS


def mg_ii_kurucz_electron_stark_rate_coefficient(
    lower_level: AtomicLevel,
    upper_level: AtomicLevel,
    temperature_kelvin: ArrayLike,
) -> FloatArray | None:
    """Kurucz's electron damping per electron for a Mg II 2p6-core transition.

    ``gf1201.all`` gives ``log10(Gamma_e/ne)`` (angular-frequency FWHM,
    10,000 K) for every Mg II multiplet; the bundled table groups it by the
    outer orbitals of both levels (components agree to 0.04 dex).  Using it for
    the whole series replaces a capped generic estimate that was up to ~10^3
    times smaller than Kurucz for the Rydberg members, while one member
    (4f-8g, 4852 A) already used Kurucz, making the series inconsistent.
    The quadratic-Stark impact dependence ``T**(1/6)`` (Lindholm--Foley;
    the convention of Kurucz's SYNTHE, MOOG and SME) is applied.  ``None`` for levels
    outside the table or not built on the ground 2p6 core.
    """

    orbitals = []
    for level in (lower_level, upper_level):
        parts = level.label.split(".")
        if len(parts) != 3 or parts[0] != "2p6":
            return None
        orbitals.append(parts[1])
    value = _mg_ii_kurucz_widths().get((orbitals[0], orbitals[1]))
    if value is None:
        return None
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    return 10.0**value * (temperature / 10_000.0) ** (1.0 / 6.0)


def mg_ii_4852_electron_stark_rate_coefficient(
    wavelength_vacuum_angstrom: float,
    temperature_kelvin: ArrayLike,
) -> FloatArray:
    """Return Kurucz's Mg II 4f--8g 4852-A damping per electron.

    Kurucz tabulates ``Gamma_e/ne`` in the angular-frequency convention used
    by the formal solver.  As for line-specific values read by
    :func:`read_kurucz_gf100_atomic_database`, the standard weak
    ``T**(1/6)`` quadratic-Stark impact dependence is applied away from the
    10,000-K reference.
    """

    center = float(wavelength_vacuum_angstrom)
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    if not 4848.0 < center < 4857.0:
        raise ValueError("wavelength does not identify Mg II 4852")
    return _MG_II_4852_ELECTRON_STARK_RATE_COEFFICIENT_10000 * (
        temperature / _MG_II_4852_ELECTRON_STARK_REFERENCE_TEMPERATURE
    ) ** (1.0 / 6.0)


def mg_i_3835_electron_stark_rate_coefficient(
    wavelength_vacuum_angstrom: float,
    temperature_kelvin: ArrayLike,
) -> FloatArray:
    """Return the measured Mg I 3835-multiplet damping per electron."""

    center = float(wavelength_vacuum_angstrom)
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    if not 3825.0 < center < 3845.0:
        raise ValueError("wavelength does not identify the Mg I 3835 multiplet")
    fwhm_per_density = (
        _MG_I_3835_ELECTRON_STARK_FWHM
        / _MG_I_3835_ELECTRON_STARK_REFERENCE_DENSITY
        * (_MG_I_3835_ELECTRON_STARK_REFERENCE_TEMPERATURE / temperature) ** 0.5
    )
    center_cm = center * 1.0e-8
    return (
        2.0 * PI * LIGHT_SPEED * fwhm_per_density * 1.0e-8 / center_cm**2
    )


def mg_i_optical_electron_stark_rate_coefficient(
    wavelength_vacuum_angstrom: float,
    temperature_kelvin: ArrayLike,
) -> FloatArray | None:
    """Return published electron damping for Mg I 4704 or the b triplet."""

    center = float(wavelength_vacuum_angstrom)
    if 4695.0 < center < 4715.0:
        multiplet = "3p1P-5d1D"
    elif 5160.0 < center < 5192.0:
        multiplet = "3p3P-4s3S"
    else:
        return None
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    fwhm = np.exp(np.interp(
        np.log(temperature),
        np.log(_MG_I_OPTICAL_STARK_TEMPERATURE_K),
        np.log(_MG_I_OPTICAL_STARK_FWHM_ANGSTROM[multiplet]),
    ))
    fwhm_per_density = fwhm / _MG_I_OPTICAL_STARK_REFERENCE_DENSITY
    center_cm = center * 1.0e-8
    return (
        2.0 * PI * LIGHT_SPEED * fwhm_per_density * 1.0e-8 / center_cm**2
    )


def na_i_optical_electron_stark_rate_coefficient(
    wavelength_vacuum_angstrom: float,
    temperature_kelvin: ArrayLike,
) -> FloatArray | None:
    """Return published Na I damping for 5684/5690 or the D doublet.

    The tabulated widths are multiplet averages.  Interpolation is log-linear
    in temperature and is held at the nearest tabulated endpoint outside the
    2500--80000 K range.
    """

    center = float(wavelength_vacuum_angstrom)
    if 5678.0 < center < 5696.0:
        multiplet = "3p-4d"
    elif 5885.0 < center < 5903.0:
        multiplet = "3s-3p"
    else:
        return None
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    fwhm = np.exp(np.interp(
        np.log(temperature),
        np.log(_NA_I_OPTICAL_STARK_TEMPERATURE_K),
        np.log(_NA_I_OPTICAL_STARK_FWHM_ANGSTROM[multiplet]),
    ))
    fwhm_per_density = fwhm / _NA_I_OPTICAL_STARK_REFERENCE_DENSITY
    center_cm = center * 1.0e-8
    return (
        2.0 * PI * LIGHT_SPEED * fwhm_per_density * 1.0e-8 / center_cm**2
    )


def na_i_d_neon_impact_rate_coefficient(
    wavelength_vacuum_angstrom: float,
    temperature_kelvin: ArrayLike,
) -> FloatArray | None:
    """Return the measured neutral-Ne impact rate for the Na I D doublet.

    The laboratory result constrains the impact core.  It intentionally does
    not claim to supply the non-Lorentzian Na--Ne far wings, which require a
    unified profile calculated from the molecular difference potentials.
    """

    center = float(wavelength_vacuum_angstrom)
    if 5888.0 < center < 5894.5:
        component = "D2"
    elif 5894.5 <= center < 5901.0:
        component = "D1"
    else:
        return None
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    return (
        _NA_I_D_NEON_DAMPING_RATE_CM3_S[component]
        * (temperature / _NA_I_D_NEON_REFERENCE_TEMPERATURE_K) ** 0.3
    )


def classical_electron_stark_rate_coefficient(
    ion: AtomicIon,
    transition: AtomicTransition,
) -> float:
    """Return SYNSPEC's classical electron damping rate per electron.

    In the absence of tabulated Stark widths, Hubeny & Lanz use
    ``Gamma_e / n_e = 1e-8 n_eff^5`` with the effective principal quantum
    number inferred from the upper-level binding energy and capped at five.
    The coefficient is in cm3 s-1.  It is only a fallback, but omitting it
    altogether makes saturated metal lines much too narrow in hydrogen- and
    helium-free C/O atmospheres.
    """

    if ion.ionization_energy_ev is None:
        return 0.0
    levels = {level.index: level for level in ion.levels}
    upper = levels.get(transition.upper_index)
    if upper is None:
        return 0.0
    binding_energy_ev = (
        ion.ionization_energy_ev
        - upper.energy_wavenumber / EV_TO_WAVENUMBER
    )
    if binding_energy_ev <= 0.0:
        return 0.0
    effective_n_squared = np.clip(
        (ion.charge + 1.0) ** 2 * 13.595 / binding_energy_ev,
        0.0,
        25.0,
    )
    return float(1.0e-8 * effective_n_squared**2.5)


_RYDBERG_FREQUENCY_HZ = 3.289_841_960_250e15
_ORBITAL_LETTERS = "spdfghik"
_PARENT_TERM_PATTERN = re.compile(r"\(([^()<]+)(?:<[^>]*>)?\)")


def _rydberg_orbital(label: str) -> tuple[int, int, str] | None:
    parts = label.split(".")
    if len(parts) < 2:
        return None
    orbital = re.fullmatch(r"(\d+)([spdfghik])", parts[-2])
    if orbital is None:
        return None
    return (
        int(orbital.group(1)),
        _ORBITAL_LETTERS.index(orbital.group(2)),
        ".".join(parts[:-2]),
    )


def _level_quantum_defect(ion: AtomicIon, level: AtomicLevel, principal: int) -> float | None:
    binding = ion.ionization_energy_ev - level.energy_wavenumber / EV_TO_WAVENUMBER
    if binding <= 0.0:
        return None
    return principal - (ion.charge + 1.0) * float(np.sqrt(_RYDBERG_ENERGY_EV / binding))


_LINEAR_STARK_LEVEL_CACHE: dict[tuple[int, int, int], object] = {}


def linear_stark_rydberg_level(
    atomic_database: AtomicDatabase,
    ion: AtomicIon,
    level: AtomicLevel,
) -> tuple[int, int, float, float] | None:
    """Cached :func:`_linear_stark_rydberg_level` (static atomic data only)."""

    key = (id(atomic_database), id(ion), level.index)
    if key not in _LINEAR_STARK_LEVEL_CACHE:
        _LINEAR_STARK_LEVEL_CACHE[key] = _linear_stark_rydberg_level(
            atomic_database, ion, level
        )
    return _LINEAR_STARK_LEVEL_CACHE[key]


def _linear_stark_rydberg_level(
    atomic_database: AtomicDatabase,
    ion: AtomicIon,
    level: AtomicLevel,
) -> tuple[int, int, float, float] | None:
    """Return ``(n, l, n_eff, defect gap)`` for a genuine Rydberg level.

    Linear (hydrogenic) Stark behaviour requires a nearly degenerate
    ``n``-manifold: the outer electron must be in an ``l >= 2`` orbital
    converging on the ground term of the next ion, with a small positive
    quantum defect.  The returned defect gap is the smallest defect difference
    to an ``l +/- 1`` neighbour of the same ``n``.  Core-excited levels and
    series converging on excited parent terms are excluded; their binding energy relative to the ground
    ionization limit does not measure their Rydberg character.
    """

    if ion.ionization_energy_ev is None:
        return None
    parsed = _rydberg_orbital(level.label)
    if parsed is None:
        return None
    principal, angular, core = parsed
    if angular < 2:
        return None
    parents = _PARENT_TERM_PATTERN.findall(core)
    next_ion = atomic_database.ions.get((ion.element, ion.charge + 1))
    if parents and next_ion is not None and next_ion.levels:
        ground = min(next_ion.levels, key=lambda item: item.energy_wavenumber)
        ground_terms = _PARENT_TERM_PATTERN.findall(ground.label)
        if not ground_terms or ground_terms[-1] != parents[-1]:
            return None
    binding = ion.ionization_energy_ev - level.energy_wavenumber / EV_TO_WAVENUMBER
    if binding <= 0.0:
        return None
    core_charge = ion.charge + 1.0
    effective_n = core_charge * float(np.sqrt(_RYDBERG_ENERGY_EV / binding))
    defect = principal - effective_n
    # High-l defects are ~0 and tabulated energies can make them marginally
    # negative; a clearly negative defect instead signals a wrong parent.
    if not -0.05 < defect < 0.5:
        return None
    # The Stark coupling mixes the level with its l +/- 1 neighbours of the
    # same n and core.  The relevant gap is the smallest defect difference to
    # a neighbour that exists: a tabulated level, or an allowed but untabulated
    # l + 1 <= n - 1 level, whose high-l defect is taken as zero.
    neighbour_defects = [
        value
        for other in ion.levels
        if (parsed_other := _rydberg_orbital(other.label)) is not None
        and parsed_other[0] == principal
        and parsed_other[2] == core
        and abs(parsed_other[1] - angular) == 1
        and (value := _level_quantum_defect(ion, other, principal)) is not None
    ]
    has_upper = any(
        (parsed_other := _rydberg_orbital(other.label)) is not None
        and parsed_other[0] == principal and parsed_other[2] == core
        and parsed_other[1] == angular + 1
        for other in ion.levels
    )
    if angular + 1 <= principal - 1 and not has_upper:
        neighbour_defects.append(0.0)
    if not neighbour_defects:
        return None
    gap_defect = min(abs(defect - value) for value in neighbour_defects)
    return principal, angular, effective_n, gap_defect


def linear_stark_mixing_fraction(
    effective_n: float,
    quantum_defect: float,
    core_charge: float,
    ionic_microfield_scale: ArrayLike,
) -> FloatArray:
    """Depth-dependent fraction of a Rydberg level in the linear-Stark regime.

    ``S`` is the hydrogenic linear-Stark extent ``(3/2) n (n-1) e a0 F0 / Z``
    in the Holtsmark normal field of the ionic perturbers and ``D`` the
    quantum-defect separation ``2 Ry Z^2 delta / n^3`` from the neighbouring
    ``l`` manifold.  The two-level mixing fraction ``1 - D / sqrt(D^2 + S^2)``
    is zero for an isolated level and one for a fully mixed manifold.  It
    contains no adjustable parameter and is smooth in temperature and density.
    """

    splitting = (
        effective_n * (effective_n - 1.0)
        * np.asarray(ionic_microfield_scale, dtype=np.float64)
        / 1.385 / core_charge
    )
    gap = 2.0 * _RYDBERG_FREQUENCY_HZ * core_charge**2 * quantum_defect / effective_n**3
    if gap <= 0.0:
        return np.ones_like(splitting)
    return 1.0 - gap / np.hypot(gap, splitting)


# Stark coupling e a0 F / h (Hz) on which manifold eigen-systems are tabulated.
_STARK_MANIFOLD_FIELD_HZ = np.geomspace(1.0e4, 1.0e16, 12 * 48 + 1)
_STARK_MANIFOLD_CACHE: dict[tuple[int, int, int], object] = {}
_TERM_MULTIPLICITY_PATTERN = re.compile(r"\((\d+)[A-Z]")
# e a0 F0 / h per unit ``ionic_microfield_scale`` (Holtsmark normal field
# F0 = 2.603 e N^(2/3)); consistent with the 1.385 used by the mixing fraction.
_STARK_COUPLING_PER_MICROFIELD_SCALE = 1.0 / (1.385 * 1.5)
# Fields per profile evaluation (every tabulated field; coarser sampling
# changes saturated J1109 lines by ~1 per cent).
_STARK_PROFILE_FIELD_STRIDE = 1
# Uniform-grid sizes of the fine (dense pattern) and coarse (sparse tail)
# convolutions in :func:`manifold_quasistatic_line_profile`.
_STARK_FINE_BINS = 16_384
# Fine bins of a quarter impact width reproduce 1/16-width bins to 3e-4 rms
# in the J1109 spectrum.
_STARK_FINE_STEP_PER_WIDTH = 0.25
_STARK_COARSE_BINS = 4_096


@dataclass(frozen=True)
class RydbergStarkManifold:
    """Stark eigen-system of one ``n``-manifold, projected on one ``l`` state.

    ``shifts_hz[j, i]`` and ``weights[j, i]`` are the energies (relative to
    the zero-field absorbing level) and the squared projections of the
    absorbing ``(n, l)`` state, averaged over ``m``, of eigenstate ``i`` at
    coupling ``_STARK_MANIFOLD_FIELD_HZ[j]``.  Each row of weights sums to one.
    """

    principal: int
    angular: int
    l_values: tuple[int, ...]
    shifts_hz: FloatArray
    weights: FloatArray
    # Strength-weighted mean square shift at each tabulated field (Hz^2).
    mean_square_shift_hz2: FloatArray
    # Largest |shift| of any component with weight > 1e-4 at each field (Hz).
    maximum_shift_hz: FloatArray


def rydberg_stark_manifold(
    atomic_database: AtomicDatabase,
    ion: AtomicIon,
    level: AtomicLevel,
) -> RydbergStarkManifold | None:
    """Cached :func:`_rydberg_stark_manifold` (static atomic data only)."""

    key = (id(atomic_database), id(ion), level.index)
    if key not in _STARK_MANIFOLD_CACHE:
        _STARK_MANIFOLD_CACHE[key] = _rydberg_stark_manifold(
            atomic_database, ion, level
        )
    return _STARK_MANIFOLD_CACHE[key]


def _rydberg_stark_manifold(
    atomic_database: AtomicDatabase,
    ion: AtomicIon,
    level: AtomicLevel,
) -> RydbergStarkManifold | None:
    """Diagonalize the ionic-field Stark problem in the level's ``n``-manifold.

    The zero-field energies are the tabulated term energies (averaged over
    ``J``) of every ``l`` of the same ``n``, parent core and spin
    multiplicity; allowed ``l`` above the highest tabulated one are given the
    hydrogenic energy.  Untabulated ``l`` below it are omitted: they carry
    large quantum defects and couple weakly.  The field couples ``l`` and
    ``l + 1`` with the hydrogenic intra-shell matrix element
    ``(3/2) n sqrt((n^2 - l'^2)(l'^2 - m^2) / (4 l'^2 - 1)) e a0 F / Z``.
    A fully degenerate manifold gives the parabolic ``(3/2) n k`` pattern; a
    manifold split by quantum defects gives the partial, field-dependent
    mixing directly, with no two-level approximation and no free parameter.
    Levels that are not near-hydrogenic Rydberg levels return ``None``.
    """

    if linear_stark_rydberg_level(atomic_database, ion, level) is None:
        return None
    parsed = _rydberg_orbital(level.label)
    assert parsed is not None
    principal, angular, core = parsed
    multiplicity = _TERM_MULTIPLICITY_PATTERN.findall(level.label.split(".")[-1])
    energy_sum: dict[int, float] = {}
    weight_sum: dict[int, float] = {}
    for other in ion.levels:
        parsed_other = _rydberg_orbital(other.label)
        if parsed_other is None or parsed_other[0] != principal or parsed_other[2] != core:
            continue
        if _TERM_MULTIPLICITY_PATTERN.findall(other.label.split(".")[-1]) != multiplicity:
            continue
        l_other = parsed_other[1]
        energy_sum[l_other] = (
            energy_sum.get(l_other, 0.0) + other.statistical_weight * other.energy_wavenumber
        )
        weight_sum[l_other] = weight_sum.get(l_other, 0.0) + other.statistical_weight
    if angular not in energy_sum:
        return None
    core_charge = ion.charge + 1.0
    hydrogenic = (
        ion.ionization_energy_ev * EV_TO_WAVENUMBER
        - _RYDBERG_FREQUENCY_HZ / LIGHT_SPEED * core_charge**2 / principal**2
    )
    energies = {l_value: energy_sum[l_value] / weight_sum[l_value] for l_value in energy_sum}
    for l_value in range(max(energies) + 1, principal):
        energies[l_value] = hydrogenic
    l_values = tuple(sorted(energies))
    reference = energies[angular]
    field = _STARK_MANIFOLD_FIELD_HZ
    shifts: list[FloatArray] = []
    weights: list[FloatArray] = []
    for m in range(angular + 1):
        block = [l_value for l_value in l_values if l_value >= m]
        size = len(block)
        diagonal = np.array(
            [(energies[l_value] - reference) * LIGHT_SPEED for l_value in block]
        )
        coupling = np.zeros((size, size))
        for index in range(size - 1):
            lower_l, upper_l = block[index], block[index + 1]
            if upper_l != lower_l + 1:
                continue
            coupling[index, index + 1] = coupling[index + 1, index] = (
                1.5 * principal
                * np.sqrt(
                    (principal**2 - upper_l**2) * (upper_l**2 - m**2)
                    / (4.0 * upper_l**2 - 1.0)
                )
                / core_charge
            )
        hamiltonian = (
            np.diag(diagonal)[np.newaxis, :, :]
            + field[:, np.newaxis, np.newaxis] * coupling[np.newaxis, :, :]
        )
        values, vectors = np.linalg.eigh(hamiltonian)
        projection = vectors[:, block.index(angular), :] ** 2
        multiplicity_m = 1.0 if m == 0 else 2.0
        shifts.append(values)
        weights.append(projection * multiplicity_m / (2.0 * angular + 1.0))
    shift_array = np.concatenate(shifts, axis=1)
    weight_array = np.concatenate(weights, axis=1)
    keep = np.max(weight_array, axis=0) > 1.0e-8
    shift_array = np.ascontiguousarray(shift_array[:, keep])
    weight_array = np.ascontiguousarray(weight_array[:, keep])
    return RydbergStarkManifold(
        principal,
        angular,
        l_values,
        shift_array,
        weight_array,
        np.sum(weight_array * shift_array**2, axis=1),
        np.max(
            np.where(weight_array > 1.0e-4, np.abs(shift_array), 0.0), axis=1
        ),
    )


def _deposit_segments(
    start: FloatArray, stop: FloatArray, weight: FloatArray, size: int
) -> FloatArray:
    """Bin masses of uniform boxes ``[start, stop]`` (fractional bin units).

    Each box's cumulative distribution is a ramp; its second derivative is a
    pair of deltas, deposited with linear weights and integrated twice.  The
    result is exact at the bin edges, so eigenvalue tracks sampled at
    discrete fields give a smooth density rather than a comb.
    """

    low = np.minimum(start, stop)
    high = np.maximum(start, stop)
    high = np.maximum(high, low + 1.0e-3)
    slope = weight / (high - low)
    # Mass outside the window is dropped; a box crossing an edge keeps the
    # part inside (same slope, clipped extent).
    low = np.maximum(low, 0.0)
    high = np.minimum(high, float(size))
    inside = high > low
    low, high, slope = low[inside], high[inside], slope[inside]
    second = np.zeros(size + 3)
    for position, sign in ((low, 1.0), (high, -1.0)):
        clipped = position
        floor = np.floor(clipped)
        fraction = clipped - floor
        index = floor.astype(np.int64)
        second += np.bincount(index, sign * slope * (1.0 - fraction), minlength=size + 3)[: size + 3]
        second += np.bincount(index + 1, sign * slope * fraction, minlength=size + 3)[: size + 3]
    cumulative = np.concatenate(([0.0], np.cumsum(np.cumsum(second))))
    return np.diff(cumulative[: size + 1])


def _stark_impact_profile(wavelength, center, gaussian_sigma, lorentz_hwhm):
    """Impact core for the manifold path: the exact frequency-space Voigt.

    The native pseudo-Voigt kernel is not used because it drops the
    ``(lambda/lambda0)^2`` asymmetry of the far impact wings.
    """
    return _voigt_profile_per_angstrom(
        wavelength, center, gaussian_sigma, lorentz_hwhm
    )


def _compiled_manifold_profile(
    wavelength, center, gaussian_sigma, lorentz_hwhm, coupling_hz,
    upper, lower, maximum_beta, support_half_width, segment_probability,
    interval_fraction, missing, bound_probability,
):
    """The reference profile algorithm with fused C component deposition.

    No atomic, field, grid, or convolution approximation is changed.  Keep
    the NumPy path below as the fallback and an independent numerical check.
    """
    from scipy.signal import fftconvolve

    to_angstrom = -center**2 / (LIGHT_SPEED * 1.0e8)
    width = max(gaussian_sigma, lorentz_hwhm, 1.0e-5)
    half_extent = (
        float(support_half_width) if support_half_width is not None
        else float(np.max(np.abs(wavelength - center)))
    ) + 1.0e-6
    dense_field = min(21.0, maximum_beta) * coupling_hz
    log_field = np.log(max(dense_field, _STARK_MANIFOLD_FIELD_HZ[0]))
    dense_shift = float(np.interp(
        log_field, np.log(_STARK_MANIFOLD_FIELD_HZ), upper.maximum_shift_hz
    ))
    if lower is not None:
        dense_shift += float(np.interp(
            log_field, np.log(_STARK_MANIFOLD_FIELD_HZ), lower.maximum_shift_hz
        ))
    fine_half = min(half_extent, abs(to_angstrom) * dense_shift + 64.0 * width)
    fine_step = max(
        _STARK_FINE_STEP_PER_WIDTH * width, 2.0 * fine_half / _STARK_FINE_BINS
    )
    coarse_step = max(fine_step, 2.0 * half_extent / _STARK_COARSE_BINS)
    half_bins = int(np.ceil(fine_half / fine_step))
    fine_grid = fine_step * (np.arange(2 * half_bins + 1) - half_bins)
    fine_mass = np.empty(fine_grid.size)
    coarse_size = 0
    if half_extent > fine_half:
        half_bins = int(np.ceil(half_extent / coarse_step))
        coarse_size = 2 * half_bins + 1
    coarse_mass = np.empty(coarse_size)
    stride = _STARK_PROFILE_FIELD_STRIDE
    core_weight, inner_mass = _rt.stark_manifold_bins(
        np.ascontiguousarray(upper.shifts_hz[::stride]),
        np.ascontiguousarray(upper.weights[::stride]),
        None if lower is None else np.ascontiguousarray(lower.shifts_hz[::stride]),
        None if lower is None else np.ascontiguousarray(lower.weights[::stride]),
        segment_probability, interval_fraction, to_angstrom, fine_half,
        fine_step, coarse_step, missing, fine_mass, coarse_mass,
    )
    kernel = _stark_impact_profile(
        center + fine_grid, center, gaussian_sigma, lorentz_hwhm
    )
    kernel_sum = float(np.sum(kernel))
    if kernel_sum > 0.0 and np.isfinite(kernel_sum):
        fine_mass = fftconvolve(fine_mass, kernel / kernel_sum, mode="same")
    # The native stark_profile_finish evaluates a pseudo-Voigt impact core,
    # so the exact frequency-space Voigt core and wings are added here.
    fine_profile = np.maximum(fine_mass, 0.0) / fine_step
    offset = wavelength - center
    result = core_weight * _stark_impact_profile(
        wavelength, center, gaussian_sigma, lorentz_hwhm
    )
    inside = np.abs(offset) <= fine_half
    result[inside] += np.interp(offset[inside], fine_grid, fine_profile)
    if coarse_size:
        coarse_grid = coarse_step * (np.arange(coarse_size) - coarse_size // 2)
        outside = ~inside
        result[outside] += (
            np.interp(offset[outside], coarse_grid, coarse_mass / coarse_step)
            + inner_mass * _stark_impact_profile(
                wavelength[outside], center, gaussian_sigma, lorentz_hwhm
            )
        )
    return result / max(bound_probability, np.finfo(np.float64).tiny)


@lru_cache(maxsize=512)
def _stark_field_probabilities(
    coupling_hz: float, maximum_beta: float, correlation: float,
    radiator_core_charge: float, stride: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Reuse identical depth/upper-level field distributions across lines.

    Keys contain the complete physical state used here, with no rounding or
    interpolation.  The bounded cache holds at most about 9 MiB on the
    default field grid; a new atmospheric state cannot reuse stale values.
    """
    field = _STARK_MANIFOLD_FIELD_HZ[::stride]
    beta = field / max(coupling_hz, np.finfo(np.float64).tiny)
    cumulative = hooper_microfield_cumulative_probability(
        np.minimum(beta, maximum_beta), correlation, radiator_core_charge
    )
    probability = np.diff(cumulative)
    # The segment crossing the critical field stops at that field.
    fraction = np.clip(
        (maximum_beta - beta[:-1]) / (beta[1:] - beta[:-1]), 0.0, 1.0
    )
    for value in (beta, cumulative, probability, fraction):
        value.flags.writeable = False
    return beta, cumulative, probability, fraction


def manifold_quasistatic_line_profile(
    wavelength: FloatArray,
    center: float,
    gaussian_sigma: float,
    lorentz_hwhm: float,
    coupling_hz: float,
    upper: RydbergStarkManifold,
    lower: RydbergStarkManifold | None,
    maximum_beta: float = np.inf,
    correlation: float = 0.0,
    radiator_core_charge: float = 1.0,
    support_half_width: float | None = None,
) -> FloatArray:
    """Impact profile convolved with the ionic quasi-static manifold pattern.

    ``coupling_hz`` is ``e a0 F0 / h`` for the Holtsmark normal field.  The
    Stark components of the upper (and, if itself mixed, the lower) manifold
    are integrated over the Holtsmark field distribution, each eigenvalue
    track being spread linearly between tabulated fields, and the resulting
    quasi-static distribution is convolved with the electron-impact Voigt
    profile.  Only microfields below ``maximum_beta`` (the Hummer--Mihalas
    critical field of the upper level, in normal-field units) are integrated:
    stronger fields dissolve the level, and that strength is represented by
    the Q-MHD occupation probability and pseudo-continuum instead.  Returns
    the bound-state (conditional) profile per angstrom, normalized to unit
    area over all wavelengths independently of the sampled ``wavelength``.
    """

    from scipy.signal import fftconvolve

    stride = _STARK_PROFILE_FIELD_STRIDE
    field = _STARK_MANIFOLD_FIELD_HZ[::stride]
    # Field probabilities from the same (Hooper-correlated) distribution as
    # the Q-MHD occupation probabilities, integrated exactly between the
    # tabulated fields; fields above the critical one dissolve the level.
    beta, cumulative, segment_probability, interval_fraction = _stark_field_probabilities(
        float(coupling_hz), float(maximum_beta), float(correlation),
        float(radiator_core_charge), stride,
    )
    to_angstrom = -center**2 / (LIGHT_SPEED * 1.0e8)
    width = max(gaussian_sigma, lorentz_hwhm, 1.0e-5)

    def shifted_fraction(manifold: RydbergStarkManifold) -> float:
        # Holtsmark-weighted strength shifted by more than a quarter width.
        weights = manifold.weights[::stride]
        shifted = np.abs(manifold.shifts_hz[::stride] * to_angstrom) > 0.25 * width
        per_field = np.sum(np.where(shifted, weights, 0.0), axis=1)
        total = float(np.sum(segment_probability))
        if total <= 0.0:
            return 0.0
        return float(
            np.sum(segment_probability * 0.5 * (per_field[1:] + per_field[:-1]))
        ) / total

    # A pattern that leaves < 1e-3 of the strength outside a quarter of the
    # impact width does not change the profile at that level.
    if lower is not None and shifted_fraction(lower) < 1.0e-3:
        lower = None
    used = segment_probability > 1.0e-12
    if not np.any(used) or (
        lower is None and shifted_fraction(upper) < 1.0e-3
    ):
        return _voigt_profile_per_angstrom(
            wavelength, center, gaussian_sigma, lorentz_hwhm
        )
    # Only the weak-field probability below the tabulated fields is missing;
    # it is unshifted.
    missing = float(cumulative[0])
    bound_probability = missing + float(np.sum(segment_probability[used]))
    if getattr(_rt, "stark_manifold_bins", None) is not None:
        return _compiled_manifold_profile(
            wavelength, center, gaussian_sigma, lorentz_hwhm, coupling_hz,
            upper, lower, maximum_beta, support_half_width, segment_probability,
            interval_fraction, missing, bound_probability,
        )
    upper_shift = upper.shifts_hz[::stride]
    upper_weight = upper.weights[::stride]
    if lower is None:
        shift = upper_shift[:, :, np.newaxis]
        weight = upper_weight[:, :, np.newaxis]
    else:
        shift = upper_shift[:, :, np.newaxis] - lower.shifts_hz[::stride, np.newaxis, :]
        weight = upper_weight[:, :, np.newaxis] * lower.weights[::stride, np.newaxis, :]
    index = np.flatnonzero(used)
    start_shift = shift[index].reshape(index.size, -1)
    stop_shift = start_shift + interval_fraction[index, np.newaxis] * (
        shift[index + 1].reshape(index.size, -1) - start_shift
    )
    segment_weight = (
        0.5 * (weight[index] + weight[index + 1]).reshape(index.size, -1)
        * segment_probability[index, np.newaxis]
    )
    start_offset = (start_shift * to_angstrom).ravel()
    stop_offset = (stop_shift * to_angstrom).ravel()
    segment_weight = segment_weight.ravel()
    significant = segment_weight > 1.0e-7 * float(np.max(segment_weight))
    start_offset = start_offset[significant]
    stop_offset = stop_offset[significant]
    segment_weight = segment_weight[significant]

    # Two uniform grids: a fine one (a fraction of the impact width) over the
    # dense part of the pattern, where it is convolved with the impact Voigt
    # profile, and a coarse one over the sparse tail, whose bins are many
    # impact widths wide and receive the pattern without convolution.
    # Internal grids depend on the line's support, never on which
    # wavelengths are requested, so any request samples the same profile.
    half_extent = (
        float(support_half_width)
        if support_half_width is not None
        else float(np.max(np.abs(wavelength - center)))
    ) + 1.0e-6
    # The fine grid spans the pattern out to the largest shift at the field
    # below which 99 per cent of Holtsmark fields lie (beta ~ 21) or at the
    # critical field if smaller; only the sparse tail uses the coarse grid.
    dense_field = min(21.0, maximum_beta) * coupling_hz
    dense_shift = float(np.interp(
        np.log(max(dense_field, field[0])), np.log(_STARK_MANIFOLD_FIELD_HZ),
        upper.maximum_shift_hz,
    ))
    if lower is not None:
        dense_shift += float(np.interp(
            np.log(max(dense_field, field[0])), np.log(_STARK_MANIFOLD_FIELD_HZ),
            lower.maximum_shift_hz,
        ))
    fine_half = min(
        half_extent, abs(to_angstrom) * dense_shift + 64.0 * width
    )
    fine_step = max(
        _STARK_FINE_STEP_PER_WIDTH * width, 2.0 * fine_half / _STARK_FINE_BINS
    )
    coarse_step = max(fine_step, 2.0 * half_extent / _STARK_COARSE_BINS)
    # Components within half a fine bin of the centre keep the exact core.
    core = 0.5 * fine_step
    unshifted = (np.abs(start_offset) <= core) & (np.abs(stop_offset) <= core)
    core_weight = missing + float(np.sum(segment_weight[unshifted]))
    low = np.minimum(start_offset[~unshifted], stop_offset[~unshifted])
    high = np.maximum(start_offset[~unshifted], stop_offset[~unshifted])
    high = np.maximum(high, low + 1.0e-6 * fine_step)
    mass = segment_weight[~unshifted]
    # Split each box at the fine-range boundaries (mass proportional to length).
    inner_low = np.clip(low, -fine_half, fine_half)
    inner_high = np.clip(high, -fine_half, fine_half)
    inner_mass = mass * (inner_high - inner_low) / (high - low)

    def convolved(grid_step, grid_half, box_low, box_high, box_mass):
        # An odd grid centred on the line keeps the kernel exactly centred.
        half_bins = int(np.ceil(grid_half / grid_step))
        grid = grid_step * (np.arange(2 * half_bins + 1) - half_bins)
        origin = grid[0] - 0.5 * grid_step
        masses = _deposit_segments(
            (box_low - origin) / grid_step, (box_high - origin) / grid_step,
            box_mass, grid.size,
        )
        kernel = _voigt_profile_per_angstrom(
            center + grid, center, gaussian_sigma, lorentz_hwhm
        )
        kernel_sum = float(np.sum(kernel))
        if kernel_sum > 0.0 and np.isfinite(kernel_sum):
            masses = fftconvolve(masses, kernel / kernel_sum, mode="same")
        return grid, np.maximum(masses, 0.0) / grid_step

    offset = wavelength - center
    fine_grid, fine_profile = convolved(
        fine_step, fine_half, inner_low, inner_high, inner_mass
    )
    result = core_weight * _voigt_profile_per_angstrom(
        wavelength, center, gaussian_sigma, lorentz_hwhm
    )
    inside = np.abs(offset) <= fine_half
    result[inside] += np.interp(offset[inside], fine_grid, fine_profile)
    if half_extent > fine_half:
        # Outer boxes: the parts below and above the fine range.  On the
        # coarse grid (bins many impact widths wide) the impact profile only
        # adds Lorentz wings, so the outer pattern is binned without
        # convolution and the inner components add their wing, V(x), beyond
        # the fine range.
        below_mass = mass * np.clip(np.minimum(high, -fine_half) - low, 0.0, None) / (high - low)
        above_mass = mass * np.clip(high - np.maximum(low, fine_half), 0.0, None) / (high - low)
        keep_below = below_mass > 0.0
        keep_above = above_mass > 0.0
        half_bins = int(np.ceil(half_extent / coarse_step))
        coarse_grid = coarse_step * (np.arange(2 * half_bins + 1) - half_bins)
        origin = coarse_grid[0] - 0.5 * coarse_step
        masses = _deposit_segments(
            (np.concatenate((low[keep_below], np.maximum(low, fine_half)[keep_above])) - origin) / coarse_step,
            (np.concatenate((np.minimum(high, -fine_half)[keep_below], high[keep_above])) - origin) / coarse_step,
            np.concatenate((below_mass[keep_below], above_mass[keep_above])),
            coarse_grid.size,
        )
        outside = ~inside
        result[outside] += (
            np.interp(offset[outside], coarse_grid, masses / coarse_step)
            + float(np.sum(inner_mass)) * _voigt_profile_per_angstrom(
                wavelength[outside], center, gaussian_sigma, lorentz_hwhm
            )
        )
    return result / max(bound_probability, np.finfo(np.float64).tiny)


def o_i_3p5p_nd5d_electron_stark_rate_coefficient(
    ion: AtomicIon,
    transition: AtomicTransition,
    temperature_kelvin: ArrayLike,
    *,
    minimum_effective_n: float | None = None,
) -> FloatArray | None:
    """Return a literature-anchored Stark rate for the O I 3p--nd series.

    The absolute normalization is the 4d multiplet calculation tabulated by
    Dimitrijevic & Sahal-Brechot (2025, Galaxies 13, 116). Higher
    series members use the asymptotic ``n_eff**5`` impact scaling. Unlike the
    generic SYNSPEC fallback, this deliberately does not cap ``n_eff`` at
    five: the function exists specifically to diagnose the merging high-n
    members of one identified series. It is capped at twelve only as a
    numerical safeguard outside the range represented by the line list.
    ``None`` identifies transitions outside this series or,
    when ``minimum_effective_n`` is supplied, below the requested high-Rydberg
    cutoff.  The cutoff is useful for testing the asymptotic extrapolation
    without replacing the generic widths of lower series members.
    """

    if ion.element != "O" or ion.charge != 0:
        return None
    levels = {level.index: level for level in ion.levels}
    lower = levels.get(transition.lower_index)
    upper = levels.get(transition.upper_index)
    if (
        lower is None
        or upper is None
        or ".3p.(5P<" not in lower.label
        or re.search(r"\.\d+d\.\(5Do<", upper.label) is None
        or ion.ionization_energy_ev is None
    ):
        return None
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    binding_energy_ev = (
        ion.ionization_energy_ev
        - upper.energy_wavenumber / EV_TO_WAVENUMBER
    )
    if binding_energy_ev <= 0.0:
        return np.zeros_like(temperature)
    uncapped_effective_n = float(np.sqrt(_RYDBERG_ENERGY_EV / binding_energy_ev))
    if (
        minimum_effective_n is not None
        and uncapped_effective_n < minimum_effective_n
    ):
        return None
    effective_n = min(12.0, uncapped_effective_n)
    reference_center_cm = (
        _O_I_3P5P_4D5D_STARK_REFERENCE_WAVELENGTH * 1.0e-8
    )
    reference_rate_per_electron = (
        2.0
        * PI
        * LIGHT_SPEED
        * _O_I_3P5P_4D5D_STARK_FWHM_ANGSTROM
        * 1.0e-8
        / reference_center_cm**2
        / _O_I_3P5P_4D5D_STARK_REFERENCE_DENSITY
    )
    return (
        reference_rate_per_electron
        * (effective_n / _O_I_3P5P_4D5D_REFERENCE_EFFECTIVE_N) ** 5
        * (_O_I_3P5P_4D5D_STARK_REFERENCE_TEMPERATURE / temperature) ** 0.5
    )


def o_i_optical_electron_stark_rate_coefficient(
    ion: AtomicIon,
    transition: AtomicTransition,
    temperature_kelvin: ArrayLike,
) -> FloatArray | None:
    """Return published electron widths for three strong O I multiplets.

    The tabulation gives FWHM at ``ne=1e16 cm^-3``.  Widths are interpolated
    log-linearly in temperature and converted to the angular-frequency rate
    convention used by the line-opacity calculation.  ``None`` is returned
    for every other transition so the ordinary hierarchy can continue to the
    measured high-series or generic classical fallback.
    """

    if ion.element != "O" or ion.charge != 0:
        return None
    center = transition.wavelength_vacuum_angstrom
    if 4360.0 < center < 4380.0:
        multiplet = "3s3S-4p3P"
    elif 7768.0 < center < 7782.0:
        multiplet = "3s5S-3p5P"
    elif 8442.0 < center < 8455.0:
        multiplet = "3s3S-3p3P"
    else:
        return None
    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    fwhm = np.exp(np.interp(
        np.log(temperature),
        np.log(_O_I_OPTICAL_STARK_TEMPERATURE_K),
        np.log(_O_I_OPTICAL_STARK_FWHM_ANGSTROM[multiplet]),
    ))
    center_cm = center * 1.0e-8
    return (
        2.0 * PI * LIGHT_SPEED * fwhm * 1.0e-8
        / center_cm**2 / _O_I_OPTICAL_STARK_REFERENCE_DENSITY
    )


def metal_rydberg_level_occupation_probability(
    ion: AtomicIon,
    level: AtomicLevel,
    electron_density: ArrayLike,
    temperature_kelvin: ArrayLike,
    *,
    correlated_microfields: bool = True,
    neutral_perturber_number_density: Mapping[str, ArrayLike] | None = None,
    neutral_perturber_radius_cm: Mapping[str, float] | None = None,
) -> FloatArray:
    """Return the joint charged- and neutral-perturber survival of a level.

    The effective principal quantum number is inferred from the actual Stout
    binding energy and the charge of the residual ionic core.  This extends
    the standard hydrogenic microfield criterion to high Rydberg levels of
    complex atoms without assigning integer shells or fitting individual
    lines.  Deeply bound states approach unit survival.  If neutral
    perturbers are supplied, their independent Hummer--Mihalas excluded-volume
    probability is multiplied into the charged-particle result.
    """

    electron_density, temperature = np.broadcast_arrays(
        np.asarray(electron_density, dtype=np.float64),
        np.asarray(temperature_kelvin, dtype=np.float64),
    )
    if (
        np.any(~np.isfinite(electron_density))
        or np.any(electron_density < 0.0)
        or np.any(~np.isfinite(temperature))
        or np.any(temperature <= 0.0)
    ):
        raise ValueError("electron density and temperature must be physical")
    if ion.ionization_energy_ev is None:
        return np.ones_like(electron_density)
    binding_energy_ev = (
        ion.ionization_energy_ev
        - level.energy_wavenumber / EV_TO_WAVENUMBER
    )
    if binding_energy_ev <= 0.0:
        return np.zeros_like(electron_density)
    core_charge = float(ion.charge + 1)
    effective_principal_quantum_number = np.sqrt(
        _RYDBERG_ENERGY_EV * core_charge**2 / binding_energy_ev
    )
    charged_survival = charged_particle_hydrogen_occupation_probability(
        electron_density,
        max(1.0, float(effective_principal_quantum_number)),
        temperature if correlated_microfields else None,
        ionic_charge=core_charge,
    )
    if neutral_perturber_number_density is None:
        return charged_survival
    if neutral_perturber_radius_cm is None:
        raise ValueError("neutral perturber radii are required with their densities")
    neutral_survival = metal_neutral_hard_sphere_occupation_probability(
        hydrogenic_metal_level_mean_radius_cm(ion, level),
        neutral_perturber_number_density,
        neutral_perturber_radius_cm,
    )
    return charged_survival * neutral_survival


def metal_rydberg_transition_survival_probability(
    ion: AtomicIon,
    lower_level: AtomicLevel,
    upper_level: AtomicLevel,
    electron_density: ArrayLike,
    temperature_kelvin: ArrayLike,
    *,
    correlated_microfields: bool = True,
    neutral_perturber_number_density: Mapping[str, ArrayLike] | None = None,
    neutral_perturber_radius_cm: Mapping[str, float] | None = None,
    continuum_cutoff_probability: float | None = None,
    level_survival_cache: dict | None = None,
) -> FloatArray:
    """Return the Q-MHD survival factor for a metal transition.

    Q-MHD occupation probabilities describe dissolution of *bound* Rydberg
    levels.  They do not apply to a discrete transition from a bound lower
    level into an autoionizing upper resonance.  NIST and Stout contain such
    observed lines (for example O I 7158.674 A), and the added O I
    6258--6271-A multiplet connects two autoionizing configurations.  A
    bound-level occupation probability is undefined for either case.  Retain
    those resonances unchanged instead of interpreting negative binding
    energy as complete pressure dissolution.  Their LTE Saha--Boltzmann
    population remains a separate approximation to the resonant-state
    population.

    ``level_survival_cache`` may be a dictionary shared by every transition
    evaluated on the same atmosphere and perturber state; level occupation
    probabilities are then computed once per level instead of once per line.
    """

    if (
        continuum_cutoff_probability is not None
        and (
            not np.isfinite(continuum_cutoff_probability)
            or not 0.0 < continuum_cutoff_probability < 1.0
        )
    ):
        raise ValueError("the continuum-cutoff probability must lie in (0, 1)")
    electron_density, temperature = np.broadcast_arrays(
        np.asarray(electron_density, dtype=np.float64),
        np.asarray(temperature_kelvin, dtype=np.float64),
    )
    if ion.ionization_energy_ev is None:
        return np.ones_like(electron_density)
    threshold_wavenumber = ion.ionization_energy_ev * EV_TO_WAVENUMBER
    if lower_level.energy_wavenumber >= threshold_wavenumber:
        return np.ones_like(electron_density)
    if upper_level.energy_wavenumber >= threshold_wavenumber:
        return np.ones_like(electron_density)
    def level_survival(level: AtomicLevel) -> FloatArray:
        key = (ion.element, ion.charge, level.index)
        if level_survival_cache is not None and key in level_survival_cache:
            return level_survival_cache[key]
        value = metal_rydberg_level_occupation_probability(
            ion,
            level,
            electron_density,
            temperature,
            correlated_microfields=correlated_microfields,
            neutral_perturber_number_density=neutral_perturber_number_density,
            neutral_perturber_radius_cm=neutral_perturber_radius_cm,
        )
        if level_survival_cache is not None:
            level_survival_cache[key] = value
        return value

    lower_survival = level_survival(lower_level)
    upper_survival = level_survival(upper_level)
    survival = np.clip(
        upper_survival
        / np.maximum(lower_survival, np.finfo(np.float64).tiny),
        0.0,
        1.0,
    )
    if continuum_cutoff_probability is not None:
        # A sharp continuum edge is a useful alternative to treating every
        # microfield realization as a separately weakened line.  The natural
        # 50-percent choice places the edge at the median critical field and
        # introduces no energy or density fitted to an individual spectrum.
        survival = (
            survival >= continuum_cutoff_probability
        ).astype(np.float64)
    return survival


def oxygen_i_dissolved_series_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    atomic_database: AtomicDatabase,
    metal_state: MetalLTEState,
    *,
    minimum_upper_effective_n: float = 6.5,
    correlated_microfields: bool = True,
    continuum_cutoff_probability: float | None = None,
) -> FloatArray:
    """Redistribute dissolved O I 3p 5P--nd 5Do oscillator strength.

    Each fine-structure component is assigned its own Rydberg-series cell,
    bounded by the midpoints to the adjacent observed members and by the
    series limit on the blue side of the highest retained member.  The
    dissolved fraction of that component is placed in a triangular profile
    whose wavelength integral is unity.  Consequently its frequency-
    integrated cross section is exactly ``pi e^2 f / (m_e c)`` times the
    local dissolved fraction.  This is a conservative alternative to using
    a threshold photoionization cross section as a proxy for missing line
    strength; no line or object-specific amplitude is fitted.

    The present implementation is intentionally restricted to the observed
    O I sequence implicated in the J1637 comparison.  General complex-atom
    series identification requires parent-term information not carried by
    every Stout level label.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be finite, positive, and increasing")
    if (
        not np.isfinite(minimum_upper_effective_n)
        or minimum_upper_effective_n <= 0.0
    ):
        raise ValueError("minimum upper effective n must be finite and positive")
    ion = atomic_database.ions.get(("O", 0))
    if ion is None or ion.ionization_energy_ev is None:
        return np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    levels = {level.index: level for level in ion.levels}
    series: dict[tuple[int, str], list[tuple[float, AtomicTransition, float]]] = {}
    for line in ion.transitions:
        lower = levels.get(line.lower_index)
        upper = levels.get(line.upper_index)
        if lower is None or upper is None or ".3p.(5P<" not in lower.label:
            continue
        match = re.search(r"\.(\d+)d\.\(5Do<([^>]+)>\)", upper.label)
        if match is None:
            continue
        binding_energy_ev = (
            ion.ionization_energy_ev
            - upper.energy_wavenumber / EV_TO_WAVENUMBER
        )
        if binding_energy_ev <= 0.0:
            continue
        effective_n = float(np.sqrt(_RYDBERG_ENERGY_EV / binding_energy_ev))
        series.setdefault((lower.index, match.group(2)), []).append(
            (line.wavelength_vacuum_angstrom, line, effective_n)
        )

    result = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    integrated_cross_section = PI * ELEMENTARY_CHARGE_ESU**2 / (
        ELECTRON_MASS * LIGHT_SPEED
    )
    lower_population_cache: dict[int, FloatArray] = {}
    populations = metal_state.ion_number_density.get("O")
    if populations is None or populations.shape[0] == 0:
        return result
    partition = metal_state.partition_function[("O", 0)]
    charged_perturber_density = metal_ionic_microfield_perturber_density(
        metal_state
    )
    for members in series.values():
        members.sort(key=lambda item: item[0])
        lower = levels[members[0][1].lower_index]
        lower_binding_ev = (
            ion.ionization_energy_ev
            - lower.energy_wavenumber / EV_TO_WAVENUMBER
        )
        if lower_binding_ev <= 0.0:
            continue
        series_limit = (
            PLANCK * LIGHT_SPEED / (lower_binding_ev * EV_TO_ERG) * 1.0e8
        )
        centers = np.asarray([item[0] for item in members], dtype=np.float64)
        boundaries = np.empty(centers.size + 1, dtype=np.float64)
        boundaries[0] = min(series_limit, centers[0])
        boundaries[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        boundaries[-1] = centers[-1] + 0.5 * (
            centers[-1] - centers[-2]
        ) if centers.size > 1 else centers[-1] + 1.0
        lower_population = lower_population_cache.get(lower.index)
        if lower_population is None:
            lower_population = (
                populations[0]
                * lower.statistical_weight
                * np.exp(
                    -lower.energy_wavenumber * WAVENUMBER_TO_ERG
                    / (BOLTZMANN * atmosphere.temperature)
                )
                / partition
            )
            lower_population_cache[lower.index] = lower_population
        for member_index, (center, line, effective_n) in enumerate(members):
            if effective_n < minimum_upper_effective_n:
                continue
            left = boundaries[member_index]
            right = boundaries[member_index + 1]
            use = (wavelength >= left) & (wavelength <= right)
            if not np.any(use) or not left < center < right:
                continue
            profile = np.zeros(np.count_nonzero(use), dtype=np.float64)
            selected_wavelength = wavelength[use]
            blue = selected_wavelength <= center
            profile[blue] = (
                2.0 * (selected_wavelength[blue] - left)
                / ((right - left) * (center - left))
            )
            profile[~blue] = (
                2.0 * (right - selected_wavelength[~blue])
                / ((right - left) * (right - center))
            )
            upper = levels[line.upper_index]
            survival = metal_rydberg_transition_survival_probability(
                ion,
                lower,
                upper,
                charged_perturber_density,
                atmosphere.temperature,
                correlated_microfields=correlated_microfields,
                continuum_cutoff_probability=continuum_cutoff_probability,
            )
            dissolved = 1.0 - survival
            photon_exponent = (
                PLANCK * LIGHT_SPEED
                / (
                    selected_wavelength[:, np.newaxis]
                    * 1.0e-8
                    * BOLTZMANN
                    * atmosphere.temperature[np.newaxis, :]
                )
            )
            stimulated = -np.expm1(-photon_exponent)
            wavelength_cm = selected_wavelength * 1.0e-8
            cross_section = (
                integrated_cross_section
                * line.absorption_oscillator_strength
                * profile
                * 1.0e8
                * wavelength_cm**2
                / LIGHT_SPEED
            )
            result[use] += (
                cross_section[:, np.newaxis]
                * lower_population[np.newaxis, :]
                / atmosphere.mass_density[np.newaxis, :]
                * dissolved[np.newaxis, :]
                * stimulated
            )
    return result


def hydrogenic_metal_level_mean_radius_cm(
    ion: AtomicIon,
    level: AtomicLevel,
) -> float:
    """Return the hydrogenic mean orbital radius used by HM88.

    The effective principal quantum number comes from the observed binding
    energy.  When an outer-orbital label is available, ``<r>`` uses the exact
    hydrogenic ``[3 n*^2-l(l+1)] a0/(2 Z)`` expression.  The shell-averaged
    radius is used for levels whose configuration label cannot identify
    ``l``.  No empirical radius multiplier is applied.
    """

    if ion.ionization_energy_ev is None:
        raise ValueError("an ionization threshold is required for a level radius")
    binding_energy_ev = (
        ion.ionization_energy_ev
        - level.energy_wavenumber / EV_TO_WAVENUMBER
    )
    if not np.isfinite(binding_energy_ev) or binding_energy_ev <= 0.0:
        raise ValueError("a positive bound-state energy is required for a level radius")
    core_charge = float(ion.charge + 1)
    effective_n = float(np.sqrt(
        _RYDBERG_ENERGY_EV * core_charge**2 / binding_energy_ev
    ))
    angular_momentum = _outer_orbital_angular_momentum(level)
    if angular_momentum is not None:
        radial_factor = 0.5 * (
            3.0 * effective_n**2
            - angular_momentum * (angular_momentum + 1.0)
        )
    else:
        radial_factor = 0.25 * (5.0 * effective_n**2 + 1.0)
    if radial_factor <= 0.0:
        radial_factor = 0.25 * (5.0 * effective_n**2 + 1.0)
    return float(BOHR_RADIUS_CM * radial_factor / core_charge)


def hydrogenic_metal_shell_mean_radius_cm(
    effective_principal_quantum_number: ArrayLike,
    core_charge: float,
) -> FloatArray:
    """Return the shell-averaged hydrogenic radius for a fictitious level."""

    effective_n = np.asarray(
        effective_principal_quantum_number, dtype=np.float64
    )
    if (
        np.any(~np.isfinite(effective_n))
        or np.any(effective_n <= 0.0)
        or not np.isfinite(core_charge)
        or core_charge <= 0.0
    ):
        raise ValueError("effective n and residual-core charge must be positive")
    return BOHR_RADIUS_CM * (5.0 * effective_n**2 + 1.0) / (4.0 * core_charge)


def metal_neutral_hard_sphere_occupation_probability(
    level_radius_cm: ArrayLike,
    neutral_perturber_number_density: Mapping[str, ArrayLike],
    neutral_perturber_radius_cm: Mapping[str, float],
) -> FloatArray:
    """Return the HM88 excluded-volume occupation probability.

    This is ``exp[-4 pi/3 sum_j n_j (r_i+r_j)^3]`` in the low-excitation
    approximation, where neutral perturbers reside in their ground states.
    """

    level_radius = np.asarray(level_radius_cm, dtype=np.float64)
    if np.any(~np.isfinite(level_radius)) or np.any(level_radius < 0.0):
        raise ValueError("level radii must be finite and non-negative")
    density_arrays = {
        element: np.asarray(density, dtype=np.float64)
        for element, density in neutral_perturber_number_density.items()
    }
    if not density_arrays:
        return np.ones_like(level_radius)
    missing = set(density_arrays) - set(neutral_perturber_radius_cm)
    if missing:
        raise ValueError(f"missing neutral perturber radii for {sorted(missing)}")
    broadcast = np.broadcast_arrays(level_radius, *density_arrays.values())
    radius = broadcast[0]
    exponent = np.zeros_like(radius)
    for (element, _), density in zip(density_arrays.items(), broadcast[1:]):
        perturber_radius = float(neutral_perturber_radius_cm[element])
        if (
            np.any(~np.isfinite(density))
            or np.any(density < 0.0)
            or not np.isfinite(perturber_radius)
            or perturber_radius < 0.0
        ):
            raise ValueError("neutral perturber densities and radii must be physical")
        exponent += density * (radius + perturber_radius) ** 3
    return np.exp(-np.minimum((4.0 * PI / 3.0) * exponent, 745.0))


def bulk_metal_neutral_perturber_data(
    atomic_database: AtomicDatabase,
    metal_state: MetalLTEState,
) -> tuple[dict[str, FloatArray], dict[str, float]]:
    """Return ground-state neutral densities and hydrogenic radii for HM88."""

    densities: dict[str, FloatArray] = {}
    radii: dict[str, float] = {}
    for element, populations in metal_state.ion_number_density.items():
        if populations.shape[0] == 0:
            continue
        ion = atomic_database.ions.get((element, 0))
        if ion is None:
            continue
        bound_levels = tuple(
            level for level in ion.levels
            if (
                level.energy_wavenumber >= 0.0
                and (
                    ion.ionization_energy_ev is None
                    or level.energy_wavenumber
                    < ion.ionization_energy_ev * EV_TO_WAVENUMBER
                )
            )
        )
        if not bound_levels:
            continue
        ground = min(bound_levels, key=lambda level: level.energy_wavenumber)
        densities[element] = np.asarray(populations[0], dtype=np.float64)
        radii[element] = hydrogenic_metal_level_mean_radius_cm(ion, ground)
    return densities, radii


def metal_ionic_microfield_perturber_density(
    metal_state: MetalLTEState,
) -> FloatArray:
    """Return the ionic density moment that sets the Holtsmark field scale.

    For a multicomponent ionic mixture the characteristic microfield obeys
    ``F_0 proportional to (sum(n_i Z_i**1.5))**(2/3)``.  Substituting the
    electron density, ``sum(n_i Z_i)``, is exact only when every perturber is
    singly charged.  The distinction is normally negligible in cool line-
    forming layers, but keeping the actual moment avoids a composition-
    dependent approximation in bulk-metal atmospheres.
    """

    result = np.zeros_like(metal_state.electron_density, dtype=np.float64)
    for populations in metal_state.ion_number_density.values():
        for charge in range(1, populations.shape[0]):
            result += charge**1.5 * populations[charge]
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("ionic populations must give a physical microfield density")
    return result


def _outer_orbital_angular_momentum(level: AtomicLevel) -> int | None:
    """Infer the outermost orbital angular momentum from a Stout label."""

    matches = re.findall(r"(?:^|\.)(\d+)([spdfghik])\d*", level.label)
    if not matches:
        return None
    return "spdfghik".index(matches[-1][1])


_CLOSED_SUBSHELL = re.compile(r"^\d+(?:s2|p6|d10|f14)$")
_ORBITAL_TOKEN = re.compile(r"^(\d+)([spdfghik])(\d*)$")


def _normalized_configuration(orbitals: Iterable[str]) -> tuple[str, ...]:
    """Drop closed subshells so labels of adjacent ions can be compared."""

    return tuple(token for token in orbitals if not _CLOSED_SUBSHELL.match(token))


def _parent_limit_excitation_ev(
    level: AtomicLevel, parent_ion: AtomicIon | None
) -> float:
    """Return the excitation of the parent core that a level converges to.

    A level such as Si II ``3s.3p.(3Po).4p`` belongs to the series converging
    on the excited Si III ``3s3p 3P`` term, not on the Si III ground state.
    Its binding energy is therefore ``I + E(parent) - E(level)``.  The parent
    is the next ion's configuration with the outermost electron removed;
    when the label names an intermediate parent term, that term is used,
    otherwise the lowest level of the parent configuration.  Zero is returned
    when the parent cannot be identified, which recovers the ground limit.
    """

    if parent_ion is None:
        return 0.0
    tokens = level.label.split(".")
    orbitals = [token for token in tokens if _ORBITAL_TOKEN.match(token)]
    if not orbitals:
        return 0.0
    outer = orbitals[-1]
    outer_index = max(index for index, token in enumerate(tokens) if token == outer)
    match = _ORBITAL_TOKEN.match(outer)
    occupancy = int(match.group(3) or 1)
    parent_orbitals = orbitals[:-1]
    if occupancy > 1:
        remaining = occupancy - 1
        parent_orbitals.append(
            match.group(1) + match.group(2) + ("" if remaining == 1 else str(remaining))
        )
    parent_configuration = _normalized_configuration(parent_orbitals)
    parent_term = None
    if outer_index > 0 and tokens[outer_index - 1].startswith("("):
        parent_term = tokens[outer_index - 1].strip("()").lstrip("abcdefghijklmnopqrstuvwxyz")

    candidates = []
    for candidate in parent_ion.levels:
        candidate_tokens = candidate.label.split(".")
        candidate_orbitals = [
            token for token in candidate_tokens if _ORBITAL_TOKEN.match(token)
        ]
        if _normalized_configuration(candidate_orbitals) != parent_configuration:
            continue
        final_term = re.sub(r"<[^>]+>", "", candidate_tokens[-1]).strip("()")
        final_term = final_term.lstrip("abcdefghijklmnopqrstuvwxyz")
        candidates.append((candidate, final_term))
    if parent_term is not None:
        named = [item for item in candidates if item[1] == parent_term]
        if named:
            candidates = named
    if not candidates:
        return 0.0
    lowest = min(candidate.energy_wavenumber for candidate, _ in candidates)
    return float(max(lowest, 0.0) / EV_TO_WAVENUMBER)


def _unsold_hydrogen_temperature_coefficient(
    ion: AtomicIon,
    transition: AtomicTransition,
    parent_ion: AtomicIon | None = None,
) -> float | None:
    """Return the temperature-independent part of the Unsold H width.

    Hydrogenic mean-square radii use the effective principal quantum number
    ``n*^2 = Z^2 R/E`` and ``<r^2> = n*^2 [5 n*^2 + 1 - 3 l(l+1)] / (2 Z^2)``
    in units of ``a0^2``, where ``Z`` is the charge seen by the optical
    electron.  Binding energies are measured from the first ionization limit,
    as in the Warner/Kurucz convention.  When either level lies above that
    limit and ``parent_ion`` (the next ionization stage) is supplied, both
    levels are measured from their own parent limits (see
    :func:`_parent_limit_excitation_ev`).
    """

    if ion.ionization_energy_ev is None:
        return None
    levels = {level.index: level for level in ion.levels}
    lower = levels.get(transition.lower_index)
    upper = levels.get(transition.upper_index)
    if lower is None or upper is None:
        return None
    lower_l = _outer_orbital_angular_momentum(lower)
    upper_l = _outer_orbital_angular_momentum(upper)
    if lower_l is None or upper_l is None:
        return None

    stage_charge = ion.charge + 1.0
    # A bound level above the first limit must belong to a series converging
    # on an excited parent term.  Both levels of such a transition are then
    # measured from their own parent limits so the radii stay comparable.
    use_parent_limits = parent_ion is not None and any(
        level.energy_wavenumber / EV_TO_WAVENUMBER >= ion.ionization_energy_ev
        for level in (lower, upper)
    )

    def mean_square_radius(
        level: AtomicLevel, angular_momentum: int
    ) -> float | None:
        binding_energy = (
            ion.ionization_energy_ev
            - level.energy_wavenumber / EV_TO_WAVENUMBER
        )
        if use_parent_limits:
            binding_energy += _parent_limit_excitation_ev(level, parent_ion)
        if binding_energy <= 0.0:
            return None
        effective_n_squared = (
            _RYDBERG_ENERGY_EV * stage_charge**2 / binding_energy
        )
        radius = 0.5 * effective_n_squared * (
            5.0 * effective_n_squared
            + 1.0
            - 3.0 * angular_momentum * (angular_momentum + 1.0)
        ) / stage_charge**2
        return radius if np.isfinite(radius) and radius > 0.0 else None

    lower_radius = mean_square_radius(lower, lower_l)
    upper_radius = mean_square_radius(upper, upper_l)
    if lower_radius is None or upper_radius is None:
        return None
    radius_difference = upper_radius - lower_radius
    if radius_difference <= 0.0:
        return None
    return float(10.0**-9.53 * radius_difference**0.4)


def unsold_hydrogen_impact_rate_coefficient(
    ion: AtomicIon,
    transition: AtomicTransition,
    temperature_kelvin: ArrayLike,
    *,
    parent_ion: AtomicIon | None = None,
) -> FloatArray | None:
    """Estimate neutral-H damping per perturber from the line's levels.

    This is the classical Unsold/Warner approximation.  Effective principal
    quantum numbers and mean-square radii are calculated for the lower and
    upper Stout levels.  The result is an angular-frequency damping rate in
    cm3 s-1.  ``None`` is returned when the labels or bound-level energies do
    not support the hydrogenic estimate, allowing the caller to use its
    explicit fallback.

    This H-perturber branch matters in cool DAZ atmospheres.  Previously the
    metal-line formal solution applied neutral-perturber broadening only to
    helium atmospheres, leaving cool hydrogen-atmosphere metal lines at their
    thermal or radiative widths.
    """

    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    coefficient = _unsold_hydrogen_temperature_coefficient(
        ion, transition, parent_ion
    )
    if coefficient is None:
        return None
    return coefficient * temperature**0.3


def unsold_helium_impact_rate_coefficient(
    ion: AtomicIon,
    transition: AtomicTransition,
    temperature_kelvin: ArrayLike,
    *,
    parent_ion: AtomicIon | None = None,
) -> FloatArray | None:
    """Estimate neutral-He damping per perturber from the line's levels.

    The classical neutral-H Unsold/Warner rate is rescaled by the He/H
    polarizability ratio and the radiator--perturber reduced mass.

    For Ca II K this reproduces Hammond's (1975) own Unsold estimate,
    3.3e-9 cm3 s-1 at 5200 K, which is about half of his laboratory width.
    That factor of about two is the known Unsold underestimate; production
    Ca II H/K therefore uses the measured width instead.
    """

    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be finite and positive")
    hydrogen_rate = unsold_hydrogen_impact_rate_coefficient(
        ion, transition, temperature, parent_ion=parent_ion
    )
    if hydrogen_rate is None:
        return None

    return neutral_impact_rate_from_hydrogen(
        ion,
        hydrogen_rate,
        perturber_atomic_mass_u=_HELIUM_ATOMIC_MASS_U,
        perturber_static_polarizability_a3=_HELIUM_STATIC_POLARIZABILITY_A3,
    )


def neutral_impact_rate_from_hydrogen(
    ion: AtomicIon,
    hydrogen_rate_coefficient: ArrayLike,
    *,
    perturber_atomic_mass_u: float,
    perturber_static_polarizability_a3: float,
) -> FloatArray:
    """Rescale an H-impact damping coefficient to another neutral gas.

    The induced-dipole width scales as polarizability to the two-fifths
    power and reduced mass to the minus three-tenths power.  This accepts
    either a classical Unsold coefficient or an explicit Kurucz gamma_w/nH,
    allowing line-specific H data to be used in hydrogen-free atmospheres.
    """

    rate = np.asarray(hydrogen_rate_coefficient, dtype=np.float64)
    if np.any(~np.isfinite(rate)) or np.any(rate < 0.0):
        raise ValueError("hydrogen_rate_coefficient must be finite and non-negative")
    if (
        not np.isfinite(perturber_atomic_mass_u)
        or perturber_atomic_mass_u <= 0.0
        or not np.isfinite(perturber_static_polarizability_a3)
        or perturber_static_polarizability_a3 <= 0.0
    ):
        raise ValueError("neutral perturber mass and polarizability must be positive")
    radiator_mass = ion.atomic_mass_u
    reduced_mass_h = (
        radiator_mass * _HYDROGEN_ATOMIC_MASS_U
        / (radiator_mass + _HYDROGEN_ATOMIC_MASS_U)
    )
    reduced_mass = (
        radiator_mass * perturber_atomic_mass_u
        / (radiator_mass + perturber_atomic_mass_u)
    )
    scale = (
        (
            perturber_static_polarizability_a3
            / _HYDROGEN_STATIC_POLARIZABILITY_A3
        ) ** 0.4
        * (reduced_mass_h / reduced_mass) ** 0.3
    )
    return rate * scale


def unsold_neutral_metal_impact_rate_coefficient(
    ion: AtomicIon,
    transition: AtomicTransition,
    temperature_kelvin: ArrayLike,
    perturber: str,
    *,
    parent_ion: AtomicIon | None = None,
) -> FloatArray | None:
    """Estimate Unsold damping by a supported neutral bulk-metal perturber.

    The induced-dipole width scales as polarizability to the two-fifths power
    and reduced mass to the minus three-tenths power.  Rescaling the existing
    H-perturber Unsold/Warner result keeps the line-specific radiator physics
    unchanged while supplying the physically relevant host gas in a bulk
    C/O atmosphere.  Polarizabilities are experimental NIST CCCBDB values.
    """

    symbol = _canonical_element(perturber)
    if symbol not in _BULK_METAL_STATIC_POLARIZABILITY_A3:
        raise ValueError(
            "neutral-metal Unsold broadening currently supports C, O, and Ne"
        )
    hydrogen_rate = unsold_hydrogen_impact_rate_coefficient(
        ion, transition, temperature_kelvin, parent_ion=parent_ion
    )
    if hydrogen_rate is None:
        return None
    return neutral_impact_rate_from_hydrogen(
        ion,
        hydrogen_rate,
        perturber_atomic_mass_u=ATOMIC_MASS_U[symbol],
        perturber_static_polarizability_a3=(
            _BULK_METAL_STATIC_POLARIZABILITY_A3[symbol]
        ),
    )


def oxygen_negative_ion_lte_number_density(
    atmosphere: Atmosphere,
    metal_state: MetalLTEState,
) -> FloatArray:
    """Return the LTE O-minus ground-state number density in ``cm^-3``.

    O-minus is treated as a trace negative ion, so its small contribution to
    charge neutrality and mass density is neglected.  The attachment Saha
    relation uses the O-(2P) statistical weight 6 and the complete O I(3P)
    ground-term weight 9.
    """

    if "O" not in metal_state.ion_number_density:
        return np.zeros(atmosphere.n_depth, dtype=np.float64)
    if metal_state.electron_density.shape != atmosphere.temperature.shape:
        raise ValueError("metal_state and atmosphere depth grids do not match")
    statistical_weight_ratio = 6.0 / (2.0 * 9.0)
    inverse_translation = (
        PLANCK**2
        / (2.0 * PI * ELECTRON_MASS * BOLTZMANN * atmosphere.temperature)
    ) ** 1.5
    attachment = np.exp(np.minimum(
        _O_MINUS_ELECTRON_AFFINITY_EV * EV_TO_ERG
        / (BOLTZMANN * atmosphere.temperature),
        745.0,
    ))
    neutral_oxygen = metal_state.ion_number_density["O"][0]
    return np.ascontiguousarray(
        neutral_oxygen
        * metal_state.electron_density
        * statistical_weight_ratio
        * inverse_translation
        * attachment
    )


def cool_metal_negative_ion_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    metal_state: MetalLTEState,
) -> FloatArray:
    """Approximate O-/Ne-/Na- continuum opacity in ``cm2 g-1``.

    The free-free part is the optical/long-wavelength John (1975a,b)
    prescription used in LP 40-365 atmosphere work.  The O- bound-free part
    is a resonance-averaged Wigner-threshold envelope constrained by measured
    visible photodetachment cross sections.  This compact implementation is
    intended for assessing cool O/Ne atmosphere structure; it does not claim
    the detailed resonances of the full negative-ion calculations cited by
    Raddi et al. (2019).
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1
        or np.any(~np.isfinite(wavelength))
        or np.any(wavelength <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be finite, positive, and 1D")
    if metal_state.electron_density.shape != atmosphere.temperature.shape:
        raise ValueError("metal_state and atmosphere depth grids do not match")
    shape = (wavelength.size, atmosphere.n_depth)
    result = np.zeros(shape, dtype=np.float64)
    temperature = atmosphere.temperature
    electron_pressure = (
        metal_state.electron_density * BOLTZMANN * temperature
    )

    for element, (value_5000, value_10000) in (
        _JOHN_NEGATIVE_ION_FREE_FREE_A_TIMES_1E34.items()
    ):
        populations = metal_state.ion_number_density.get(element)
        if populations is None or populations.shape[0] == 0:
            continue
        exponent = np.log(value_10000 / value_5000) / np.log(2.0)
        coefficient = (
            value_5000 * (temperature / 5000.0) ** exponent * 1.0e-34
        )
        result += (
            wavelength[:, np.newaxis] ** 2
            * coefficient[np.newaxis, :]
            * populations[0, np.newaxis, :]
            * electron_pressure[np.newaxis, :]
            / atmosphere.mass_density[np.newaxis, :]
        )

    if "O" in metal_state.ion_number_density:
        photon_energy_ev = (
            PLANCK * LIGHT_SPEED
            / (wavelength[:, np.newaxis] * 1.0e-8) / EV_TO_ERG
        )
        above = photon_energy_ev > _O_MINUS_ELECTRON_AFFINITY_EV
        cross_section = np.where(
            above,
            _O_MINUS_ASYMPTOTIC_CROSS_SECTION_CM2 * np.sqrt(np.maximum(
                1.0
                - _O_MINUS_ELECTRON_AFFINITY_EV / photon_energy_ev,
                0.0,
            )),
            0.0,
        )
        stimulated = -np.expm1(
            -PLANCK * LIGHT_SPEED
            / (
                wavelength[:, np.newaxis] * 1.0e-8
                * BOLTZMANN * temperature[np.newaxis, :]
            )
        )
        result += (
            cross_section
            * oxygen_negative_ion_lte_number_density(
                atmosphere, metal_state
            )[np.newaxis, :]
            * stimulated
            / atmosphere.mass_density[np.newaxis, :]
        )
    return np.ascontiguousarray(np.maximum(result, 0.0))


def metal_bound_free_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    atomic_database: AtomicDatabase,
    metal_state: MetalLTEState,
    photoionization_database: VernerPhotoionizationDatabase,
    *,
    ion_departure_coefficient: Mapping[tuple[str, int], ArrayLike] | None = None,
    level_departure_coefficient: Mapping[tuple[str, int, int], ArrayLike] | None = None,
    excluded_elements: Iterable[str] = (),
    excluded_ions: Iterable[tuple[str, int]] = (),
) -> FloatArray:
    """Return LTE ground-state metal bound-free opacity in cm2 g-1.

    The Verner fits smooth the Opacity Project resonance structure and cover
    only ground-state photoionization.  Like the Opacity Project data, they
    are cross sections per atom in the LS ground term, so the population
    multiplier is the Boltzmann population of every fine-structure level of
    that term (for example all three Si I 3P levels), not only the lowest
    ``J`` level.  Departure coefficients multiply absorption; stimulated
    recombination remains at its LTE value, ``n*(b - exp(-h nu/kT))``.  The
    spectroscopic edge remains at the isolated-atom energy even when the
    dense-helium free-energy correction is enabled in the Saha balance.
    ``excluded_ions`` permits level-resolved Opacity Project data to replace
    only the covered charge states while retaining Verner fallbacks for other
    stages of the same element.
    """

    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if (
        wavelength.ndim != 1 or wavelength.size < 2
        or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be finite, positive, and increasing")
    if metal_state.electron_density.shape != atmosphere.temperature.shape:
        raise ValueError("metal_state and atmosphere depth grids do not match")
    excluded = {_canonical_element(element) for element in excluded_elements}
    excluded_stages = {
        (_canonical_element(element), int(charge))
        for element, charge in excluded_ions
    }

    wavelength_cm = wavelength * 1.0e-8
    photon_energy_ev = PLANCK * LIGHT_SPEED / wavelength_cm / EV_TO_ERG
    stimulated = -np.expm1(
        -PLANCK * LIGHT_SPEED
        / (wavelength_cm[:, np.newaxis] * BOLTZMANN * atmosphere.temperature)
    )
    result = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    for element in metal_state.log_number_abundance:
        if element in excluded:
            continue
        stages = atomic_database.ion_stages(element)
        populations = metal_state.ion_number_density[element]
        for ion in stages:
            if (element, ion.charge) in excluded_stages:
                continue
            fit = photoionization_database.fits.get((element, ion.charge))
            if fit is None or ion.charge >= populations.shape[0]:
                continue
            partition = metal_state.partition_function[(element, ion.charge)]
            ion_departure = None
            if ion_departure_coefficient is not None:
                ion_departure = np.asarray(
                    ion_departure_coefficient.get(
                        (element, ion.charge), np.ones(atmosphere.n_depth)
                    ),
                    dtype=np.float64,
                )
                if ion_departure.shape != atmosphere.temperature.shape:
                    raise ValueError("ion departure coefficients must match depth")
            # The fits are per atom in the LS ground term, so every J level
            # of that term contributes.  Non-LTE levels absorb in proportion
            # to b while stimulated recombination stays at the LTE rate.
            ground_population = np.zeros(atmosphere.n_depth)
            departure_weighted = np.zeros(atmosphere.n_depth)
            for level in ground_term_levels(ion):
                level_population = (
                    populations[ion.charge]
                    * level.statistical_weight
                    * np.exp(
                        -level.energy_wavenumber * WAVENUMBER_TO_ERG
                        / (BOLTZMANN * atmosphere.temperature)
                    )
                    / partition
                )
                level_key = (element, ion.charge, level.index)
                if (
                    level_departure_coefficient is not None
                    and level_key in level_departure_coefficient
                ):
                    departure = np.asarray(
                        level_departure_coefficient[level_key], dtype=np.float64
                    )
                    if departure.shape != atmosphere.temperature.shape:
                        raise ValueError("level departure coefficients must match depth")
                elif ion_departure is not None:
                    departure = ion_departure
                else:
                    departure = 1.0
                ground_population += level_population
                departure_weighted += level_population * departure
            departure_factor = np.divide(
                departure_weighted,
                ground_population,
                out=np.ones_like(ground_population),
                where=ground_population > 0.0,
            )
            cross_section = fit.cross_section(photon_energy_ev)
            result += (
                cross_section[:, np.newaxis]
                * ground_population[np.newaxis, :]
                * (departure_factor[np.newaxis, :] - 1.0 + stimulated)
                / atmosphere.mass_density[np.newaxis, :]
            )
    return result


def metal_line_mass_absorption_coefficient(
    atmosphere: Atmosphere,
    wavelength_angstrom: ArrayLike,
    atomic_database: AtomicDatabase,
    metal_state: MetalLTEState,
    *,
    mg_he_red_wing_table: MgHeRedWingTable | None = None,
    mg_ii_he_profile_table: MgIIHeProfileTable | None = None,
    ca_i_he_profile_table: CaIHeProfileTable | None = None,
    ca_ii_he_profile_table: CaIIHeProfileTable | None = None,
    minimum_oscillator_strength: float = 1.0e-4,
    maximum_lines: int | None = 20_000,
    microturbulent_velocity_kms: float = 0.0,
    helium_impact_rate_coefficient: float = 1.0e-9,
    hydrogen_impact_rate_coefficient: float = 1.0e-9,
    ca_ii_helium_impact_scale: float = 1.0,
    include_classical_electron_stark: bool = False,
    classical_electron_stark_scale: float = 1.0,
    classical_electron_stark_scale_by_ion: Mapping[tuple[str, int], float] | None = None,
    include_oxygen_i_series_stark: bool = True,
    oxygen_i_series_stark_minimum_effective_n: float | None = None,
    include_oxygen_i_quasistatic_microfields: bool = False,
    oxygen_i_quasistatic_minimum_effective_n: float = 6.5,
    include_linear_stark_quasistatic: bool = False,
    linear_stark_profile: str = "two-level",
    mg_ii_kurucz_series_widths: bool = False,
    include_rydberg_dissolution: bool = False,
    rydberg_dissolution_cutoff_probability: float | None = None,
    rydberg_dissolution_elements: Iterable[str] | None = None,
    rydberg_correlated_microfields: bool = True,
    rydberg_neutral_perturbers: bool = False,
    excluded_elements: Iterable[str] = (),
    transition_keys: Iterable[tuple[str, int, int, int]] | None = None,
    ion_departure_coefficient: Mapping[tuple[str, int], ArrayLike] | None = None,
    level_departure_coefficient: Mapping[tuple[str, int, int], ArrayLike] | None = None,
    profile_edge_optical_depth: float | None = None,
    profile_edge_optical_depth_elements: Iterable[str] | None = None,
    profile_edge_optical_depth_ions: Iterable[tuple[str, int]] | None = None,
    profile_support_maximum_rosseland_optical_depth: float = 2.0,
    profile_support_maximum_half_window_angstrom: float = 100.0,
    uv_resonance_support_angstrom: Mapping[tuple[str, int, int, int], float] | None = None,
) -> FloatArray:
    """Return LTE metal bound-bound opacity in cm^2 g^-1.

    ``microturbulent_velocity_kms`` is the most-probable microturbulent speed
    ``xi`` of the usual Doppler width ``sqrt(2kT/m + xi^2)`` (the Kurucz and
    MOOG convention), so it adds ``xi^2/2`` to the Gaussian variance.

    Ordinary lines use thermal (plus optional microturbulent), radiative, and
    line-specific Unsold neutral-H and neutral-He impact broadening.  The
    optional classical electron-Stark
    fallback follows SYNSPEC when no measured width is available. Optional
    Q-MHD Rydberg dissolution multiplies transitions by the upper-to-lower
    level survival ratio.  The optional neutral-perturber term multiplies the
    charged survival by the parameter-free hydrogenic-radius excluded-volume
    probability of Hummer & Mihalas (1988). Ca II H/K
    and Mg II 4481 retain their more accurate tabulated widths. Ca II H/K use Hammond's
    laboratory He widths for the helium contribution. Mg I
    2852 A uses the Allard absolute red-wing cross section
    wherever that table exceeds the ordinary profile. If supplied, the
    summed Allard Mg II h/k table is enveloped with both ordinary resonance
    profiles because the figure-recovered cores are clipped. If supplied, the complete
    factorized temperature-density Blouin et al. Ca I--He table replaces the
    ordinary Ca I 4227-A resonance line. A supplied single-condition Ca II table is
    likewise enveloped with the laboratory-width ordinary profiles.
    ``transition_keys`` optionally restricts the result to explicit
    ``(element, charge, lower_index, upper_index)`` transitions.  An explicit
    declaration is independent of the queried wavelength interval: profiles
    centered outside that interval are still evaluated so their wings cannot
    disappear when a spectrum is synthesized in chunks.  This is useful when
    reusing one immutable line selection for structure and formal synthesis.
    ``profile_edge_optical_depth`` replaces the fixed ordinary-line cutoff
    with an optical-depth-aware support criterion.  It expands saturated
    profiles until the vertical line optical depth at the edge is below the
    requested value, considering layers above
    ``profile_support_maximum_rosseland_optical_depth``.
    ``profile_edge_optical_depth_elements`` can restrict that diagnostic to
    complete elements.  ``None`` preserves the all-element behavior and an
    empty iterable disables adaptive support without selecting individual
    observed transitions.
    ``profile_edge_optical_depth_ions`` instead restricts the diagnostic to
    complete ion stages.  It is mutually exclusive with the element filter;
    this supports physically distinct neutral/ionized profile families
    without selecting individual observed lines.
    ``include_linear_stark_quasistatic`` adds the ionic quasi-static
    (Holtsmark) linear-Stark envelope to every genuine Rydberg line (see
    :func:`linear_stark_rydberg_level`), weighted depth by depth by
    :func:`linear_stark_mixing_fraction`; the oscillator strength is conserved.
    ``mg_ii_kurucz_series_widths`` gives every Mg II 2p6-core multiplet its
    Kurucz electron width (D6); otherwise only 4f-8g (4852 A) uses Kurucz.
    ``rydberg_dissolution_elements`` limits ``include_rydberg_dissolution`` to
    the named elements; None (the default) dissolves every element's lines.
    ``uv_resonance_support_angstrom`` supplies already-decided minimum support
    for eligible strong UV resonances. A caller evaluating depth subsets must
    derive this mapping from the complete atmosphere, not the subset. Every
    eligible selected resonance requires an entry. Other callers retain the
    original optical-depth gate when this argument is None.
    """

    if linear_stark_profile not in ("two-level", "manifold"):
        raise ValueError("linear_stark_profile must be 'two-level' or 'manifold'")
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if uv_resonance_support_angstrom is not None and any(
        not np.isfinite(value) or value < 0.0
        for value in uv_resonance_support_angstrom.values()
    ):
        raise ValueError('UV resonance support must be finite and nonnegative')
    if (
        wavelength.ndim != 1 or wavelength.size < 2
        or np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0.0)
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("wavelength_angstrom must be finite, positive, and increasing")
    if (
        minimum_oscillator_strength <= 0.0
        or not np.isfinite(microturbulent_velocity_kms)
        or microturbulent_velocity_kms < 0.0
        or helium_impact_rate_coefficient < 0.0
        or hydrogen_impact_rate_coefficient < 0.0
        or not np.isfinite(ca_ii_helium_impact_scale)
        or ca_ii_helium_impact_scale < 0.0
        or not np.isfinite(classical_electron_stark_scale)
        or classical_electron_stark_scale < 0.0
    ):
        raise ValueError("line thresholds and broadening coefficients must be physical")
    if profile_edge_optical_depth is not None and (
        not np.isfinite(profile_edge_optical_depth)
        or profile_edge_optical_depth <= 0.0
    ):
        raise ValueError("profile_edge_optical_depth must be positive")
    # Line dissolution and its returned pseudo-continuum are two halves of
    # one occupation-probability treatment.  A caller that returns the
    # dissolved strength only for some elements must restrict dissolution to
    # the same elements, or the removed opacity is silently lost.
    dissolution_elements = (
        None
        if rydberg_dissolution_elements is None
        else frozenset(
            _canonical_element(element) for element in rydberg_dissolution_elements
        )
    )

    # A trace state whose partitions already carry occupation probabilities
    # must apply the same level survival, or populations would not sum to
    # the ion density.
    include_rydberg_dissolution = include_rydberg_dissolution or (
        metal_state.metal_level_dissolution
        and metal_state.composition_mode == "trace"
    )

    def dissolves(ion: AtomicIon) -> bool:
        return include_rydberg_dissolution and (
            dissolution_elements is None or ion.element in dissolution_elements
        )

    profile_support_elements = (
        None
        if profile_edge_optical_depth_elements is None
        else frozenset(
            _canonical_element(element)
            for element in profile_edge_optical_depth_elements
        )
    )
    if (
        profile_edge_optical_depth_elements is not None
        and profile_edge_optical_depth_ions is not None
    ):
        raise ValueError(
            "profile-edge element and ion filters are mutually exclusive"
        )
    profile_support_ions = (
        None
        if profile_edge_optical_depth_ions is None
        else frozenset(
            (_canonical_element(element), int(charge))
            for element, charge in profile_edge_optical_depth_ions
        )
    )
    if profile_support_ions is not None and any(
        charge < 0 for _element, charge in profile_support_ions
    ):
        raise ValueError("profile-edge ion charges must be non-negative")
    if (
        not np.isfinite(profile_support_maximum_rosseland_optical_depth)
        or profile_support_maximum_rosseland_optical_depth <= 0.0
        or not np.isfinite(profile_support_maximum_half_window_angstrom)
        or profile_support_maximum_half_window_angstrom <= 0.0
    ):
        raise ValueError("profile-support limits must be positive")
    if (
        oxygen_i_series_stark_minimum_effective_n is not None
        and (
            not np.isfinite(oxygen_i_series_stark_minimum_effective_n)
            or oxygen_i_series_stark_minimum_effective_n <= 0.0
        )
    ):
        raise ValueError("the O I series Stark effective-n cutoff must be positive")
    if classical_electron_stark_scale_by_ion is not None and any(
        not np.isfinite(value) or value < 0.0
        for value in classical_electron_stark_scale_by_ion.values()
    ):
        raise ValueError("ion-specific classical Stark scales must be finite and non-negative")
    if metal_state.electron_density.shape != atmosphere.temperature.shape:
        raise ValueError("metal_state and atmosphere depth grids do not match")
    excluded = {_canonical_element(element) for element in excluded_elements}
    line_abundances = {
        element: abundance
        for element, abundance in metal_state.log_number_abundance.items()
        if element not in excluded
    }
    result = np.zeros((wavelength.size, atmosphere.n_depth), dtype=np.float64)
    replacement_ordinary: dict[str, FloatArray] = {}
    if ca_i_he_profile_table is not None:
        replacement_ordinary["CaI"] = np.zeros_like(result)
        # Impact profile with its He width evaluated at the least-dense
        # unified calculation and scaled linearly to the local density.
        replacement_ordinary["CaI_reference"] = np.zeros_like(result)
        ca_i_reference_density = float(ca_i_he_profile_table.densities[0])
    if mg_ii_he_profile_table is not None:
        replacement_ordinary["MgII"] = np.zeros_like(result)
    if ca_ii_he_profile_table is not None:
        replacement_ordinary["CaII"] = np.zeros_like(result)
    if transition_keys is not None:
        allowed = frozenset(
            (
                _canonical_element(element),
                int(charge),
                int(lower_index),
                int(upper_index),
            )
            for element, charge, lower_index, upper_index in transition_keys
        )
        # Explicit keys are a complete caller-owned declaration.  Do not
        # apply a second ranking or an endpoint-dependent center filter here:
        # a distant strong line can contribute a wing inside a narrow query.
        selected = []
        for element in line_abundances:
            for ion in atomic_database.ion_stages(element):
                for line in ion.transitions:
                    key = (
                        ion.element,
                        ion.charge,
                        line.lower_index,
                        line.upper_index,
                    )
                    if (
                        key in allowed
                        and line.transition_type == "E1"
                        and line.absorption_oscillator_strength
                        >= minimum_oscillator_strength
                    ):
                        selected.append((ion, line))
    else:
        selected = selected_metal_lines(
            atomic_database,
            line_abundances,
            float(wavelength[0]),
            float(wavelength[-1]),
            minimum_oscillator_strength,
            maximum_lines,
            atmosphere.effective_temperature,
            {
                (element, charge): float(np.max(
                    populations[charge]
                    * (
                        1.0
                        if ion_departure_coefficient is None
                        else np.asarray(
                            ion_departure_coefficient.get(
                                (element, charge), np.ones(atmosphere.n_depth)
                            ),
                            dtype=np.float64,
                        )
                    )
                    / np.maximum(
                        metal_state.element_number_density[element],
                        np.finfo(np.float64).tiny,
                    )
                ))
                for element, populations in metal_state.ion_number_density.items()
                for charge in range(populations.shape[0])
            },
        )
    helium_density = (
        atmosphere.helium_lte_state.neutral_he_density
        if atmosphere.helium_lte_state is not None else np.zeros(atmosphere.n_depth)
    )
    hydrogen_density = (
        atmosphere.neutral_h_density
        if metal_state.reference_species == "H" else np.zeros(atmosphere.n_depth)
    )
    bulk_neutral_perturber_density = (
        {
            element: metal_state.ion_number_density[element][0]
            for element in _BULK_METAL_STATIC_POLARIZABILITY_A3
            if (
                metal_state.reference_species == "metal"
                and element in metal_state.ion_number_density
                and metal_state.ion_number_density[element].shape[0] > 0
            )
        }
    )
    ionic_microfield_charge_sum = metal_ionic_microfield_perturber_density(
        metal_state
    )
    ionic_microfield_scale = np.maximum(
        ionic_microfield_charge_sum, 0.0
    ) ** (2.0 / 3.0)
    if rydberg_neutral_perturbers:
        (
            rydberg_neutral_density,
            rydberg_neutral_radius,
        ) = bulk_metal_neutral_perturber_data(atomic_database, metal_state)
    elif metal_state.neutral_perturber_number_density is not None:
        rydberg_neutral_density = metal_state.neutral_perturber_number_density
        rydberg_neutral_radius = metal_state.neutral_perturber_radius_cm
    else:
        rydberg_neutral_density = None
        rydberg_neutral_radius = None
    integrated_cross_section = PI * ELEMENTARY_CHARGE_ESU**2 / (
        ELECTRON_MASS * LIGHT_SPEED
    )
    atomic_mass_unit = 1.660_539_068_92e-24

    # Total downward rate of each level supplies the natural width.
    upper_rates: dict[tuple[str, int, int], float] = {}
    selected_ions = {
        (ion.element, ion.charge): ion for ion, _ in selected
    }.values()
    levels_by_ion = {
        (ion.element, ion.charge): {level.index: level for level in ion.levels}
        for ion in selected_ions
    }
    for ion in selected_ions:
        for transition in ion.transitions:
            key = (ion.element, ion.charge, transition.upper_index)
            upper_rates[key] = upper_rates.get(key, 0.0) + transition.einstein_a

    lower_population_cache: dict[tuple[str, int, int], FloatArray] = {}
    level_survival_cache: dict[tuple[str, int, int], FloatArray] = {}

    def empty_profile_batch() -> dict[str, list[FloatArray | float]]:
        return {
            "center": [],
            "integrated_strength": [],
            "gaussian_sigma": [],
            "lorentz_hwhm": [],
            "minimum_half_window": [],
            "population_scale": [],
        }

    ordinary_profile_batches = {
        "result": empty_profile_batch(),
        "CaI": empty_profile_batch(),
        "MgII": empty_profile_batch(),
        "CaII": empty_profile_batch(),
        "CaI_reference": empty_profile_batch(),
    }

    def flush_profile_batch(key: str) -> None:
        batch = ordinary_profile_batches[key]
        if not batch["center"]:
            return
        target = result if key == "result" else replacement_ordinary[key]
        _accumulate_lte_metal_line_profiles(
            wavelength,
            np.asarray(batch["center"], dtype=np.float64),
            np.asarray(batch["integrated_strength"], dtype=np.float64),
            np.stack(batch["gaussian_sigma"]),
            np.stack(batch["lorentz_hwhm"]),
            np.asarray(batch["minimum_half_window"], dtype=np.float64),
            np.stack(batch["population_scale"]),
            target,
        )
        for values in batch.values():
            values.clear()

    for ion, line in selected:
        lower_key = (ion.element, ion.charge, line.lower_index)
        lower_population = lower_population_cache.get(lower_key)
        if lower_population is None:
            lower = levels_by_ion[(ion.element, ion.charge)][line.lower_index]
            partition = metal_state.partition_function[(ion.element, ion.charge)]
            ion_population = metal_state.ion_number_density[ion.element][ion.charge]
            lower_population = (
                ion_population * lower.statistical_weight
                * np.exp(
                    -lower.energy_wavenumber * WAVENUMBER_TO_ERG
                    / (BOLTZMANN * atmosphere.temperature)
                ) / partition
            )
            if (
                metal_state.metal_level_dissolution
                and dissolves(ion)
                and (
                    ion.ionization_energy_ev is None
                    or lower.energy_wavenumber
                    < ion.ionization_energy_ev * EV_TO_WAVENUMBER
                )
            ):
                # Occupation probabilities describe bound levels.  Keep the
                # LTE resonance-population approximation for autoionizing
                # lower states, as in the transition-survival calculation;
                # their negative binding energy must not erase the line.
                lower_population = lower_population * (
                    metal_rydberg_level_occupation_probability(
                        ion,
                        lower,
                        ionic_microfield_charge_sum,
                        atmosphere.temperature,
                        correlated_microfields=rydberg_correlated_microfields,
                        neutral_perturber_number_density=(
                            rydberg_neutral_density
                        ),
                        neutral_perturber_radius_cm=rydberg_neutral_radius,
                    )
                )
            if (
                level_departure_coefficient is not None
                and lower_key in level_departure_coefficient
            ):
                departure = np.asarray(
                    level_departure_coefficient[lower_key], dtype=np.float64
                )
                if departure.shape != atmosphere.temperature.shape:
                    raise ValueError("level departure coefficients must match depth")
                lower_population = lower_population * departure
            elif ion_departure_coefficient is not None:
                departure = np.asarray(
                    ion_departure_coefficient.get(
                        (ion.element, ion.charge), np.ones(atmosphere.n_depth)
                    ),
                    dtype=np.float64,
                )
                if departure.shape != atmosphere.temperature.shape:
                    raise ValueError("ion departure coefficients must match depth")
                lower_population = lower_population * departure
            lower_population_cache[lower_key] = lower_population
        center = line.wavelength_vacuum_angstrom
        center_cm = center * 1.0e-8
        stimulated = -np.expm1(
            -PLANCK * LIGHT_SPEED
            / (center_cm * BOLTZMANN * atmosphere.temperature)
        )
        # The natural width is the sum of both levels' radiative decay rates;
        # a subordinate line such as Mg II 4481 has a short-lived lower level.
        natural_rate = (
            line.radiative_damping_rate_s
            if line.radiative_damping_rate_s is not None
            else upper_rates[(ion.element, ion.charge, line.upper_index)]
            + upper_rates.get((ion.element, ion.charge, line.lower_index), 0.0)
        )
        lower_level = levels_by_ion[(ion.element, ion.charge)][line.lower_index]
        upper_level = levels_by_ion[(ion.element, ion.charge)][line.upper_index]
        if dissolves(ion):
            line_survival = metal_rydberg_transition_survival_probability(
                ion,
                lower_level,
                upper_level,
                ionic_microfield_charge_sum,
                atmosphere.temperature,
                correlated_microfields=rydberg_correlated_microfields,
                neutral_perturber_number_density=rydberg_neutral_density,
                neutral_perturber_radius_cm=rydberg_neutral_radius,
                continuum_cutoff_probability=(
                    rydberg_dissolution_cutoff_probability
                ),
                level_survival_cache=level_survival_cache,
            )
        else:
            line_survival = np.ones(atmosphere.n_depth, dtype=np.float64)
        local_classical_stark_scale = (
            classical_electron_stark_scale
            if classical_electron_stark_scale_by_ion is None
            else classical_electron_stark_scale_by_ion.get(
                (ion.element, ion.charge), classical_electron_stark_scale
            )
        )
        classical_stark_rate = (
            local_classical_stark_scale
            * classical_electron_stark_rate_coefficient(ion, line)
            if include_classical_electron_stark else 0.0
        )
        oxygen_series_stark_rate = (
            o_i_3p5p_nd5d_electron_stark_rate_coefficient(
                ion,
                line,
                atmosphere.temperature,
                minimum_effective_n=(
                    oxygen_i_series_stark_minimum_effective_n
                ),
            )
            if include_classical_electron_stark and include_oxygen_i_series_stark
            else None
        )
        oxygen_optical_stark_rate = (
            o_i_optical_electron_stark_rate_coefficient(
                ion, line, atmosphere.temperature
            )
            # Published multiplet widths are part of the ordinary line data,
            # not the optional generic classical fallback below.
            if ion.element == "O" and ion.charge == 0 else None
        )
        resonance_half_window = (
            strong_uv_resonance_minimum_half_window_angstrom(
                ion, line, lower_level
            )
        )
        if resonance_half_window > 0.0 and uv_resonance_support_angstrom is not None:
            support_key = (ion.element, ion.charge, line.lower_index, line.upper_index)
            if support_key not in uv_resonance_support_angstrom:
                raise ValueError('Missing declared UV resonance support')
            resonance_half_window = float(uv_resonance_support_angstrom[support_key])
        elif resonance_half_window > 0.0:
            # Use the narrow radiative+thermal profile as a conservative
            # upper bound on the line-center optical depth.  This cheaply
            # suppresses extended formal support for absent LTE ion stages;
            # neutral/electron collisions only lower the actual core peak.
            sigma = center * np.sqrt(
                BOLTZMANN * atmosphere.temperature
                / (ion.atomic_mass_u * atomic_mass_unit * LIGHT_SPEED**2)
                + 0.5 * (microturbulent_velocity_kms * 1.0e5 / LIGHT_SPEED) ** 2
            )
            natural_hwhm = np.full(
                atmosphere.n_depth,
                center_cm**2 * natural_rate
                / (4.0 * PI * LIGHT_SPEED) * 1.0e8,
            )
            center_profile = np.asarray([
                _voigt_profile_per_angstrom(
                    np.asarray([center]),
                    center,
                    float(local_sigma),
                    float(local_hwhm),
                )[0]
                for local_sigma, local_hwhm in zip(sigma, natural_hwhm)
            ])
            center_opacity = (
                integrated_cross_section
                * line.absorption_oscillator_strength
                * center_profile
                * 1.0e8 * center_cm**2 / LIGHT_SPEED
                * lower_population * stimulated
                / atmosphere.mass_density
            )
            center_optical_depth = float(
                center_opacity[0] * atmosphere.column_mass[0]
                + np.sum(
                    0.5 * (center_opacity[1:] + center_opacity[:-1])
                    * np.diff(atmosphere.column_mass)
                )
            )
            if center_optical_depth < 1.0e3:
                resonance_half_window = 0.0
        # Resonance-line data apply to transitions from the ground level only;
        # a wavelength window alone would also catch subordinate lines such
        # as Mg II 3p--3d at 2791/2798 A.
        from_ground = lower_level.energy_wavenumber < 1.0
        is_mg_resonance = (
            ion.element == "Mg" and ion.charge == 0 and 2845.0 < center < 2860.0
            and from_ground
        )
        is_mg_ii_resonance = (
            ion.element == "Mg" and ion.charge == 1 and 2790.0 < center < 2810.0
            and from_ground
        )
        is_mg_ii_4481 = (
            ion.element == "Mg" and ion.charge == 1 and 4478.0 < center < 4487.0
        )
        # D6 opts into Kurucz widths for the whole Mg II series; other
        # families keep their validated 4f-8g-only line-specific width.
        mg_ii_series_rate = (
            mg_ii_kurucz_electron_stark_rate_coefficient(
                lower_level, upper_level, atmosphere.temperature
            )
            if ion.element == "Mg" and ion.charge == 1
            and mg_ii_kurucz_series_widths
            else (
                mg_ii_4852_electron_stark_rate_coefficient(
                    center, atmosphere.temperature
                )
                if ion.element == "Mg" and ion.charge == 1
                and 4848.0 < center < 4857.0
                else None
            )
        )
        is_mg_i_3835 = (
            ion.element == "Mg" and ion.charge == 0 and 3825.0 < center < 3845.0
        )
        mg_i_optical_stark_rate = (
            mg_i_optical_electron_stark_rate_coefficient(
                center, atmosphere.temperature
            )
            if ion.element == "Mg" and ion.charge == 0
            else None
        )
        na_i_optical_stark_rate = (
            na_i_optical_electron_stark_rate_coefficient(
                center, atmosphere.temperature
            )
            if ion.element == "Na" and ion.charge == 0
            else None
        )
        is_na_i_d_resonance = (
            ion.element == "Na"
            and ion.charge == 0
            and 5888.0 < center < 5901.0
            and from_ground
        )
        is_ca_i_resonance = (
            ion.element == "Ca" and ion.charge == 0 and 4215.0 < center < 4235.0
            and from_ground
        )
        is_ca_ii_resonance = (
            ion.element == "Ca" and ion.charge == 1 and 3920.0 < center < 3990.0
            and from_ground
        )
        is_ca_ii_measured_stark_line = (
            ion.element == "Ca"
            and ion.charge == 1
            and (
                3153.0 < center < 3167.0
                or 3173.0 < center < 3187.0
                or 3700.0 < center < 3715.0
                or 3730.0 < center < 3745.0
                or 3925.0 < center < 3950.0
                or 3955.0 < center < 3985.0
            )
        )
        replacement_key = (
            "CaI" if is_ca_i_resonance and ca_i_he_profile_table is not None
            else "MgII" if is_mg_ii_resonance and mg_ii_he_profile_table is not None
            else "CaII" if is_ca_ii_resonance and ca_ii_he_profile_table is not None
            else None
        )
        if line.neutral_h_vdw_rate_coefficient_cm3_s is not None:
            # Kurucz gamma_w/nH and ABO FWHM/NH are tabulated at 10,000 K.
            # ABO supplies a line-specific exponent; legacy Kurucz records
            # retain the induced-dipole T^0.3 approximation. Rescale the
            # resulting H coefficient to the actual neutral perturber below.
            neutral_temperature_exponent = (
                0.3
                if line.neutral_h_vdw_temperature_exponent is None
                else line.neutral_h_vdw_temperature_exponent
            )
            hydrogen_impact_rate = (
                line.neutral_h_vdw_rate_coefficient_cm3_s
                * (atmosphere.temperature / 10_000.0)
                ** neutral_temperature_exponent
            )
            helium_impact_rate = neutral_impact_rate_from_hydrogen(
                ion,
                hydrogen_impact_rate,
                perturber_atomic_mass_u=_HELIUM_ATOMIC_MASS_U,
                perturber_static_polarizability_a3=(
                    _HELIUM_STATIC_POLARIZABILITY_A3
                ),
            )
            bulk_metal_impact_rates = {
                element: neutral_impact_rate_from_hydrogen(
                    ion,
                    hydrogen_impact_rate,
                    perturber_atomic_mass_u=ATOMIC_MASS_U[element],
                    perturber_static_polarizability_a3=(
                        _BULK_METAL_STATIC_POLARIZABILITY_A3[element]
                    ),
                )
                for element in bulk_neutral_perturber_density
            }
        else:
            unsold_key = (
                ion.element,
                ion.charge,
                line.lower_index,
                line.upper_index,
            )
            if unsold_key not in atomic_database._unsold_hydrogen_coefficient_cache:
                atomic_database._unsold_hydrogen_coefficient_cache[unsold_key] = (
                    _unsold_hydrogen_temperature_coefficient(
                        ion,
                        line,
                        atomic_database.ions.get((ion.element, ion.charge + 1)),
                    )
                )
            unsold_coefficient = (
                atomic_database._unsold_hydrogen_coefficient_cache[unsold_key]
            )
            hydrogen_impact_rate = (
                None
                if unsold_coefficient is None
                else unsold_coefficient * atmosphere.temperature**0.3
            )
            helium_impact_rate = (
                None
                if hydrogen_impact_rate is None
                else neutral_impact_rate_from_hydrogen(
                    ion,
                    hydrogen_impact_rate,
                    perturber_atomic_mass_u=_HELIUM_ATOMIC_MASS_U,
                    perturber_static_polarizability_a3=(
                        _HELIUM_STATIC_POLARIZABILITY_A3
                    ),
                )
            )
            bulk_metal_impact_rates = (
                {
                    element: None
                    for element in bulk_neutral_perturber_density
                }
                if hydrogen_impact_rate is None
                else {
                    element: neutral_impact_rate_from_hydrogen(
                        ion,
                        hydrogen_impact_rate,
                        perturber_atomic_mass_u=ATOMIC_MASS_U[element],
                        perturber_static_polarizability_a3=(
                            _BULK_METAL_STATIC_POLARIZABILITY_A3[element]
                        ),
                    )
                    for element in bulk_neutral_perturber_density
                }
            )
        if is_na_i_d_resonance and "Ne" in bulk_neutral_perturber_density:
            # Use the measured Na--Ne impact core rather than inferring it
            # from an H coefficient and the static polarizability.  Other
            # neutral perturbers retain their ordinary line-specific rates.
            bulk_metal_impact_rates["Ne"] = (
                na_i_d_neon_impact_rate_coefficient(
                    center, atmosphere.temperature
                )
            )
        gaussian_sigma = center * np.sqrt(
            BOLTZMANN * atmosphere.temperature
            / (ion.atomic_mass_u * atomic_mass_unit * LIGHT_SPEED**2)
            + 0.5 * (microturbulent_velocity_kms * 1.0e5 / LIGHT_SPEED) ** 2
        )
        if is_mg_resonance:
            collision_rate = (
                0.306e-9 * helium_density * atmosphere.temperature**0.39
            )
        elif is_mg_ii_resonance:
            collision_rate = (
                mg_ii_helium_impact_rate_coefficient(center, atmosphere.temperature)
                * helium_density
            )
        elif is_ca_ii_resonance:
            collision_rate = (
                ca_ii_helium_impact_scale
                *
                ca_ii_helium_impact_rate_coefficient(
                    center, atmosphere.temperature
                ) * helium_density
            )
        elif helium_impact_rate is not None:
            collision_rate = helium_impact_rate * helium_density
        else:
            collision_rate = (
                helium_impact_rate_coefficient * helium_density
                * (atmosphere.temperature / 10_000.0) ** 0.3
            )
        helium_collision_rate = np.array(collision_rate, dtype=np.float64)
        if hydrogen_impact_rate is not None:
            collision_rate += hydrogen_impact_rate * hydrogen_density
        else:
            collision_rate += (
                hydrogen_impact_rate_coefficient * hydrogen_density
                * (atmosphere.temperature / 10_000.0) ** 0.3
            )
        for element, density in bulk_neutral_perturber_density.items():
            rate = bulk_metal_impact_rates[element]
            if rate is not None:
                collision_rate += rate * density
        if is_ca_ii_measured_stark_line:
            electron_stark_rate = (
                ca_ii_electron_stark_rate_coefficient(
                    center, atmosphere.temperature
                ) * metal_state.electron_density
            )
        elif is_mg_ii_4481:
            electron_stark_rate = (
                mg_ii_4481_electron_stark_rate_coefficient(
                    center, atmosphere.temperature
                ) * metal_state.electron_density
            )
        elif mg_ii_series_rate is not None:
            electron_stark_rate = mg_ii_series_rate * metal_state.electron_density
        elif is_mg_i_3835:
            electron_stark_rate = (
                mg_i_3835_electron_stark_rate_coefficient(
                    center, atmosphere.temperature
                ) * metal_state.electron_density
            )
        elif mg_i_optical_stark_rate is not None:
            electron_stark_rate = (
                mg_i_optical_stark_rate * metal_state.electron_density
            )
        elif na_i_optical_stark_rate is not None:
            electron_stark_rate = (
                na_i_optical_stark_rate * metal_state.electron_density
            )
        elif oxygen_optical_stark_rate is not None:
            electron_stark_rate = (
                oxygen_optical_stark_rate * metal_state.electron_density
            )
        elif oxygen_series_stark_rate is not None:
            electron_stark_rate = (
                oxygen_series_stark_rate * metal_state.electron_density
            )
        elif line.electron_stark_rate_coefficient_cm3_s is not None:
            electron_stark_rate = (
                line.electron_stark_rate_coefficient_cm3_s
                * metal_state.electron_density
                * (atmosphere.temperature / 10_000.0) ** (1.0 / 6.0)
            )
        elif classical_stark_rate > 0.0:
            electron_stark_rate = (
                classical_stark_rate * metal_state.electron_density
            )
        else:
            electron_stark_rate = np.zeros(atmosphere.n_depth)
        lorentz_hwhm = (
            center_cm**2
            * (natural_rate + collision_rate + electron_stark_rate)
            / (4.0 * PI * LIGHT_SPEED) * 1.0e8
        )
        population_factor = (
            lower_population * stimulated / atmosphere.mass_density
            * line_survival
        )
        if profile_edge_optical_depth is not None and (
            (
                profile_support_ions is not None
                and (ion.element, ion.charge) in profile_support_ions
            )
            or (
                profile_support_ions is None
                and (
                    profile_support_elements is None
                    or ion.element in profile_support_elements
                )
            )
        ):
            resonance_half_window = max(
                resonance_half_window,
                optically_thick_line_minimum_half_window_angstrom(
                    atmosphere,
                    center,
                    integrated_cross_section
                    * line.absorption_oscillator_strength,
                    gaussian_sigma,
                    lorentz_hwhm,
                    population_factor,
                    initial_half_window_angstrom=resonance_half_window,
                    maximum_rosseland_optical_depth=(
                        profile_support_maximum_rosseland_optical_depth
                    ),
                    edge_optical_depth=profile_edge_optical_depth,
                    maximum_half_window_angstrom=(
                        profile_support_maximum_half_window_angstrom
                    ),
                ),
            )
        stark_mixing = None
        if include_linear_stark_quasistatic:
            rydberg = linear_stark_rydberg_level(atomic_database, ion, upper_level)
            if rydberg is not None:
                candidate = linear_stark_mixing_fraction(
                    rydberg[2], rydberg[3], ion.charge + 1.0, ionic_microfield_scale
                )
                if np.max(candidate) > 1.0e-3:
                    stark_mixing = candidate
        quasistatic_series_rate = (
            o_i_3p5p_nd5d_electron_stark_rate_coefficient(
                ion,
                line,
                atmosphere.temperature,
                minimum_effective_n=oxygen_i_quasistatic_minimum_effective_n,
            )
            if include_oxygen_i_quasistatic_microfields else None
        )
        if quasistatic_series_rate is not None or stark_mixing is not None:
            core_charge = ion.charge + 1.0
            lower_binding_ev = (
                ion.ionization_energy_ev
                - lower_level.energy_wavenumber / EV_TO_WAVENUMBER
            )
            upper_binding_ev = (
                ion.ionization_energy_ev
                - upper_level.energy_wavenumber / EV_TO_WAVENUMBER
            )
            lower_effective_n = core_charge * np.sqrt(
                _RYDBERG_ENERGY_EV / lower_binding_ev
            )
            upper_effective_n = core_charge * np.sqrt(
                _RYDBERG_ENERGY_EV / upper_binding_ev
            )
            stark_sum = (
                upper_effective_n * (upper_effective_n - 1.0)
                + lower_effective_n * (lower_effective_n - 1.0)
            )
            frequency_scale = (
                stark_sum * ionic_microfield_scale / 1.385 / core_charge
            )
            static_half_window = (
                30.0 * frequency_scale * center_cm**2 / LIGHT_SPEED * 1.0e8
            )
            upper_manifold = lower_manifold = None
            if (
                quasistatic_series_rate is None
                and linear_stark_profile == "manifold"
            ):
                upper_manifold = rydberg_stark_manifold(
                    atomic_database, ion, upper_level
                )
                lower_manifold = rydberg_stark_manifold(
                    atomic_database, ion, lower_level
                )
                microfield_correlation = (
                    np.clip(
                        0.09 * np.maximum(ionic_microfield_charge_sum, 0.0) ** (1.0 / 6.0)
                        / np.sqrt(atmosphere.temperature),
                        0.0,
                        0.8,
                    )
                    if rydberg_correlated_microfields
                    else np.zeros(atmosphere.n_depth)
                )
                upper_critical_beta = hydrogenic_critical_microfield_beta(
                    np.maximum(ionic_microfield_charge_sum, np.finfo(np.float64).tiny),
                    max(1.0, float(upper_effective_n)),
                    core_charge,
                )
                if upper_manifold is not None:
                    # No bound-state component lies beyond the largest shift at
                    # the critical field; 30 impact widths cover the Voigt
                    # wings of the outermost components.
                    critical_coupling = np.log(np.maximum(
                        upper_critical_beta * ionic_microfield_scale
                        * _STARK_COUPLING_PER_MICROFIELD_SCALE,
                        _STARK_MANIFOLD_FIELD_HZ[0],
                    ))
                    log_field = np.log(_STARK_MANIFOLD_FIELD_HZ)
                    maximum_shift = np.interp(
                        critical_coupling, log_field, upper_manifold.maximum_shift_hz
                    )
                    if lower_manifold is not None:
                        maximum_shift = maximum_shift + np.interp(
                            critical_coupling, log_field, lower_manifold.maximum_shift_hz
                        )
                    static_half_window = (
                        1.02 * maximum_shift * center_cm**2 / LIGHT_SPEED * 1.0e8
                        + 30.0 * np.maximum(gaussian_sigma, lorentz_hwhm)
                    )
            for depth in range(atmosphere.n_depth):
                local_scale = float(frequency_scale[depth])
                sigma = float(gaussian_sigma[depth])
                hwhm = float(lorentz_hwhm[depth])
                mixing = (
                    1.0 if stark_mixing is None or quasistatic_series_rate is not None
                    else float(stark_mixing[depth])
                )
                half_window = max(
                    METAL_LINE_MINIMUM_HALF_WINDOW_ANGSTROM,
                    10.0 * sigma,
                    100.0 * hwhm,
                    float(static_half_window[depth]),
                    float(resonance_half_window),
                )
                start = int(np.searchsorted(wavelength, center - half_window))
                stop = int(np.searchsorted(
                    wavelength, center + half_window, side="right"
                ))
                if stop - start < 2:
                    continue
                selected_wavelength = wavelength[start:stop]

                if upper_manifold is not None:
                    combined_profile = manifold_quasistatic_line_profile(
                        selected_wavelength, center, sigma, hwhm,
                        float(ionic_microfield_scale[depth])
                        * _STARK_COUPLING_PER_MICROFIELD_SCALE,
                        upper_manifold, lower_manifold,
                        float(upper_critical_beta[depth]),
                        float(microfield_correlation[depth]),
                        core_charge,
                        half_window,
                    )
                    # Normalized analytically to unit bound strength.
                    area = 1.0
                else:
                    combined_profile = _quasistatic_line_profile(
                        selected_wavelength, center, sigma, hwhm, local_scale, mixing,
                        half_window,
                    )
                    # The normalization is a property of the line, not of the
                    # requested wavelengths: integrate over its full support.
                    area = _quasistatic_line_profile_area(
                        center, sigma, hwhm, local_scale, mixing, half_window
                    )
                if not np.isfinite(area) or area <= 0.0:
                    continue
                selected_cm = selected_wavelength * 1.0e-8
                result[start:stop, depth] += (
                    integrated_cross_section
                    * line.absorption_oscillator_strength
                    / area
                    * combined_profile
                    * 1.0e8 * selected_cm**2 / LIGHT_SPEED
                    * population_factor[depth]
                )
        elif is_mg_resonance and mg_he_red_wing_table is not None:
            # The unified red wing must be enveloped depth by depth, so retain
            # this rare special line outside the ordinary compiled batch.
            for depth in range(atmosphere.n_depth):
                half_window = max(
                    METAL_LINE_MINIMUM_HALF_WINDOW_ANGSTROM,
                    resonance_half_window,
                    10.0 * gaussian_sigma[depth],
                    100.0 * lorentz_hwhm[depth],
                )
                start = int(np.searchsorted(wavelength, center - half_window))
                stop = int(np.searchsorted(
                    wavelength, center + half_window, side="right"
                ))
                ordinary_cross_section = np.empty(0, dtype=np.float64)
                if stop > start:
                    profile_lambda = _voigt_profile_per_angstrom(
                        wavelength[start:stop],
                        center,
                        float(gaussian_sigma[depth]),
                        float(lorentz_hwhm[depth]),
                    )
                    ordinary_cross_section = (
                        integrated_cross_section
                        * line.absorption_oscillator_strength
                        * profile_lambda * 1.0e8 * center_cm**2 / LIGHT_SPEED
                    )
                allard_cross_section = mg_he_red_wing_table.cross_section(
                    wavelength,
                    atmosphere.temperature[depth],
                    helium_density[depth],
                ).reshape(wavelength.size)
                valid_temperature = (
                    mg_he_red_wing_table.temperatures[0]
                    <= atmosphere.temperature[depth]
                    <= mg_he_red_wing_table.temperatures[-1]
                )
                if valid_temperature:
                    result[:, depth] += (
                        allard_cross_section * population_factor[depth]
                    )
                if stop > start and valid_temperature:
                    result[start:stop, depth] += (
                        np.maximum(
                            ordinary_cross_section,
                            allard_cross_section[start:stop],
                        ) - allard_cross_section[start:stop]
                    ) * population_factor[depth]
                elif stop > start:
                    result[start:stop, depth] += (
                        ordinary_cross_section * population_factor[depth]
                    )
        else:
            batch_key = "result" if replacement_key is None else replacement_key
            batch = ordinary_profile_batches[batch_key]
            batch["center"].append(center)
            batch["integrated_strength"].append(
                integrated_cross_section * line.absorption_oscillator_strength
            )
            batch["gaussian_sigma"].append(gaussian_sigma)
            batch["lorentz_hwhm"].append(lorentz_hwhm)
            batch["minimum_half_window"].append(resonance_half_window)
            batch["population_scale"].append(population_factor)
            if len(batch["center"]) >= 2_048:
                flush_profile_batch(batch_key)
            if replacement_key == "CaI":
                reference_scale = np.minimum(
                    helium_density / ca_i_reference_density, 1.0
                )
                helium_hwhm_per_density = np.divide(
                    center_cm**2 * helium_collision_rate
                    / (4.0 * PI * LIGHT_SPEED) * 1.0e8,
                    helium_density,
                    out=np.zeros(atmosphere.n_depth),
                    where=helium_density > 0.0,
                )
                reference_batch = ordinary_profile_batches["CaI_reference"]
                reference_batch["center"].append(center)
                reference_batch["integrated_strength"].append(
                    integrated_cross_section * line.absorption_oscillator_strength
                )
                reference_batch["gaussian_sigma"].append(gaussian_sigma)
                reference_batch["lorentz_hwhm"].append(
                    lorentz_hwhm
                    + helium_hwhm_per_density
                    * np.maximum(ca_i_reference_density - helium_density, 0.0)
                )
                reference_batch["minimum_half_window"].append(resonance_half_window)
                reference_batch["population_scale"].append(
                    np.where(reference_scale < 1.0, population_factor * reference_scale, 0.0)
                )
    for batch_key in ordinary_profile_batches:
        flush_profile_batch(batch_key)
    if mg_ii_he_profile_table is not None and "Mg" in line_abundances:
        mg_ii = atomic_database.ions.get(("Mg", 1))
        if mg_ii is not None and metal_state.ion_number_density["Mg"].shape[0] > 1:
            ground = min(mg_ii.levels, key=lambda level: level.energy_wavenumber)
            partition = metal_state.partition_function[("Mg", 1)]
            ground_population = (
                metal_state.ion_number_density["Mg"][1]
                * ground.statistical_weight
                * np.exp(
                    -ground.energy_wavenumber * WAVENUMBER_TO_ERG
                    / (BOLTZMANN * atmosphere.temperature)
                )
                / partition
            )
            profile_cross_section = mg_ii_he_profile_table.cross_section(
                wavelength, helium_density
            )
            stimulated = -np.expm1(
                -PLANCK * LIGHT_SPEED
                / (
                    wavelength[:, np.newaxis] * 1.0e-8
                    * BOLTZMANN * atmosphere.temperature[np.newaxis, :]
                )
            )
            unified_opacity = (
                profile_cross_section
                * ground_population[np.newaxis, :]
                * stimulated
                / atmosphere.mass_density[np.newaxis, :]
            )
            # The public bridge is recovered from a plotted curve.  Its
            # strongest core points are clipped by the figure axes, and a
            # single-condition profile scaled to a much lower density can
            # also fall below the ordinary impact core.  Preserve the larger
            # opacity everywhere instead of erasing a valid ordinary core
            # merely because its wavelength lies inside the figure panel.
            combined = np.maximum(
                unified_opacity, replacement_ordinary["MgII"]
            )
            result += combined
    if ca_i_he_profile_table is not None and "Ca" in line_abundances:
        ca_i = atomic_database.ions.get(("Ca", 0))
        if ca_i is not None:
            ground = min(ca_i.levels, key=lambda level: level.energy_wavenumber)
            partition = metal_state.partition_function[("Ca", 0)]
            ground_population = (
                metal_state.ion_number_density["Ca"][0]
                * ground.statistical_weight
                * np.exp(
                    -ground.energy_wavenumber * WAVENUMBER_TO_ERG
                    / (BOLTZMANN * atmosphere.temperature)
                )
                / partition
            )
            profile_cross_section = ca_i_he_profile_table.cross_section(
                wavelength,
                helium_density,
                temperature=atmosphere.temperature,
            )
            stimulated = -np.expm1(
                -PLANCK * LIGHT_SPEED
                / (
                    wavelength[:, np.newaxis] * 1.0e-8
                    * BOLTZMANN * atmosphere.temperature[np.newaxis, :]
                )
            )
            unified_opacity = (
                profile_cross_section
                * ground_population[np.newaxis, :]
                * stimulated
                / atmosphere.mass_density[np.newaxis, :]
            )
            ordinary_opacity = replacement_ordinary["CaI"]
            # Inside the published density range the unified profile is the
            # complete line, core included, so it replaces the impact profile.
            # Below the least-dense calculation the table is scaled linearly
            # (binary collisions).  That scaling is right for the wings but
            # not for the impact core, whose width shrinks with density.  The
            # line is then the local impact profile plus the part of the
            # unified profile beyond impact theory: unified(n_min) minus the
            # impact profile at n_min, both scaled by n/n_min.  This is
            # continuous at n_min and recovers impact theory as n -> 0.
            below_table = helium_density < ca_i_reference_density
            unified_with_core = np.where(
                below_table[np.newaxis, :],
                np.maximum(
                    unified_opacity
                    + ordinary_opacity
                    - replacement_ordinary["CaI_reference"],
                    0.0,
                ),
                unified_opacity,
            )
            if ca_i_he_profile_table.temperatures.size:
                valid_temperature = (
                    (atmosphere.temperature >= ca_i_he_profile_table.temperatures[0])
                    & (atmosphere.temperature <= ca_i_he_profile_table.temperatures[-1])
                )
                result += np.where(
                    valid_temperature[np.newaxis, :],
                    unified_with_core,
                    ordinary_opacity,
                )
            else:
                result += unified_with_core
    if ca_ii_he_profile_table is not None and "Ca" in line_abundances:
        ca_ii = atomic_database.ions.get(("Ca", 1))
        if ca_ii is not None and metal_state.ion_number_density["Ca"].shape[0] > 1:
            ground = min(ca_ii.levels, key=lambda level: level.energy_wavenumber)
            partition = metal_state.partition_function[("Ca", 1)]
            ground_population = (
                metal_state.ion_number_density["Ca"][1]
                * ground.statistical_weight
                * np.exp(
                    -ground.energy_wavenumber * WAVENUMBER_TO_ERG
                    / (BOLTZMANN * atmosphere.temperature)
                )
                / partition
            )
            profile_cross_section = ca_ii_he_profile_table.cross_section(
                wavelength,
                helium_density,
                temperature=atmosphere.temperature,
            )
            stimulated = -np.expm1(
                -PLANCK * LIGHT_SPEED
                / (
                    wavelength[:, np.newaxis] * 1.0e-8
                    * BOLTZMANN * atmosphere.temperature[np.newaxis, :]
                )
            )
            unified_opacity = (
                profile_cross_section
                * ground_population[np.newaxis, :]
                * stimulated
                / atmosphere.mass_density[np.newaxis, :]
            )
            # As for Mg II, these vector-recovered data are a clipped figure
            # bridge rather than a complete numerical profile grid.  An
            # envelope is the only safe merge at the clipped core and when
            # scaling away from the published density/temperature.
            combined = np.maximum(
                unified_opacity, replacement_ordinary["CaII"]
            )
            result += combined
    return result
