"""Reproduce ``_O_I_EXCITED_CORE_3D_TRANSITIONS`` (requires sympy).

Spectator-core LS strengths for O I 2p3(2D)3p 3D, 3F -> 2p3(2D)4d, 5s 3Do,
normalized to Opacity Project (TOPbase) LS multiplets of the same transition
arrays between bound terms.  The OP values were retrieved on 2026-10-25 from
https://cdsweb.u-strasbg.fr/cgi-bin/topbase/topbase.sh (ent=f, NZ=NE=8,
gf and wavelength columns); level energies are NIST ASD O I values (identical
to the bundled Stout levels).  4d' 3Fo J=2, 3 have no observed energies and
are not included.

    python research/o_i_excited_core_ls_strengths.py
"""
from sympy.physics.wigner import wigner_6j
import numpy as np

CORE_L = 2
# (lower L, lower 2S+1, upper L), OP multiplet gf and OP wavelength (A).
OP_3P_4D = {(2, 3, 1): (0.124, 6153.0), (2, 3, 3): (0.614, 6274.0)}  # 3p' 3D -> 4d' 3Po, 3Fo
OP_3P_5S = {(1, 1, 2): (0.0440, 6416.0), (3, 1, 2): (0.118, 6745.0)}  # 3p' 1P, 1F -> 5s' 1Do
LEVELS = {  # NIST energies (cm^-1) by J
    "3p 3D": {3: 113294.816, 2: 113294.854, 1: 113298.320},
    "3p 3F": {4: 113714.444, 3: 113721.413, 2: 113727.165},
    "4d 3Do": {3: 129692.300, 2: 129697.500, 1: 129703.000},
    "5s 3Do": {3: 128978.800, 2: 128990.200, 1: 128998.200},
}
TERM_L = {"3p 3D": 2, "3p 3F": 3, "4d 3Do": 2, "5s 3Do": 2}


def multiplet_strength(lower_l, upper_l, l, lp):
    return float((2 * lower_l + 1) * (2 * upper_l + 1)
                 * wigner_6j(l, lower_l, CORE_L, upper_l, lp, 1) ** 2)


def radial_factor(table, l, lp):
    values = [gf * wavelength / multiplicity / multiplet_strength(lower_l, upper_l, l, lp)
              for (lower_l, multiplicity, upper_l), (gf, wavelength) in table.items()]
    return float(np.mean(values)), float(np.std(values) / np.mean(values))


def main():
    radial = {"4d 3Do": radial_factor(OP_3P_4D, 1, 2), "5s 3Do": radial_factor(OP_3P_5S, 1, 0)}
    orbital = {"4d 3Do": (1, 2), "5s 3Do": (1, 0)}
    for upper, (value, spread) in radial.items():
        print(f"# 3p -> {upper.split()[0]} radial factor {value:.0f} (OP spread {100 * spread:.0f}%)")
    for upper in ("4d 3Do", "5s 3Do"):
        for lower in ("3p 3D", "3p 3F"):
            l, lp = orbital[upper]
            lower_l, upper_l = TERM_L[lower], TERM_L[upper]
            per_spin = multiplet_strength(lower_l, upper_l, l, lp) * radial[upper][0]
            for j, lower_energy in LEVELS[lower].items():
                for jp, upper_energy in LEVELS[upper].items():
                    weight = float((2 * j + 1) * (2 * jp + 1)
                                   * wigner_6j(lower_l, 1, j, jp, 1, upper_l) ** 2)
                    wavelength = 1.0e8 / (upper_energy - lower_energy)
                    f = weight * per_spin / wavelength / (2 * j + 1)
                    if f >= 1.0e-4:
                        print(f"    ({lower_energy:.3f}, {upper_energy:.3f}, {f:.4e}),"
                              f"  # {wavelength:.2f} {lower[3:]}{j}-{upper}{jp}")


if __name__ == "__main__":
    main()
