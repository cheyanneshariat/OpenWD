"""Published 2024 C2 C–A band strengths (no fitted scale).

Lino da Silva 2024 Einstein coefficients, with ExoMol lower-state populations
and the unchanged 8states partition. Historical band origins are retained.
The finite-bin rigid-rotor envelope is approximate, not a modern line list.
No Swan pressure-shift law is silently assigned to this different system.
"""
import numpy as np
from wd_spectra.carbon_molecular import read_c2_cross_section_table
from .data import data_root


def load_ca_table(parent):
    table = read_c2_cross_section_table(data_root()/'c2-ca-2024.npz')
    for key in ('partition_temperature_K', 'partition_function'):
        if not np.array_equal(getattr(table, key), getattr(parent, key)):
            raise ValueError('C–A and chemistry partition conventions differ')
    return table
