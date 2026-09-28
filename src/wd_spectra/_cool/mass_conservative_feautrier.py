"""Compatibility imports for existing research runs; one canonical operator.

New solver calls select ``transfer_discretization="column-mass"`` explicitly.
The historical optical-depth discretization remains the production default.
"""
from wd_spectra._mass_feautrier import (
    MassFactors, mass_energy, mass_energy_response, mass_field, mass_response,
    mass_width, validate_chunk,
)
