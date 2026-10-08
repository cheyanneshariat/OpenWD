"""Run the existing strict local-energy experiment with explicit molecular physics.

All atmosphere/EOS/opacity calls use the same molecular closure. Command-line
arguments are those of check_cool_db_transport_seed.py; a supplied atmosphere
is explicitly re-solved, never represented as a cold molecular calculation.
"""
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch
from . import check_cool_db_transport_seed as runner
from wd_spectra.models.stellar import _mixed_molecular_data
from wd_spectra.models import ModelData

from .research_paths import data_directory
CIA=data_directory()/'H2-He_2011.cia'


def physics():
    return _mixed_molecular_data(str(CIA),str(ModelData.default().h2_h2_cia))


def main():
    molecule=physics()
    original_compute=runner.compute_dab
    def compute(config,*args,**kwargs):
        return original_compute(replace(config,include_molecules=True,h2_he_cia_path=str(CIA)),*args,**kwargs)
    original_save=runner.save_model_result
    def save(result,*args,**kwargs):
        a=replace(result.atmosphere,metadata={**result.atmosphere.metadata,
            'diagnostic_molecular_h_he':True,
            'molecular_physics_limitations':'No nonideal dissociation or pressure-distorted CIA; no HeH+ or He2+ charge.'})
        return original_save(replace(result,atmosphere=a),*args,**kwargs)
    with ExitStack() as stack:
        replacements=dict(
            hummer_mihalas_hydrogen_helium_lte=molecule.lte,
            hummer_mihalas_hydrogen_helium_thermodynamics=molecule.thermodynamics,
            load_atmosphere_checkpoint=partial(runner.load_atmosphere_checkpoint,include_molecules=True),
            hydrogen_helium_continuum_mass_absorption_coefficient=partial(
                runner.hydrogen_helium_continuum_mass_absorption_coefficient,molecular_h_he=molecule),
            rosseland_mean_hydrogen_helium_continuum_opacity=partial(
                runner.rosseland_mean_hydrogen_helium_continuum_opacity,molecular_h_he=molecule),
            radiative_equilibrium_hydrogen_helium_atmosphere=partial(
                runner.radiative_equilibrium_hydrogen_helium_atmosphere,molecular_h_he=molecule),
            compute_dab=compute,save_model_result=save)
        for name,value in replacements.items(): stack.enter_context(patch.object(runner,name,value))
        runner.main()


if __name__=='__main__': main()
