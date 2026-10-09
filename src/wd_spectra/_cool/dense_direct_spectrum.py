"""Explicit experimental coupled synthesis with the structure's transfer law."""
from dataclasses import replace
import numpy as np
from wd_spectra._stable_feautrier import cancellation_safe_field,cancellation_safe_scalar_field
from wd_spectra.opacity import optical_depth_from_mass_opacity
from wd_spectra.spectrum import planck_lambda_angstrom
from .audit_dense_spectrum import pure_helium_opacities


def direct_result(result,n_angle,*,mass_conservative=False):
    atmosphere=result.atmosphere
    if 'experimental_dense_helium' not in atmosphere.metadata:
        raise ValueError('direct research synthesis requires explicitly experimental dense-helium metadata')
    from .heminus_join_experiment import METADATA_KEY,HARD_JOIN,active_policy
    if atmosphere.metadata.get(METADATA_KEY,HARD_JOIN)!=active_policy():
        raise ValueError('synthesis must retain the declared He-minus wavelength join')
    wave=result.spectrum.wavelength_angstrom
    absorption,scattering=pure_helium_opacities(atmosphere,wave)
    depths=optical_depth_from_mass_opacity(atmosphere.column_mass,absorption+scattering)
    planck=planck_lambda_angstrom(wave[:,None],atmosphere.temperature[None,:])
    declared=bool(atmosphere.metadata.get('experimental_mass_conservative_transfer',False))
    if declared!=mass_conservative:
        raise ValueError('synthesis must use the same declared mass/optical-depth transfer as the structure')
    if mass_conservative:
        from .mass_conservative_feautrier import mass_field
        source,field=mass_field(depths,planck,absorption,scattering,
            column_mass=atmosphere.column_mass,n_angle=n_angle,reconstruct_intensity=True)
        _,independent=mass_field(depths,source,absorption+scattering,np.zeros_like(scattering),
            column_mass=atmosphere.column_mass,n_angle=n_angle,reconstruct_intensity=True)
    else:
        source,field=cancellation_safe_field(depths,planck,absorption,scattering,n_angle=n_angle)
        independent=cancellation_safe_scalar_field(depths,source,n_angle=n_angle)
    # A prescribed-source solve checks transfer closure independently of
    # the coupled J that constructed S; the latter is an algebraic identity.
    closure=source-(absorption*planck+scattering*independent.mean_intensity)/(absorption+scattering)
    error=float(np.max(wave[:,None]*abs(closure))/np.max(wave[:,None]*source))
    if error>1e-10:
        raise RuntimeError('coupled experimental synthesis failed its source-equation check')
    method='mass-coupled-feautrier' if mass_conservative else 'coupled-feautrier'
    metadata=dict(experimental_spectrum_transfer=f'{method}-{n_angle}',
        experimental_positive_intensity_reconstruction=mass_conservative,
        experimental_spectrum_source_check='independent prescribed-source formal solve',
        experimental_spectrum_radiation_scale_source_error=error,
        experimental_ordinary_four_sweep_bolometric_flux=result.spectrum.bolometric_flux)
    spectrum=replace(result.spectrum,surface_flux_lambda=field.interface_flux[:,0],
        metadata={**result.spectrum.metadata,**metadata,'n_angle':int(n_angle),
            'scattering_source_solver':'direct coupled mass-volume Feautrier' if mass_conservative else 'direct coupled cancellation-safe Feautrier',
            'scattering_source_iterations':1,'scattering_source_converged':True})
    return replace(result,spectrum=spectrum,
        atmosphere=replace(atmosphere,metadata={**atmosphere.metadata,**metadata}),
        metadata={**result.metadata,**metadata})
