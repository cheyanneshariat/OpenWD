"""Independent spectrum gate, separate from the structure-grid certificate."""
import numpy as np
from wd_spectra.constants import STEFAN_BOLTZMANN


def independent_grid():
    # Keep the optical midpoint grid independent of structure sampling, and
    # resolve the infrared molecular spectrum too. Round to eight intervals
    # so R≈3000/6000/12000/24000 nested diagnostic meshes share endpoints.
    infrared_intervals = 8*int(np.ceil(3000*np.log(100000./6800.)))
    return np.unique(np.r_[np.geomspace(1000., 100000., 4000),
                           3800.+(np.arange(150000)+.5)*.02,
                           np.geomspace(6800., 100000., infrared_intervals+1)])


def qualify_spectrum(atmosphere, spectrum, config):
    certificate = atmosphere.metadata.get('equilibrium_certificate', {})
    checks = certificate.get('checks', {})
    required = {'all_depth_flux', 'local_energy', 'temperature_stationarity',
                'source_closure', 'boundary_screening'}
    # Shared certificates also contain optional diagnostics. Enforce DQ's
    # mandatory checks plus any additional requirements the certificate declares.
    required.update(certificate.get('required_checks', ()))
    if (not certificate.get('verified') or certificate.get('failures') or
            not required.issubset(checks) or
            not all(checks[name].get('passed') for name in required)):
        raise ValueError('DQ atmosphere certificate is incomplete or failed')
    limits = dict(all_depth_flux=.002, local_energy=.002,
                  temperature_stationarity=.0002, source_closure=1e-6,
                  boundary_screening=.002)
    for name, limit in limits.items():
        check = checks[name]
        value, tolerance = check.get('value'), check.get('tolerance')
        if (not check.get('measured') or value is None or tolerance is None or
                not np.isfinite(value) or not np.isfinite(tolerance) or
                not 0 < tolerance <= limit or not 0 <= value <= tolerance):
            raise ValueError(f'DQ atmosphere certificate has an invalid {name} check')
    if (atmosphere.effective_temperature != config.effective_temperature or
            atmosphere.logg != config.logg or
            atmosphere.metadata.get('carbon_abundance') != config.log_carbon_to_helium):
        raise ValueError('DQ atmosphere parameters changed')
    if not atmosphere.metadata.get('dq_refractive_transfer'):
        raise ValueError('DQ atmosphere did not use refractive transfer')
    if not np.array_equal(spectrum.wavelength_angstrom, independent_grid()):
        raise ValueError('DQ qualification requires the exact independent optical/infrared grid')
    flux = spectrum.surface_flux_lambda
    if np.any(~np.isfinite(flux)) or np.any(flux <= 0):
        raise ValueError('DQ final flux must be finite and positive')
    ratio = float(spectrum.bolometric_flux/(STEFAN_BOLTZMANN*config.effective_temperature**4))
    if not np.isfinite(ratio) or abs(ratio-1.) > .002:
        raise ValueError(f'DQ independent bolometric error {ratio-1.:+.6g} exceeds 0.002')
    return dict(independent_spectrum_flux_ratio=ratio,
                independent_spectrum_points=len(flux), spectral_qualification=True)
