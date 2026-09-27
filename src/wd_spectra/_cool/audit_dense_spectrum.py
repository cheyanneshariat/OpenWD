"""Compare transfer discretizations on one fixed experimental dense-He state.

Never relax or normalize the saved atmosphere/spectrum. Independent wavelength
and angular quadratures separate source convergence from grid error.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from wd_spectra import helium
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra._compat import trapezoid
from wd_spectra.opacity import optical_depth_from_mass_opacity, electron_scattering_mass_coefficient
from wd_spectra.spectrum import planck_lambda_angstrom
from wd_spectra.radiative_transfer import radiation_field, emergent_flux
from wd_spectra._stable_feautrier import cancellation_safe_field, cancellation_safe_scalar_field


def coupled_linear_source(tau, planck, absorption, scattering, n_angle=4, chunk_size=16):
    """Direct Lambda solve using the public piecewise-linear formal operator.

    Basis sweeps are deliberately independent of Feautrier block elimination;
    this slower construction is a diagnostic, not a production implementation.
    """
    nw, nd = planck.shape
    source = np.empty_like(planck)
    eye = np.eye(nd)
    for start in range(0, nw, chunk_size):
        stop = min(start+chunk_size, nw)
        count = stop-start
        basis = np.broadcast_to(eye, (count, nd, nd)).reshape(count*nd, nd).copy()
        depths = np.repeat(tau[start:stop], nd, axis=0)
        response = radiation_field(depths, basis, n_angle=n_angle).mean_intensity
        lam = response.reshape(count, nd, nd).transpose(0, 2, 1)
        ext = absorption[start:stop]+scattering[start:stop]
        eps = absorption[start:stop]/ext
        frac = scattering[start:stop]/ext
        matrix = eye-frac[:, :, None]*lam
        # An explicit singleton RHS axis has identical vector semantics in
        # NumPy 1 and 2 (a 2-D batched RHS is interpreted differently in 2).
        source[start:stop] = np.linalg.solve(matrix, (eps*planck[start:stop])[..., None])[..., 0]
    return source


def pure_helium_opacities(atmosphere, wave):
    from wd_spectra.models import ModelData, DBConfig
    from wd_spectra.models.stellar import _helium_tables
    hi, hii = _helium_tables(ModelData.default())
    absorption = helium.helium_continuum_mass_absorption_coefficient(
        atmosphere, wave, include_electron_scattering=False, include_rayleigh_scattering=False)
    absorption += helium.helium_i_line_mass_absorption_coefficient(
        atmosphere, wave, hi, include_occupation_probability=True,
        neutral_broadening=DBConfig().neutral_broadening)
    absorption += helium.helium_i_resonance_line_mass_absorption_coefficient(
        atmosphere, wave, include_occupation_probability=True)
    absorption += helium.helium_ii_line_mass_absorption_coefficient(
        atmosphere, wave, stark_table=hii, include_occupation_probability=True)
    scattering = (helium.helium_rayleigh_scattering_mass_coefficient(atmosphere, wave)
                  + electron_scattering_mass_coefficient(atmosphere)[None, :])
    return absorption, scattering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--interaction-table', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wavelength-count', type=int, default=4000)
    parser.add_argument('--skip-linear', action='store_true')
    parser.add_argument('--angles', type=int, nargs='+', default=[3,4,8])
    parser.add_argument('--maximum-wavelength', type=float, default=1e7)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refusing to overwrite spectrum audit')
    args.output.mkdir(parents=True)
    from . import check_cool_db_transport_seed as runner
    from wd_spectra.models import ModelData
    from wd_spectra.dense_eos import read_helium_reos3_table
    from .dense_helium_molecular_experiment import MolecularDenseEOS
    from .dense_helium_atomic_experiment import atomic_dense_experiment
    from .dense_helium_fluid_experiment import KB_EV
    from .smooth_reos3_experiment import SmoothREOS3
    model = MolecularDenseEOS(SmoothREOS3(read_helium_reos3_table(
        ModelData.default().helium_reos3), .1/KB_EV, 17000), args.interaction_table)
    meta = json.loads((args.run_directory/'experimental-metadata.json').read_text())
    mass_conservative=bool(meta['atmosphere'].get('experimental_mass_conservative_transfer',False))
    feautrier_name='mass-coupled-feautrier' if mass_conservative else 'coupled-feautrier'
    if meta['interaction_table_sha256'] != model.sha256 or meta['physics'] != model.physics:
        raise ValueError('saved state physics do not match diagnostic model')
    with np.load(args.run_directory/'experimental-molecular-dense-structure.npz') as saved:
        temperature = saved['experimental_temperature']
        pressure = saved['experimental_pressure']
        tau = saved['experimental_tau']
        density = saved['experimental_density']
    teff = float(args.run_directory.name)
    wave = np.geomspace(100., args.maximum_wavelength, args.wavelength_count)
    from .heminus_join_experiment import heminus_join_scope,METADATA_KEY,HARD_JOIN
    join_policy=meta['atmosphere'].get(METADATA_KEY,HARD_JOIN)
    with heminus_join_scope(join_policy),atomic_dense_experiment(runner, model, opacity_factory=model.opacity_factory):
        atmosphere = runner.atmosphere_at(teff, pressure, temperature, tau)
        np.testing.assert_allclose(atmosphere.mass_density, density, rtol=1e-13)
        absorption, scattering = pure_helium_opacities(atmosphere, wave)
    depths = optical_depth_from_mass_opacity(atmosphere.column_mass, absorption+scattering)
    planck = planck_lambda_angstrom(wave[:, None], temperature[None, :])
    target = STEFAN_BOLTZMANN*teff**4
    records=[]
    def save(name, flux, source, field):
        closure = source-(absorption*planck+scattering*field.mean_intensity)/(absorption+scattering)
        # Unweighted relative errors can be enormous where both intensities
        # underflow in the Wien tail. Keep that diagnostic, and separately
        # report its magnitude relative to the global radiation scale.
        scale = float(np.max(wave[:, None]*source))
        row = dict(method=name, flux_ratio=float(trapezoid(flux, wave)/target),
                   maximum_relative_source_equation_error=float(np.max(abs(closure)/np.maximum(source, 1e-300))),
                   radiation_scale_source_equation_error=float(np.max(wave[:, None]*abs(closure))/scale))
        if name.startswith(feautrier_name):
            from .dense_thermal_verification import thermal_diagnostics
            conv=target*np.asarray(meta['atmosphere']['convective_flux_fraction_by_interface'])
            row.update(thermal_diagnostics(wave,depths,planck,field.mean_intensity,
                absorption/(absorption+scattering),conv,
                **(dict(column_mass=atmosphere.column_mass,absorption=absorption) if mass_conservative else {})))
            row['mass_conservative_transfer']=mass_conservative
        records.append(row)
        print(json.dumps(row), flush=True)
        np.savetxt(args.output/(name+'.txt'), np.column_stack((wave, flux)))
        (args.output/'summary.json').write_text(json.dumps(records, indent=2)+'\n')
    source = planck.copy()
    for _ in range(4):
        field = radiation_field(depths, source, n_angle=4)
        source = (absorption*planck+scattering*field.mean_intensity)/(absorption+scattering)
    save('linear-four-sweeps', emergent_flux(depths, source, n_angle=4), source,
         radiation_field(depths, source, n_angle=4))
    if (args.wavelength_count == 4000 and args.maximum_wavelength==1e7
            and not meta['atmosphere'].get('experimental_spectrum_transfer')):
        previous = np.loadtxt(args.run_directory/'experimental-spectrum.txt')
        np.testing.assert_allclose(previous[:, 0], wave, rtol=1e-14)
        np.testing.assert_allclose(previous[:, 1], emergent_flux(depths, source, n_angle=4), rtol=1e-12)
    for angles in args.angles:
        if mass_conservative:
            from .mass_conservative_feautrier import mass_field
            source,field=mass_field(depths,planck,absorption,scattering,column_mass=atmosphere.column_mass,n_angle=angles)
            _,independent=mass_field(depths,source,absorption+scattering,np.zeros_like(scattering),
                column_mass=atmosphere.column_mass,n_angle=angles)
        else:
            source, field = cancellation_safe_field(depths, planck, absorption, scattering, n_angle=angles)
            independent = cancellation_safe_scalar_field(depths, source, n_angle=angles)
        save(f'{feautrier_name}-{angles}', field.interface_flux[:, 0], source, independent)
    if not args.skip_linear:
        source = coupled_linear_source(depths, planck, absorption, scattering)
        save('coupled-linear-4', emergent_flux(depths, source, n_angle=4), source,
             radiation_field(depths, source, n_angle=4))
    # An audit qualifies a numerical result only after checking a broader,
    # independently sampled thermal interval. A spectrum integral alone is
    # not enough, and neither numerical gate validates missing physics.
    structure=meta['atmosphere']
    original_angles=structure.get('experimental_structure_n_angle',3)
    checks=[r for r in records if r['method'].startswith(feautrier_name)
            and int(r['method'].rsplit('-',1)[1])>=original_angles]
    original_maximum=structure.get('experimental_thermal_wavelength_maximum_angstrom',1e5)
    broader=bool(args.maximum_wavelength>=10*original_maximum)
    finer_angles=bool(max(args.angles)>original_angles)
    physical_flag=bool(structure['radiative_equilibrium_converged']
        and structure['maximum_all_depth_total_flux_residual']<.003
        and structure['maximum_relative_cell_energy_balance_residual']<.003
        and structure['radiative_equilibrium_maximum_log_temperature_correction']<.0003)
    thermal_pass=bool(checks and all(r['maximum_independent_cell_energy_defect']<.003 for r in checks))
    flux_pass=bool(checks and all(abs(r['flux_ratio']-1)<.003 for r in checks))
    source_pass=bool(checks and all(r['radiation_scale_source_equation_error']<1e-10 for r in checks))
    qualification=dict(numerically_qualified_for_declared_experimental_physics=(
        physical_flag and broader and finer_angles and thermal_pass and flux_pass and source_pass),
        qualification_scope='fixed-depth Feautrier atmosphere: independent wavelength/angular/source checks only',
        mass_conservative_transfer=mass_conservative,
        independent_depth_resolution_verified=False,
        structure_grid_converged=physical_flag,broader_thermal_interval_checked=broader,
        finer_angular_quadrature_checked=finer_angles,structure_angles=original_angles,
        independent_local_energy_verified=thermal_pass,independent_spectrum_flux_verified=flux_pass,
        independent_source_equation_verified=source_pass,validated_full_physics=False,
        structure_maximum_wavelength_angstrom=original_maximum,
        independent_maximum_wavelength_angstrom=args.maximum_wavelength,
        independent_wavelength_count=args.wavelength_count,independent_angles=args.angles,
        interaction_table_sha256=model.sha256)
    qualification[METADATA_KEY]=join_policy
    (args.output/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
    print(json.dumps(qualification),flush=True)


if __name__ == '__main__':
    main()
