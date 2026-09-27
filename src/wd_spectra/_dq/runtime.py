"""The current-energy, corner-aware, continuum-refined cold DQ protocol.

Adapters are process-local and are used only inside an isolated worker.
There is no restart, reference atmosphere, fixed structure grid, fitted
parameter, reduced-line experiment, or alternate-physics fallback here.
"""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import gc
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from wd_spectra._domain import append_lower_domain, solve_with_screened_boundary
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.models.common import ModelData, ModelResult, save_model_result
from . import base, automatic_conditioning as controller, dq_explicit_gradient
from .data import data_root, validate_data
from .c2_ca import load_ca_table
from .swan_strength_cache import CachedSwan
from .continuum_batches import CONTINUUM_BATCH_SIZE
from .dq_current_energy_experiment import current_energy_phase_rows
from .dq_thermal_corner_experiment import corner_aware_thermal_proposals
from .dq_uv_sampling_experiment import refined_material
from .inexact_thermal import error_controlled_thermal_steps
from .planck_remainder import nonlinear_planck_trials
from .provenance import digest, source_hashes
from .superadiabatic import SuperadiabaticSystem
from .validation import independent_grid, qualify_spectrum

PROTOCOL = 'refractive-current-energy-ca2024-swan-complete-stride4-v4'


@contextmanager
def numerical_policy(output):
    """Compose the successful J1235/J0752 policy once, without launchers."""
    original_condition = controller.thermal_condition
    records = []

    def bounded_condition(system, state, tolerance, emit, maximum_steps=30):
        return original_condition(system, state, tolerance, emit, maximum_steps,
                                  handoff_at_budget=True)

    def emit(row):
        records.append(row)
        (output/'thermal-proposals.json').write_text(json.dumps(records, indent=2)+'\n')

    # Keep the original wrapper ordering. In particular, the Planck local
    # model wraps the corner observer, and transient tolerance capture wraps
    # the bounded conditioning helper, not the other way round.
    with current_energy_phase_rows(), corner_aware_thermal_proposals(emit) as release_proposals, \
            controller.thermal_relative_norm_limit(.8), \
            patch.object(controller, 'thermal_condition', bounded_condition), \
            error_controlled_thermal_steps(predict_exhaustion=True), \
            nonlinear_planck_trials():
        yield release_proposals


def material_class(output, *, structure_stride=4):
    """Construct after entering numerical_policy so coordinate selection is explicit."""
    class ColdDQ(dq_explicit_gradient.gradient_material(None, system_class=SuperadiabaticSystem)):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.experiment_metadata.update(
                protocol=PROTOCOL,
                refined_convection_proposal=False,
                nonlinear_material_proposal=True,
                nonlinear_material_variable_scaling='unit native coordinates',
                thermal_relative_norm_limit=.8,
                thermal_eos_corner_repair=True,
                energy_equations='integral flux minus local Q/current thermal-plus-convective scale',
                energy_row_equilibration='current state; full quotient-rule tangent',
                current_energy_experiment=True,
                thermal_initialization='bounded pseudo-time helper; no pre-equilibrium gate',
                convection_coordinates='log T and superadiabatic excess in fixed local ML2 units',
                fixed_wavelength_grid=None,
                domain_extension='one canonical cell; at most eight same-run extensions',
                full_physics_validated=False,
                carbon_uv_support='complete-atmosphere resonance gate; depth-batch independent',
                carbon_line_evaluation='union screening with conservative opacity bound',
            )

        def solve(self, *args, **kwargs):
            # Structure iterations retain large response matrices. Keep one
            # exact-temperature entry there; restore the full synthesis cache
            # only after solve-owned arrays are released by the caller.
            cache = self.integrated_swan
            budget = cache.strength_cache_bytes if isinstance(cache, CachedSwan) else None
            if budget is not None:
                cache.set_strength_cache_bytes(min(budget, 32*1024**2))
            try:
                with controller.automatic_material_trials(output):
                    return super().solve(*args, **kwargs)
            finally:
                if budget is not None:
                    cache.set_strength_cache_bytes(budget)

    return refined_material(ColdDQ, 4000, full_continuum=True,
                            structure_stride=structure_stride)


def make_material(config, data, output, *, structure_stride=4):
    validate_data()
    root = data_root()
    physical = base.DQConfig(
        effective_temperature=config.effective_temperature,
        logg=config.logg, log_carbon_to_helium=config.log_carbon_to_helium,
        quality='standard', c2_table_path=str(root/'c2-8states-r15000.npz'),
        helium_eos='reos3', swan_pressure_shift='blouin2019',
        helium_dense_continuum_path=str(root/'correction.npz'))
    table = base.read_c2_cross_section_table(physical.c2_table_path)
    material = material_class(output, structure_stride=structure_stride)(
        physical, data, table, sampling_r=10000., sampling_phase=.5)
    material.integrated_swan = CachedSwan(table, branches=root/'swan-completed.npz',
                                         cache_grids=192, strength_cache_bytes=1024**3)
    if config.include_c2_ca:
        material.ca_table = load_ca_table(table)
    material.experiment_metadata.update(
        c2_ca_included=config.include_c2_ca,
        c2_ca_profile='historical finite-bin rigid-rotor envelope; unshifted',
        c2_ca_strength='Lino da Silva 2024 Einstein coefficients; no fitted multiplier',
        swan_coverage='Hornkohl retained; ExoMol outside per-band v/J envelope',
        swan_line_count=len(material.integrated_swan.nu),
        swan_strength_cache='exact floating-point temperature; byte-bounded LRU',
        swan_strength_cache_bytes=material.integrated_swan.strength_cache_bytes,
        swan_structure_strength_cache_bytes=32*1024**2,
        helium_continuum_batch_size=CONTINUUM_BATCH_SIZE,
        transfer_kernel='allocation-free scalar ray and analytic-response loops')
    return material


def run_cold(config, output, *, data=None, wavelength=None):
    """Fresh atmosphere, certified structure, then independent fine spectrum."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frozen = source_hashes()
    params = {key: getattr(config, key) for key in
              ('effective_temperature', 'logg', 'log_carbon_to_helium')}
    report = dict(schema=1, protocol=PROTOCOL,
        status='running', requested_parameters=params, config=asdict(config),
        cold_start=True, external_atmosphere=None, external_structure_grid=None,
        prior_spectrum=None, opacity_scale=1., source_sha256=frozen,
        started_utc=datetime.now(timezone.utc).isoformat(), process_id=os.getpid(),
        independent_depth_convergence=False, full_physics_validation=False)

    def record():
        report['seconds'] = time.monotonic()-started
        (output/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    def checkpoint(path, atmosphere):
        np.savez_compressed(path, **{key: getattr(atmosphere, key) for key in
            ('temperature', 'gas_pressure', 'column_mass', 'rosseland_optical_depth')})

    record()
    try:
        with numerical_policy(output) as release_proposals:
            material = make_material(config, ModelData.default() if data is None else data, output)
            material.deadline = started+config.maximum_seconds
            report['constitutive_data'] = validate_data()
            report['physical_config'] = asdict(material.config)
            report['physics_metadata'] = material.experiment_metadata

            def progress(i, atmosphere, diagnostics):
                row = dict(iteration=i, depths=atmosphere.n_depth,
                    phase=diagnostics.get('solver_phase'),
                    flux=diagnostics.get('maximum_total_flux_residual'),
                    energy=diagnostics.get('maximum_relative_cell_energy_balance_residual'),
                    correction=diagnostics.get('maximum_log_temperature_correction'),
                    seconds=time.monotonic()-started)
                report['latest_iteration'] = row
                checkpoint(output/'diagnostic-latest.npz', atmosphere)
                if material.saved_structure_grid is not None:
                    np.savez_compressed(output/'diagnostic-structure-grid.npz',
                                        wavelength=material.saved_structure_grid)
                (output/'progress.json').write_text(json.dumps(row, indent=2)+'\n')
                print(json.dumps(row), flush=True)
                record()
                material.check_budget()

            def solve_segment(*args, **kwargs):
                try:
                    atmosphere = material.solve(*args, **kwargs)
                finally:
                    release_proposals()
                    # Solver closures can form cycles around full radiation
                    # arrays. NumPy allocations do not reliably trigger cyclic
                    # collection before the next domain allocates its arrays.
                    gc.collect()
                report.setdefault('completed_domain_diagnostics', []).append(dict(
                    depths=atmosphere.n_depth,
                    certificate=atmosphere.metadata.get('equilibrium_certificate', {}),
                    lower_boundary_response=atmosphere.metadata.get('lower_boundary_absorption_escape_bound')))
                checkpoint(output/f'diagnostic-depths-{atmosphere.n_depth}.npz', atmosphere)
                record()
                return atmosphere

            atmosphere = solve_with_screened_boundary(solve_segment,
                config.effective_temperature, config.logg, n_depth=40, n_continuum=300,
                max_iterations=150, n_angle=3, iteration_callback=progress,
                maximum_domain_expansions=8,
                lower_domain_extension=partial(append_lower_domain, maximum_new_cells=1))
            report['certificate'] = atmosphere.metadata.get('equilibrium_certificate', {})
            if not report['certificate'].get('verified'):
                raise RuntimeError('DQ atmosphere failed its physical certificate')
            material.check_budget()
            np.savez_compressed(output/'structure-grid.npz', wavelength=material.saved_structure_grid)
            structural = material.spectrum(atmosphere, material.saved_structure_grid, 3)
            report['structure_spectrum_flux_ratio'] = float(structural.bolometric_flux/
                (STEFAN_BOLTZMANN*config.effective_temperature**4))
            report['status'] = 'structure_converged'
            save_model_result(ModelResult('DQ', atmosphere, structural, config,
                              {'spectral_qualification': False}), output/'structure')
            record()
            fine = material.spectrum(atmosphere, independent_grid(), 3)
            path = output/'independent-spectrum.npz'
            np.savez_compressed(path, wavelength=fine.wavelength_angstrom,
                                flux=fine.surface_flux_lambda)
            report['independent_spectrum_sha256'] = digest(path)
            qualification = qualify_spectrum(atmosphere, fine, config)
            report.update(qualification)
            after = source_hashes()
            report['source_consistency_verified'] = after == frozen
            if after != frozen:
                raise RuntimeError('DQ sources changed during calculation; qualification refused')
            if validate_data() != report['constitutive_data']:
                raise RuntimeError('DQ constitutive data changed during calculation')
            # A requested output grid never determines the atmosphere or
            # bypasses the independent, absolute bolometric flux gate.
            spectrum = fine if wavelength is None else material.spectrum(atmosphere, wavelength, 3)
            result = ModelResult('DQ', atmosphere, spectrum, config, dict(
                atmosphere_convergence_status='converged', spectral_qualification=True,
                atmosphere_initialization='self-contained hydrostatic gray cold start',
                dq_protocol=report['protocol'], independent_spectrum_flux_ratio=
                    report['independent_spectrum_flux_ratio'],
                cold_start=True, full_physics_validation=False,
                independent_depth_grid_validation=False))
            save_model_result(result, output/'model')
            report['status'] = 'completed'
            record()
            return result
    except BaseException as error:
        report.update(status=('budget_exhausted' if isinstance(error, TimeoutError) else
                             'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'),
                      error=str(error))
        record()
        raise
