#!/usr/bin/env python3
"""Bounded transport-seed and explicit mesh-refinement experiments.

The approximate seed is NOT a solution or a fallback. All reported final flux
residuals come from the non-gray physical solver. The default constructs a
fresh seed; --refine-atmosphere explicitly refines a named previously solved
model and is never reported as a cold start.
"""
import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import logging
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from wd_spectra._compat import trapezoid
from wd_spectra.atmosphere import (Atmosphere, helium_continuum_atmosphere,
    radiative_equilibrium_helium_atmosphere, radiative_equilibrium_hydrogen_helium_atmosphere)
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.convection import ml2_temperature_gradient_for_total_flux_from_thermodynamics
from wd_spectra.eos import (hummer_mihalas_helium_lte, hummer_mihalas_helium_thermodynamics,
    hummer_mihalas_helium_lte_with_reos3, hummer_mihalas_hydrogen_helium_lte,
    hummer_mihalas_hydrogen_helium_thermodynamics)
from wd_spectra.helium import helium_continuum_mass_absorption_coefficient, rosseland_mean_helium_continuum_opacity
from wd_spectra.models import (DBConfig, DABConfig, ModelData, compute_db, compute_dab,
    save_model_result, load_atmosphere_checkpoint)
from wd_spectra.models.common import _jsonable
from wd_spectra.models.stellar import _helium_tables, _da_self_broadening_prescription
from wd_spectra.mixture import (hydrogen_helium_continuum_mass_absorption_coefficient,
                              rosseland_mean_hydrogen_helium_continuum_opacity)
from wd_spectra.opacity import optical_depth_from_mass_opacity
from wd_spectra.spectrum import planck_lambda_angstrom


def atmosphere_at(teff, pressure, temperature, tau, table=None, log_h_he=None):
    if log_h_he is not None:
        if table is not None:
            raise ValueError("Mixed diagnostic does not support changing the bulk EOS")
        state = hummer_mihalas_hydrogen_helium_lte(temperature, pressure, log_h_he,
                                                  correlated_microfields=True)
        hydrogen = state.hydrogen_lte_state
        return Atmosphere(teff, 8., tau, pressure/1e8, temperature, pressure, state.mass_density,
            hydrogen.neutral_h_density, hydrogen.proton_density, state.electron_density, {},
            hydrogen_lte_state=hydrogen, helium_lte_state=state.helium_lte_state)
    if table is None:
        state = hummer_mihalas_helium_lte(temperature, pressure, correlated_microfields=True)
    else:
        if not np.all(table.evaluate(pressure, temperature)[2]):
            raise ValueError("Experiment left REOS3 domain; no ideal-EOS substitution allowed")
        state = hummer_mihalas_helium_lte_with_reos3(temperature, pressure, table, correlated_microfields=True)
    return Atmosphere(teff, 8., tau, pressure / 1e8, temperature, pressure,
                      state.mass_density, np.zeros_like(temperature), np.zeros_like(temperature),
                      state.electron_density, {}, helium_lte_state=state)


def transport_seed(teff, n_depth, bottom_tau, table=None, mesh="pressure", *, log_h_he=None):
    print(f"seed: constructing gray surface boundary for {teff:g} K, {n_depth} layers", flush=True)
    gray = helium_continuum_atmosphere(teff, 8., n_depth=n_depth, helium_reos3_table=table,
        **({} if log_h_he is None else {'log_hydrogen_abundance':log_h_he}))
    print(f"seed: gray boundary ready; integrating hydrostatic ML2 transport from "
          f"P={gray.gas_pressure[0]:.6g}, T={gray.temperature[0]:.6g}", flush=True)
    target = STEFAN_BOLTZMANN * teff**4
    evaluations = 0

    def rhs(logp, y):
        nonlocal evaluations
        temp = np.array([np.exp(y[0])])
        pressure = np.array([np.exp(logp)])
        point = atmosphere_at(teff, pressure, temp, np.ones(1), table, log_h_he=log_h_he)
        if log_h_he is None:
            opacity = rosseland_mean_helium_continuum_opacity(point, n_frequency=160)
            thermo = hummer_mihalas_helium_thermodynamics(temp, pressure, correlated_microfields=True,
                                                       helium_reos3_table=table)
        else:
            opacity = rosseland_mean_hydrogen_helium_continuum_opacity(point, n_frequency=160)
            thermo = hummer_mihalas_hydrogen_helium_thermodynamics(temp, pressure, log_h_he,
                                                                correlated_microfields=True)
        gradient = ml2_temperature_gradient_for_total_flux_from_thermodynamics(
            point, opacity, target, thermo.specific_heat_constant_pressure,
            thermo.density_temperature_derivative, thermo.adiabatic_temperature_gradient)
        evaluations += 1
        if evaluations % 100 == 0:
            print(f"seed: evaluations={evaluations} P={pressure[0]:.4g} T={temp[0]:.4g} tau={y[1]:.4g}", flush=True)
        return [gradient[0], opacity[0] * pressure[0] / point.gravity]

    def bottom(logp, y):
        return y[1] - bottom_tau
    bottom.terminal = True
    bottom.direction = 1
    initial_logp = np.log(gray.gas_pressure[0])
    integrated = solve_ivp(rhs, (initial_logp, initial_logp + 40),
                           [np.log(gray.temperature[0]), gray.rosseland_optical_depth[0]],
                           rtol=1e-5, atol=[1e-7, 1e-9], max_step=.25, first_step=.01,
                           events=bottom, dense_output=True)
    if not integrated.success or not integrated.t_events[0].size:
        raise RuntimeError(f"Transport seed did not reach the lower optical depth: {integrated.message}")
    logp = np.linspace(initial_logp, integrated.t[-1], n_depth)
    if mesh == "optical":
        requested = np.geomspace(gray.rosseland_optical_depth[0], bottom_tau, n_depth)
        for j in range(1, n_depth-1):
            logp[j] = brentq(lambda x: integrated.sol(x)[1]-requested[j],
                             initial_logp, integrated.t[-1])
    logt, tau = integrated.sol(logp)
    return atmosphere_at(teff, np.exp(logp), np.exp(logt), tau, table, log_h_he=log_h_he)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("temperature", type=int)
    parser.add_argument("--log-h-he", type=float,
                        help="Explicit atomic DAB continuation/refinement instead of pure He")
    parser.add_argument("--n-depth", type=int, default=80)
    parser.add_argument("--structure-wavelength-count",type=int,default=600,
                        help="Continuum transfer quadrature count; independent spectrum uses 4000 points")
    parser.add_argument("--bottom-tau", type=float, default=100.)
    parser.add_argument("--mesh", choices=("pressure", "optical"), default="pressure")
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument("--physical-only", action="store_true",
                        help="Test full physical flux equations from the fresh transport seed")
    parser.add_argument("--convective-trial-correction", action="store_true")
    parser.add_argument("--finite-material-repair", action="store_true",
                        help="Research TLUSTY-inspired EOS-consistent convection proposal repair")
    parser.add_argument("--inverse-ml2-step", action="store_true",
                        help="Research signed convective-velocity proposal coordinates")
    parser.add_argument("--stable-helium-thermodynamics", action="store_true")
    parser.add_argument("--nonlinear-materials", action="store_true")
    parser.add_argument("--smooth-h2", action="store_true")
    parser.add_argument("--state-sum-h2", action="store_true")
    parser.add_argument("--stable-transfer", action="store_true",
                        help="Research cancellation-resistant field AND direct tangent")
    parser.add_argument("--cell-conservation", action="store_true",
                        help="Experimental invertible cell-energy rows in the physical phase")
    parser.add_argument("--transport-conditioned", action="store_true",
                        help="Equivalent cell/flux rows based on fixed local optical thickness")
    parser.add_argument("--auxiliary-ml2", action="store_true",
                        help="Experimental exact auxiliary ML2 coordinate plus physical flux gate")
    parser.add_argument("--local-energy-scaling", action="store_true",
                        help="Combine auxiliary ML2 with fixed local thermal/convective energy scales")
    parser.add_argument("--step-method", choices=("legacy", "box", "box-row", "temperature-penalty", "nonlinear-convection", "nonlinear-convection-regularized", "nonlinear-convection-local-energy", "nonlinear-convection-direct-energy", "nonlinear-convection-current-energy", "nonlinear-convection-cell-energy"),
                        default="legacy", help="Research proposal strategy in the full physical phase")
    parser.add_argument("--no-continuations", action="store_true",
                        help="Enforce the stated per-phase budget without extra formal segments")
    parser.add_argument("--inner-scaling", choices=("unit", "jac", "svd", "bvls"), default="unit",
                        help="Research inner-coordinate scaling; leaves physical bounds and residuals unchanged")
    parser.add_argument("--inner-row-scaling", action="store_true",
                        help="Fixed Jacobian row equilibration of the proposal only")
    parser.add_argument("--inner-max-evaluations", type=int, default=100,
                        help="Bound the inexpensive inner proposal solve; not atmosphere iterations")
    parser.add_argument("--reuse-fresh-seed", type=Path,
                        help="Explicit controlled experiment on an already generated analytic seed, never a relaxed model")
    parser.add_argument("--reos3", action="store_true", help="Explicit bulk-EOS experiment; HM chemistry is still approximate")
    parser.add_argument("--helium-dimer", action="store_true",
                        help="Isolated ideal He2+ charge/enthalpy/opacity experiment, not production physics")
    parser.add_argument("--refine-atmosphere", type=Path,
                        help="Explicit mesh-convergence test of a newly solved atmosphere; NOT a cold-start claim")
    parser.add_argument("--resume-atmosphere", type=Path,
                        help="Explicit continuation on the identical saved grid; NOT a new cold start")
    parser.add_argument("--extend-atmosphere", type=Path,
                        help="Append a transport seed below an explicitly named model; NOT a cold start")
    parser.add_argument("--extension-escape-tolerance", type=float, default=3e-4,
                        help="Absorption-only lower-boundary screen for the appended seed")
    parser.add_argument("--output-root", type=Path, default=Path("results/cool-db-transport-seed-20260904"))
    args = parser.parse_args()
    if args.state_sum_h2:
        if args.smooth_h2 or args.log_h_he is None or args.stable_helium_thermodynamics:
            raise ValueError('State-sum H2 requires an independent molecular H/He experiment')
        from .state_sum_h2_experiment import state_sum_h2_experiment
        with state_sum_h2_experiment():
            return run(args)
    if args.smooth_h2:
        if args.log_h_he is None or args.stable_helium_thermodynamics:
            raise ValueError('Smooth H2 experiment requires molecular H/He; do not combine thermal experiments')
        from .smooth_h2_thermodynamics_experiment import smooth_h2_experiment
        with smooth_h2_experiment():
            return run(args)
    if args.stable_helium_thermodynamics:
        if args.helium_dimer or args.reos3 or args.log_h_he is not None:
            raise ValueError('Stable He thermal experiment requires unmodified pure-He chemistry')
        from .stable_helium_thermodynamics_experiment import stable_helium_thermal_experiment
        with stable_helium_thermal_experiment():
            return run(args)
    if args.helium_dimer:
        if args.reos3 or args.log_h_he is not None or any(x is not None for x in (
                args.reuse_fresh_seed, args.refine_atmosphere, args.resume_atmosphere, args.extend_atmosphere)):
            raise ValueError("He2+ experiment requires a fresh pure-He calculation without other EOS substitutions")
        import sys
        from .helium_dimer_eos_experiment import helium_dimer_atmosphere_experiment
        with helium_dimer_atmosphere_experiment(sys.modules[__name__]):
            return run(args)
    return run(args)


def run(args):
    from .solver_step_experiments import DIRECT_ENERGY_METHODS, LOCAL_ENERGY_METHODS
    if args.structure_wavelength_count < 2:
        raise ValueError('structure wavelength count must be at least two')
    if (args.stable_transfer and args.step_method not in DIRECT_ENERGY_METHODS
            and not getattr(args,'experimental_full_coupled_newton',False)):
        raise ValueError("Stable transfer experiment requires the direct-energy tangent")
    if args.finite_material_repair and (args.step_method not in DIRECT_ENERGY_METHODS
                                      or args.convective_trial_correction):
        raise ValueError("Finite material repair requires a direct-energy proposal and no old cap")
    if args.inverse_ml2_step and args.step_method not in ('nonlinear-convection-current-energy','nonlinear-convection-cell-energy'):
        raise ValueError("Inverse ML2 proposals require current local-energy rows")
    if args.nonlinear_materials and (args.step_method not in DIRECT_ENERGY_METHODS
            or (args.inverse_ml2_step and not getattr(args,'experimental_exact_dense_materials',False))
            or args.finite_material_repair):
        raise ValueError('Local material table requires a nodal direct-energy proposal, tested independently')
    if sum(x is not None for x in (args.reuse_fresh_seed, args.refine_atmosphere,
                                 args.resume_atmosphere, args.extend_atmosphere)) > 1:
        raise ValueError("Choose exactly one supplied seed, refinement or continuation source")
    if args.log_h_he is not None and (args.reos3 or args.reuse_fresh_seed is not None):
        raise ValueError("Mixed diagnostics forbid pure-He bulk-EOS substitution or an untyped seed archive")
    checkpoint_options = ({} if args.log_h_he is None else {"log_hydrogen_to_helium": args.log_h_he})
    composition = "helium" if args.log_h_he is None else "mixed"
    experimental_physics = getattr(args, 'experimental_physics', None)
    if (args.resume_atmosphere is not None or args.extend_atmosphere is not None) and (args.reos3 or not args.physical_only):
        raise ValueError("A continuation requires --physical-only and the unchanged EOS")
    if args.inner_max_evaluations < 1:
        raise ValueError("Inner evaluation budget must be positive")
    if (args.inner_scaling != "unit" or args.inner_max_evaluations != 100) and not args.step_method.startswith("nonlinear-convection"):
        raise ValueError("Inner optimizer options require a nonlinear convection proposal")
    if sum((args.cell_conservation, args.transport_conditioned, args.auxiliary_ml2)) > 1:
        raise ValueError("Choose one energy-balance representation")
    if args.auxiliary_ml2 and args.convective_trial_correction:
        raise ValueError("Test auxiliary convection independently of trial correction")
    if args.local_energy_scaling and not args.auxiliary_ml2:
        raise ValueError("Local energy experiment requires --auxiliary-ml2")
    if args.step_method != "legacy" and (args.auxiliary_ml2 or args.cell_conservation
                                         or args.transport_conditioned or args.convective_trial_correction):
        raise ValueError("Compare proposal strategies independently of other experiments")
    auxiliary_metadata = []
    out = args.output_root / str(args.temperature)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "solver.log").exists():
        raise RuntimeError("Refusing to overwrite an experiment")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "solver.log")])
    (out/'experiment-options.json').write_text(json.dumps(
        {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2)+'\n')
    started = time.monotonic()
    extension_metadata = None
    table = None
    if args.reos3:
        from wd_spectra.dense_eos import read_helium_reos3_table
        table = read_helium_reos3_table(ModelData.default().helium_reos3)
        if args.reuse_fresh_seed is not None:
            raise ValueError("Regenerate the seed when changing the EOS")
    if args.extend_atmosphere is not None:
        from .domain_extension_experiment import extend_domain
        previous = load_atmosphere_checkpoint(args.extend_atmosphere, args.temperature, 8., composition, **checkpoint_options)
        boundary_wave = np.geomspace(100, 1e7, 600)
        opacity_function = (helium_continuum_mass_absorption_coefficient if args.log_h_he is None
                            else hydrogen_helium_continuum_mass_absorption_coefficient)
        absorption = opacity_function(previous, boundary_wave, include_electron_scattering=False,
                                      include_rayleigh_scattering=False)
        boundary_tau = optical_depth_from_mass_opacity(previous.column_mass, absorption)[:, -1]
        target = STEFAN_BOLTZMANN*args.temperature**4
        def local_transport(pressure, temperature):
            p, t = np.array([pressure]), np.array([temperature])
            point = atmosphere_at(args.temperature, p, t, np.ones(1), log_h_he=args.log_h_he)
            if args.log_h_he is None:
                rosseland = rosseland_mean_helium_continuum_opacity(point, n_frequency=160)
                thermo = hummer_mihalas_helium_thermodynamics(t, p, correlated_microfields=True)
            else:
                rosseland = rosseland_mean_hydrogen_helium_continuum_opacity(point, n_frequency=160)
                thermo = hummer_mihalas_hydrogen_helium_thermodynamics(t, p, args.log_h_he,
                                                                    correlated_microfields=True)
            gradient = ml2_temperature_gradient_for_total_flux_from_thermodynamics(
                point, rosseland, target, thermo.specific_heat_constant_pressure,
                thermo.density_temperature_derivative, thermo.adiabatic_temperature_gradient,
                mixing_length_alpha=1.25)
            absorption = opacity_function(point, boundary_wave, include_electron_scattering=False,
                                          include_rayleigh_scattering=False)[:, 0]
            return gradient[0], rosseland[0], absorption
        p, t, tau, extension_metadata = extend_domain(previous.gas_pressure, previous.temperature,
            previous.rosseland_optical_depth, boundary_tau, boundary_wave, previous.gravity,
            target, local_transport, escape_tolerance=args.extension_escape_tolerance)
        seed = atmosphere_at(args.temperature, p, t, tau, log_h_he=args.log_h_he)
        args.n_depth = seed.n_depth
        provenance = f"explicit domain extension of {args.extend_atmosphere}; not a cold start"
        print(f"Explicit domain extension: {extension_metadata}", flush=True)
    elif args.resume_atmosphere is not None:
        seed = load_atmosphere_checkpoint(args.resume_atmosphere, args.temperature, 8., composition, **checkpoint_options)
        args.n_depth = seed.n_depth
        provenance = f"explicit same-grid continuation of {args.resume_atmosphere}; not a cold start"
        print(f"Explicit continuation: {seed.n_depth} layers; {args.resume_atmosphere}", flush=True)
    elif args.refine_atmosphere is not None:
        if args.reuse_fresh_seed is not None or args.reos3:
            raise ValueError("Mesh refinement must not also change the seed source or EOS")
        previous = load_atmosphere_checkpoint(args.refine_atmosphere, args.temperature, 8., composition, **checkpoint_options)
        old_logp = np.log(previous.gas_pressure)
        logp = np.sort(np.concatenate((old_logp, .5*(old_logp[:-1]+old_logp[1:]))))
        seed = atmosphere_at(args.temperature, np.exp(logp),
                            np.exp(np.interp(logp, old_logp, np.log(previous.temperature))),
                            np.exp(np.interp(logp, old_logp, np.log(previous.rosseland_optical_depth))),
                            log_h_he=args.log_h_he)
        args.n_depth = seed.n_depth
        provenance = f"explicit mesh refinement of {args.refine_atmosphere}; not a cold start"
        print(f"Explicit refinement: {previous.n_depth} -> {seed.n_depth} layers; {args.refine_atmosphere}", flush=True)
    elif args.reuse_fresh_seed is None:
        seed = transport_seed(args.temperature, args.n_depth, args.bottom_tau, table, args.mesh,
                              **({} if args.log_h_he is None else {'log_h_he':args.log_h_he}))
        provenance = "fresh hydrostatic ML2/diffusion ODE; not a saved atmosphere"
    else:
        if args.reuse_fresh_seed.name != "seed.npz" or args.reuse_fresh_seed.parent.name != str(args.temperature):
            raise ValueError("Only this experiment's temperature-matched fresh seed.npz may be reused")
        with np.load(args.reuse_fresh_seed) as saved:
            if "diagnostic_seed" not in saved or not str(saved["diagnostic_seed"]).startswith("fresh hydrostatic"):
                raise ValueError("Seed provenance does not certify a fresh analytic seed")
            seed = atmosphere_at(args.temperature, saved["gas_pressure"], saved["temperature"],
                                 saved["rosseland_optical_depth"])
        if seed.n_depth != args.n_depth:
            raise ValueError("Reused seed layer count must match --n-depth")
        provenance = f"fresh hydrostatic ML2/diffusion ODE reused from {args.reuse_fresh_seed}"
        print(f"Explicit analytic seed comparison: {args.reuse_fresh_seed}; not a converged checkpoint", flush=True)
    if (args.physical_only or args.convective_trial_correction or args.no_continuations
            or args.step_method in DIRECT_ENERGY_METHODS):
        import wd_spectra.adaptive_structure as adaptive
        original_solve = adaptive.solve_adaptive_lte_structure
        def full_flux(*positional, **kwargs):
            # This qualified worker already owns direct local-energy rows,
            # thermal initialization and measured nonlinear-ML2 proposals.
            # Do not apply the public atomic solver's completion a second time.
            kwargs['enforce_local_energy_balance'] = False
            if args.finite_material_repair or args.nonlinear_materials:
                from .convective_consistency_experiment import MaterialCoefficients
                material_context['materials'] = MaterialCoefficients(
                    kwargs['with_temperature'], kwargs['thermodynamics'],
                    kwargs['rosseland_opacity'], kwargs['mixing_length_alpha'])
            if args.physical_only:
                kwargs.update(use_convective_gradient_preconditioner=False,
                              project_initial_convective_gradient=False,
                              use_initial_bolometric_rescaling=False)
            if args.convective_trial_correction:
                kwargs['use_convective_trial_correction'] = True
            if args.no_continuations:
                kwargs['maximum_formal_flux_continuations'] = 0
            if args.step_method in DIRECT_ENERGY_METHODS:
                kwargs['compute_local_energy_response'] = True
            return original_solve(*positional, **kwargs)
        adaptive.solve_adaptive_lte_structure = full_flux
    if args.cell_conservation or args.transport_conditioned:
        import wd_spectra.adaptive_structure as adaptive
        from wd_spectra._energy_balance import cell_conservation_rows, transport_conditioned_rows
        from wd_spectra.nonlinear import NonlinearEvaluation
        original_newton = adaptive.solve_trust_region_newton
        def cell_newton(initial, evaluate, **options):
            initial_evaluation = evaluate(initial, False)
            if not initial_evaluation.payload["energy_balance_is_physical_flux"]:
                return original_newton(initial, evaluate, **options)
            transform = cell_conservation_rows
            if args.transport_conditioned:
                payload = initial_evaluation.payload
                rosseland = payload["convection_transport"]["rosseland"]
                tau_r = optical_depth_from_mass_opacity(
                    payload["atmosphere"].column_mass, rosseland[None, :])[0]
                dtau = np.diff(tau_r, prepend=0.)
                cell_width = .5*(dtau[:-1] + dtau[1:])
                transform = lambda values: transport_conditioned_rows(values, cell_width)
            print("Physical phase: invertible energy-balance rows; all-depth flux gate retained", flush=True)
            def cell_evaluate(state, need_jacobian):
                ev = evaluate(state, need_jacobian)
                return NonlinearEvaluation(
                    transform(ev.residual),
                    None if ev.jacobian is None else transform(ev.jacobian),
                    ev.payload)
            return original_newton(initial, cell_evaluate, **options)
        adaptive.solve_trust_region_newton = cell_newton
    if args.auxiliary_ml2:
        import wd_spectra.adaptive_structure as adaptive
        from wd_spectra._ml2_auxiliary import solve_auxiliary_ml2_experiment
        original_newton = adaptive.solve_trust_region_newton
        def auxiliary_newton(initial, evaluate, **options):
            if not evaluate(initial, False).payload["energy_balance_is_physical_flux"]:
                return original_newton(initial, evaluate, **options)
            print("Physical phase: exact auxiliary ML2 coordinate; actual-gradient flux gate retained", flush=True)
            result, metadata = solve_auxiliary_ml2_experiment(
                initial, evaluate, original_newton,
                STEFAN_BOLTZMANN * args.temperature**4,
                local_energy_scaling=args.local_energy_scaling, **options)
            auxiliary_metadata.append(metadata)
            return result
        adaptive.solve_trust_region_newton = auxiliary_newton
    if args.step_method != "legacy":
        import wd_spectra.adaptive_structure as adaptive
        from .solver_step_experiments import make_step, CURRENT_ENERGY_METHODS
        original_newton = adaptive.solve_trust_region_newton
        material_context = {}
        def proposal_newton(initial, evaluate, **options):
            # Neither an analytic seed nor an interpolated/refined structure
            # supplies a measured small correction on the equations just built.
            options["allow_initial_convergence"] = False
            first = evaluate(initial, False)
            if first.payload["energy_balance_is_physical_flux"]:
                print(f"Physical phase: {args.step_method} proposals; all physical gates retained", flush=True)
                options["step_builder"] = lambda state, ev, jac, radius: make_step(
                    state, ev, jac, radius, args.step_method, inner_scaling=args.inner_scaling,
                    inner_max_evaluations=args.inner_max_evaluations, inner_row_scaling=args.inner_row_scaling)
                if args.inverse_ml2_step:
                    from .inverse_ml2_proposal import inverse_step
                    options['step_builder'] = lambda state, ev, jac, radius: inverse_step(
                        state, ev, radius, args.inner_max_evaluations,
                        cell_only=args.step_method == 'nonlinear-convection-cell-energy')
                if args.step_method.startswith("nonlinear-convection"):
                    # The cheap inner model must use current radiation and
                    # material responses, not an earlier accepted atmosphere.
                    options["jacobian_refresh_interval"] = 1
                if args.step_method in LOCAL_ENERGY_METHODS:
                    from wd_spectra._energy_balance import locally_scaled_energy_rows
                    from wd_spectra.nonlinear import NonlinearEvaluation
                    target = STEFAN_BOLTZMANN * args.temperature**4
                    scale = first.payload["cell_energy_scale"] / target
                    physical_evaluate = evaluate
                    def evaluate(state, need_jacobian):
                        ev = physical_evaluate(state, need_jacobian)
                        payload = {**ev.payload, "diagnostic_fixed_cell_scale": scale}
                        normalization = (payload["cell_energy_scale"]/target
                                         if args.step_method in CURRENT_ENERGY_METHODS else scale)
                        values = ev.residual.copy()
                        cell_defect = (payload["radiative_cell_energy_defect"]
                                       + np.diff(payload["convective_flux_interface"])) / target
                        values[:-1] -= cell_defect / normalization
                        tangent = None
                        if ev.jacobian is not None:
                            if args.step_method in DIRECT_ENERGY_METHODS:
                                tangent = ev.jacobian.copy()
                                tangent[:-1] -= (payload["cell_energy_log_temperature_jacobian"]
                                               @ payload["log_temperature_from_state"]
                                               / target / normalization[:, None])
                                if args.step_method in CURRENT_ENERGY_METHODS:
                                    tangent[:-1] += ((cell_defect/normalization)[:, None]
                                        * (payload["cell_energy_scale_log_temperature_jacobian"]
                                           @ payload["log_temperature_from_state"])
                                        / target / normalization[:, None])
                            else:
                                tangent = locally_scaled_energy_rows(ev.jacobian, scale)
                        if args.step_method == "nonlinear-convection-cell-energy":
                            values=np.concatenate(([ev.residual[0]],cell_defect/normalization))
                            if tangent is not None:
                                tangent=np.vstack((ev.jacobian[0],ev.jacobian[:-1]-tangent[:-1]))
                        return NonlinearEvaluation(values, tangent, payload)
                    physical_convergence = options.get("convergence_test")
                    if (physical_convergence is not None
                            or args.step_method in DIRECT_ENERGY_METHODS):
                        local_tolerance = options.get("residual_tolerance", 2e-3)
                        def convergence_test(state, ev, step):
                            if args.step_method in DIRECT_ENERGY_METHODS:
                                # Fixed inner scales must not hide a growing
                                # relative defect as thermal emission changes.
                                local = ev.payload["cell_energy_balance_relative_residual"]
                                if np.any(~np.isfinite(local)) or np.max(abs(local)) >= local_tolerance:
                                    return False
                            physical = NonlinearEvaluation(
                                ev.payload["total_flux_interface"]/target-1, None, ev.payload)
                            return (physical_convergence is None
                                    or physical_convergence(state, physical, step))
                        options["convergence_test"] = convergence_test
                if args.finite_material_repair:
                    from .convective_consistency_experiment import attach_repair
                    attach_repair(options, material_context['materials'])
                if args.nonlinear_materials:
                    from .convective_consistency_experiment import TabulatedMaterials
                    original_builder = options['step_builder']
                    def material_builder(state, ev, jac, radius):
                        table = TabulatedMaterials(material_context['materials'],
                            np.log(ev.payload['atmosphere'].temperature), radius)
                        actual = NonlinearEvaluation(ev.residual, ev.jacobian,
                            {**ev.payload, 'diagnostic_material_model': table,
                             'diagnostic_positive_radiative_rates':(
                                 ev.payload.get('diagnostic_positive_radiative_rates',False)
                                 or getattr(args,'experimental_positive_radiative_rates',False))})
                        direction = original_builder(state, actual, jac, radius)
                        # An actual-material trial projector needs the inner
                        # auxiliary proposal, not just its temperature part.
                        if 'diagnostic_augmented_proposed_velocity' in actual.payload:
                            ev.payload['diagnostic_augmented_proposed_velocity'] = (
                                actual.payload['diagnostic_augmented_proposed_velocity'].copy())
                        return direction
                    options['step_builder'] = material_builder
                if args.step_method in DIRECT_ENERGY_METHODS:
                    from .measured_proposal_guard import require_measured_proposal
                    options=require_measured_proposal(options)
            return original_newton(initial, evaluate, **options)
        adaptive.solve_trust_region_newton = proposal_newton
    if getattr(args,'experimental_initializer_description',None):
        provenance=args.experimental_initializer_description
    if args.helium_dimer:
        provenance = "EXPERIMENTAL HM atoms + He2+ charge/thermal/opacity closure: " + provenance
    if experimental_physics is not None:
        provenance = experimental_physics + ': ' + provenance
    print(f"{provenance}: Pbottom={seed.gas_pressure[-1]:.6g} Tbottom={seed.temperature[-1]:.6g} "
          f"rhobottom={seed.mass_density[-1]:.6g}", flush=True)
    def save_diagnostic_structure(name, atmosphere, **metadata):
        fields = dict(temperature=atmosphere.temperature, gas_pressure=atmosphere.gas_pressure,
                      column_mass=atmosphere.column_mass,
                      rosseland_optical_depth=atmosphere.rosseland_optical_depth)
        if experimental_physics is not None:
            name = 'experimental-'+name
            fields = {'experimental_'+key: value for key, value in fields.items()}
            metadata['experimental_physics'] = experimental_physics
        np.savez_compressed(out / name, **fields, **metadata)
    save_diagnostic_structure('seed.npz', seed, diagnostic_seed=provenance)
    he_i, he_ii = _helium_tables(ModelData.default())
    from .stable_feautrier_experiment import stable_transfer_experiment
    with (out / "iterations.jsonl").open("w") as telemetry, \
            (stable_transfer_experiment() if args.stable_transfer else nullcontext()):
        def report(iteration, atmosphere, diagnostics):
            row = dict(elapsed_seconds=time.monotonic() - started, iteration=iteration,
                       diagnostics=_jsonable(diagnostics))
            telemetry.write(json.dumps(row) + "\n")
            telemetry.flush()
            save_diagnostic_structure('latest-iteration.npz', atmosphere)
            print(json.dumps(row), flush=True)
        solver = (radiative_equilibrium_helium_atmosphere if args.log_h_he is None
                  else radiative_equilibrium_hydrogen_helium_atmosphere)
        solver_arguments = ((args.temperature, 8.) if args.log_h_he is None
                            else (args.temperature, 8., args.log_h_he))
        mixture_options = ({} if args.log_h_he is None else dict(
            hydrogen_self_broadening_prescription=_da_self_broadening_prescription(args.temperature, None),
            hydrogen_self_broadening_truncation_closure="stark-core"))
        relaxed = solver(
            *solver_arguments, **mixture_options, stark_table=he_i, helium_ii_stark_table=he_ii,
            structure_solver="adaptive-newton", n_depth=args.n_depth, n_continuum_wavelength=args.structure_wavelength_count,
            max_iterations=args.max_iterations, initial_temperature=seed.temperature,
            initial_column_mass=seed.column_mass, initial_gas_pressure=seed.gas_pressure,
            initial_rosseland_optical_depth=seed.rosseland_optical_depth,
            resume_supplied_structure_in_formal_flux_phase=(args.refine_atmosphere is not None
                                                           or args.resume_atmosphere is not None
                                                           or args.extend_atmosphere is not None),
            iteration_callback=report,
            helium_reos3_table=table)
    relaxed = replace(relaxed, metadata={**relaxed.metadata,
                      "diagnostic_seed": provenance,
                      "diagnostic_log_hydrogen_to_helium": args.log_h_he,
                      "diagnostic_domain_extension": extension_metadata,
                      "diagnostic_stable_transfer": args.stable_transfer,
                      "diagnostic_helium_dimer_equilibrium": args.helium_dimer,
                      "diagnostic_step_method": args.step_method,
                      "diagnostic_inner_scaling": args.inner_scaling,
                      "diagnostic_inner_row_scaling": args.inner_row_scaling,
                      "diagnostic_inner_max_evaluations": args.inner_max_evaluations,
                      "diagnostic_continuation_requires_evaluated_correction": args.step_method != "legacy",
                      "diagnostic_requires_non_trust_limited_correction": (
                          args.step_method in DIRECT_ENERGY_METHODS
                          or getattr(args,'experimental_full_coupled_newton',False)),
                      "diagnostic_auxiliary_ml2_segments": auxiliary_metadata,
                      "diagnostic_energy_balance_form": (
                          "flux-minus-current-local-energy" if args.step_method == "nonlinear-convection-current-energy" else
                          "flux-minus-direct-local-energy" if args.step_method == "nonlinear-convection-direct-energy" else
                          "flux-minus-local-energy" if args.step_method == "nonlinear-convection-local-energy" else
                          "auxiliary-ml2" if args.auxiliary_ml2 else
                          "transport-conditioned" if args.transport_conditioned else
                          "cell-conservation" if args.cell_conservation else "flux")})
    wave = np.geomspace(100, 1e7, 4000)
    compute = compute_db if args.log_h_he is None else compute_dab
    config = (DBConfig(effective_temperature=args.temperature, quality="production") if args.log_h_he is None
              else DABConfig(effective_temperature=args.temperature, quality="production", log_hydrogen_to_helium=args.log_h_he))
    if args.smooth_h2:
        relaxed=replace(relaxed,metadata={**relaxed.metadata,
            'experimental_h2_partition':'C4 quintic logQ; analytic energy derivative; not production default'})
    if args.state_sum_h2:
        relaxed=replace(relaxed,metadata={**relaxed.metadata,
            'experimental_h2_partition':'Roueff 2019 RACPPK 302 states; Q and energy from identical sum; not production default'})
    result = compute(config,
                        wave, initial_atmosphere=relaxed, relax_atmosphere=False)
    save_model_result(result, out)
    absorption_function = (helium_continuum_mass_absorption_coefficient if args.log_h_he is None
                           else hydrogen_helium_continuum_mass_absorption_coefficient)
    absorption = absorption_function(
        relaxed, wave, include_electron_scattering=False, include_rayleigh_scattering=False)
    tau = optical_depth_from_mass_opacity(relaxed.column_mass, absorption)
    # Conservative vertical escape estimate: oblique paths are longer.
    leakage = trapezoid(np.pi * planck_lambda_angstrom(wave, relaxed.temperature[-1])
                        * np.exp(-tau[:, -1]), wave) / (STEFAN_BOLTZMANN * args.temperature**4)
    spectrum_ratio = float(trapezoid(result.spectrum.surface_flux_lambda, wave)
                           / (STEFAN_BOLTZMANN * args.temperature**4))
    summary = dict(elapsed_seconds=time.monotonic() - started,
                   refined_from=None if args.refine_atmosphere is None else str(args.refine_atmosphere),
                   resumed_from=None if args.resume_atmosphere is None else str(args.resume_atmosphere),
                   extended_from=None if args.extend_atmosphere is None else str(args.extend_atmosphere),
                   n_depth=relaxed.n_depth,
                   physics=(experimental_physics if experimental_physics is not None else
                            "EXPERIMENTAL HM atoms + ideal He2+; no validated nonideal chemical potentials" if args.helium_dimer else
                            "molecular H/He equilibrium + H2-He/H2-H2 CIA + neutral Lyalpha wings; no dense-fluid corrections"
                            if relaxed.metadata.get("mixed_chemical_model") == "molecular-h-he-hm" else
                            "pure-He production EOS/opacity; dense-ionization limitations remain"
                            if args.log_h_he is None else "atomic H/He only; no H2 equilibrium or H2-He CIA"),
                   converged=bool(relaxed.metadata["radiative_equilibrium_converged"]),
                   maximum_all_depth_total_flux_residual=relaxed.metadata["maximum_all_depth_total_flux_residual"],
                   maximum_relative_cell_energy_balance_residual=relaxed.metadata[
                       "maximum_relative_cell_energy_balance_residual"],
                   structure_grid_local_energy_verified=bool(relaxed.metadata[
                       "maximum_relative_cell_energy_balance_residual"] < 3e-3),
                   independent_local_energy_verified=None,
                   spectrum_integral_over_sigma_teff4=spectrum_ratio,
                   independent_spectrum_flux_verified=bool(abs(spectrum_ratio-1) < .003),
                   bottom_absorption_escape_bound=float(leakage), metadata=_jsonable(relaxed.metadata))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "metadata"}), flush=True)


if __name__ == "__main__":
    main()
