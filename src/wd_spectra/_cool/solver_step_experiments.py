"""Research-only bounded temperature-step constructions, without new physics.

SciPy is intentionally a research-script dependency, not a package dependency.
"""
import logging
import time
import numpy as np
from scipy.optimize import lsq_linear, least_squares
from wd_spectra.constants import STEFAN_BOLTZMANN
from wd_spectra.convection import _ml2_contrast_and_root, _ml2_flux_coefficient_response
from .radiative_rate_proposal import radiative_rates

_LOGGER = logging.getLogger(__name__)
DIRECT_ENERGY_METHODS = ("nonlinear-convection-direct-energy", "nonlinear-convection-current-energy", "nonlinear-convection-cell-energy")
CURRENT_ENERGY_METHODS = ("nonlinear-convection-current-energy", "nonlinear-convection-cell-energy")
LOCAL_ENERGY_METHODS = ("nonlinear-convection-local-energy",) + DIRECT_ENERGY_METHODS


def nonlinear_convection_model(evaluation, cell_scale=None, *, direct_energy=False,
                                current_energy=False, thermal_mass_over_timestep=None, cell_only=False):
    """Cheap trial model: tangent radiation/materials, nonlinear local ML2.

    Its residual is a proposal model only. Every outer trial still rebuilds
    actual thermodynamics, opacity, radiation and convection.
    """
    payload = evaluation.payload
    target = STEFAN_BOLTZMANN * payload["atmosphere"].effective_temperature**4
    radiation = payload["radiative_flux_interface"] / target
    radiation_jacobian = payload["radiative_flux_log_temperature_jacobian"] / target
    gradient = payload["temperature_gradient"]
    gradient_operator = payload["interface_gradient_operator"]
    transport = payload["convection_transport"]
    ad, loss, coefficient = (transport[k] for k in
                             ("adiabatic_gradient", "ml2_radiative_loss", "ml2_flux_coefficient"))
    ad_response, loss_response, coefficient_response = payload["ml2_coefficient_log_temperature_responses"]
    log_loss_response = loss_response / loss[:, None]
    log_coefficient_response = coefficient_response / coefficient[:, None]
    if thermal_mass_over_timestep is not None:
        inertia = np.asarray(thermal_mass_over_timestep, dtype=float)/target
        if (not current_energy or cell_scale is None or inertia.shape != (gradient.size-1,)
                or np.any(inertia <= 0) or np.any(~np.isfinite(inertia))):
            raise ValueError("Thermal inertia requires positive cell masses and current energy rows")
    if cell_scale is not None:
        cell_radiation_jacobian = (
            payload["radiative_cell_energy_log_temperature_jacobian"]/target
            if direct_energy else np.diff(radiation_jacobian, axis=0))
        if current_energy:
            if not direct_energy:
                raise ValueError("Current energy normalization requires direct energy derivatives")
            emission = payload["thermal_cell_emission"]/target
            log_emission_response = (payload["thermal_cell_emission_log_temperature_jacobian"]
                                     / payload["thermal_cell_emission"][:, None])
    def model(delta):
        g = gradient + gradient_operator @ delta
        if 'diagnostic_material_model' in payload:
            (a,b,c), responses = payload['diagnostic_material_model'](delta)
        else:
            a = ad + ad_response @ delta
            b = loss * np.exp(log_loss_response @ delta)
            c = coefficient * np.exp(log_coefficient_response @ delta)
            responses = (ad_response, b[:, None]*log_loss_response, c[:, None]*log_coefficient_response)
        excess = np.maximum(g-a, 0.)
        contrast, root = _ml2_contrast_and_root(excess, b)
        flux = c*contrast**3
        derivative = 3*c*contrast**2/np.maximum(2*root, np.finfo(float).tiny)
        material_jacobian = _ml2_flux_coefficient_response(
            g, (a, b, c), responses)
        convection_jacobian = derivative[:, None]*gradient_operator + material_jacobian
        flux[0] = 0.
        convection_jacobian[0] = 0.
        residual = radiation + radiation_jacobian @ delta + flux/target - 1
        tangent = radiation_jacobian + convection_jacobian/target
        if cell_scale is not None:
            # Use the directly evaluated absorption*(J-B) term at the base
            # state, avoiding subtraction of nearly equal surface fluxes.
            radiative_cell = payload["radiative_cell_energy_defect"]/target
            cell_defect = (radiative_cell + cell_radiation_jacobian@delta
                           + np.diff(flux)/target)
            cell_radiation_response = cell_radiation_jacobian
            normalization = cell_scale
            if direct_energy and payload.get('diagnostic_positive_radiative_rates',False):
                exchange,cell_radiation_response,_,_=radiative_rates(payload,delta)
                cell_defect=exchange+np.diff(flux)/target
            if current_energy:
                exchange,cell_radiation_response,thermal,thermal_response=radiative_rates(payload,delta)
                cell_defect=exchange+np.diff(flux)/target
                normalization = thermal + (flux[:-1]+flux[1:])/target
                normalization_response = (thermal_response
                    + (convection_jacobian[:-1]+convection_jacobian[1:])/target)
            normalized_defect = cell_defect/normalization
            residual[:-1] -= normalized_defect
            tangent[:-1] -= (cell_radiation_response
                            + np.diff(convection_jacobian, axis=0)/target) / normalization[:, None]
            if current_energy:
                tangent[:-1] += normalized_defect[:, None]*normalization_response/normalization[:, None]
            if cell_only:
                return (np.concatenate(([radiation[0]+radiation_jacobian[0]@delta-1],normalized_defect)),
                    np.vstack((radiation_jacobian[0], (cell_radiation_response+np.diff(convection_jacobian,axis=0)/target)
                        /normalization[:,None]-normalized_defect[:,None]*normalization_response/normalization[:,None])))
            if thermal_mass_over_timestep is not None:
                # Backward-Euler-like proposal with Cp frozen at the current
                # physical state: q = Cp*T*Delta_m*(exp(dlnT)-1)/dt. Transform
                # E-q through the SAME flux-minus-cell rows. The bottom flux
                # remains an algebraic boundary condition, without inertia.
                # No transient term is used in the actual outer residual.
                q = inertia*np.expm1(delta[:-1])
                q_response = np.diag(inertia*np.exp(delta[:-1]))
                q_response = np.pad(q_response, ((0, 0), (0, 1)))
                residual[:-1] += np.cumsum(q[::-1])[::-1] + q/normalization
                tangent[:-1] += np.cumsum(q_response[::-1], axis=0)[::-1]
                tangent[:-1] += q_response/normalization[:, None]
                tangent[:-1] -= (q/normalization)[:, None]*normalization_response/normalization[:, None]
        return residual, tangent
    return model


def make_step(state, evaluation, jacobian, radius, method, *, inner_scaling="unit",
              inner_max_evaluations=100, thermal_mass_over_timestep=None,
              inner_row_scaling=False):
    residual = evaluation.residual
    mapping = evaluation.payload["log_temperature_from_state"]
    if method.startswith("nonlinear-convection"):
        cell_scale = evaluation.payload.get("diagnostic_fixed_cell_scale")
        local_energy = method in LOCAL_ENERGY_METHODS
        if local_energy and cell_scale is None:
            raise ValueError("Local-energy proposals require the segment's fixed cell scales")
        model = nonlinear_convection_model(
            evaluation, cell_scale if local_energy else None,
            direct_energy=method in DIRECT_ENERGY_METHODS,
            current_energy=method in CURRENT_ENERGY_METHODS,
            cell_only=method == "nonlinear-convection-cell-energy",
            thermal_mass_over_timestep=thermal_mass_over_timestep)
        # The explicit secant experiment can use a rank-one correction from rejected
        # physical trials. Retain that measured information alongside the
        # nonlinear ML2 model, instead of rebuilding the identical failed
        # proposal from its unchanged payload. A fresh analytic Jacobian is
        # an exact no-op here. This is a proposal correction only; all outer
        # residuals and convergence gates still use the actual physics.
        base_jacobian = getattr(evaluation, 'jacobian', None)
        if (evaluation.payload.get('diagnostic_measured_proposals',False)
                and base_jacobian is not None and not np.array_equal(jacobian, base_jacobian)):
            correction = np.linalg.solve(mapping.T, (jacobian-base_jacobian).T).T
            uncorrected_model = model
            def model(delta):
                values, tangent = uncorrected_model(delta)
                return values+correction@delta, tangent+correction
            _LOGGER.info('Proposal includes measured secant tangent correction; maximum %.6g',
                         np.max(abs(correction)))
        if method == "nonlinear-convection-regularized":
            # Retain the existing Newton driver's row scaling and 1e-8
            # Tikhonov penalty in its original structure coordinates. The
            # bounds still act on actual nodal dlnT. This is a proposal
            # penalty only, not an extra atmosphere equation.
            scale = np.maximum(np.max(abs(jacobian), axis=1), np.finfo(float).tiny)
            penalty = 1e-4*np.linalg.inv(mapping)
            def objective(delta):
                values, tangent = model(delta)
                return (np.concatenate((values/scale, penalty @ delta)),
                        np.vstack((tangent/scale[:, None], penalty)))
        elif method == "nonlinear-convection" or local_energy:
            objective = model
        else:
            raise ValueError(f"Unknown experimental step method {method}")
        if inner_row_scaling:
            unscaled_objective = objective
            proposal_row_scale = np.maximum(np.max(abs(objective(np.zeros_like(state))[1]), axis=1),
                                            np.finfo(float).tiny)
            def objective(delta):
                values, tangent = unscaled_objective(delta)
                return values/proposal_row_scale, tangent/proposal_row_scale[:, None]
        started = time.monotonic()
        if inner_scaling not in ("unit", "jac", "svd", "bvls"):
            raise ValueError("inner_scaling must be unit, jac, svd or bvls")
        if not isinstance(inner_max_evaluations, int) or inner_max_evaluations < 1:
            raise ValueError("inner_max_evaluations must be a positive integer")
        _LOGGER.info("Starting %s inner proposal; trust radius %.6g; scaling=%s; row_scaling=%s",
                     method, radius, inner_scaling, inner_row_scaling)
        if inner_scaling == 'bvls':
            from .bounded_nonlinear_proposal import solve_bounded_model
            result=solve_bounded_model(objective,state.size,radius,inner_max_evaluations)
            # A failed inner solve can still propose a useful outer trial, but
            # a tiny failed proposal must never certify stationarity.
            evaluation.payload['diagnostic_inner_proposal_root_solved']=result.root_solved
            _LOGGER.info('BVLS inner proposal ready in %.2fs; evaluations=%d status=%d '
                'model_max_residual=%.6g max_dlnT=%.6g optimality=%.6g root_solved=%s',
                time.monotonic()-started,result.nfev,result.status,result.residual_maximum,
                np.max(abs(result.x)),result.optimality,result.root_solved)
            return np.linalg.solve(mapping,result.x)
        # Research-only rotation separates weak global temperature modes from
        # stiff adjacent-layer ML2 differences. No residual row is changed;
        # the nonlinear coordinate map below enforces nodal, not rotated, bounds.
        basis = (np.linalg.svd(objective(np.zeros_like(state))[1], full_matrices=False)[2].T
                 if inner_scaling == "svd" else np.eye(state.size))
        # Bounds belong to physical nodal temperatures, regardless of the
        # material callback. A box on SVD coordinates arbitrarily shrinks a
        # uniform thermal mode by sqrt(N), while admitting larger oscillatory
        # modes. Rotation must not change the physical trust region.
        curved_box = inner_scaling == 'svd'
        def coordinates(z):
            raw = basis@z
            if curved_box:
                bounded = np.tanh(raw/radius)
                return radius*bounded, (1-bounded**2)[:,None]*basis
            return raw, basis
        def rotated(z):
            delta, derivative = coordinates(z)
            values, tangent = objective(delta)
            return values, tangent@derivative
        # With tanh enforcing the true nodal bound, an additional box in
        # rotated coordinates is unnecessary and excludes legitimate smooth
        # global corrections by a factor sqrt(N). Let those coordinates be
        # unconstrained; every actual nodal trial remains inside the radius.
        coordinate_bounds = (-np.inf,np.inf) if curved_box else (-radius,radius)
        result = least_squares(lambda delta: rotated(delta)[0], np.zeros_like(state),
                               jac=lambda delta: rotated(delta)[1],
                               bounds=coordinate_bounds, method="trf", max_nfev=inner_max_evaluations,
                               x_scale=1.0 if inner_scaling == "unit" else "jac",
                               ftol=None if inner_scaling == "svd" else 1e-10,
                               xtol=1e-14 if inner_scaling == "svd" else 1e-10, gtol=1e-10)
        temperature_step = coordinates(result.x)[0]
        temperature_step *= min(1., radius/max(np.max(abs(temperature_step)), np.finfo(float).tiny))
        _LOGGER.info("Inner proposal ready in %.2fs; evaluations=%d status=%d "
                     "model_max_residual=%.6g max_dlnT=%.6g depth_index=%d optimality=%.6g",
                     time.monotonic()-started, result.nfev, result.status,
                     np.max(abs(model(temperature_step)[0])), np.max(abs(temperature_step)),
                     np.argmax(abs(temperature_step)), result.optimality)
        return np.linalg.solve(mapping, temperature_step)
    if method == "legacy":
        scale = np.maximum(np.max(abs(jacobian), axis=1), np.finfo(float).tiny)
        matrix = jacobian / scale[:, None]
        rhs = -residual / scale
        step = np.linalg.lstsq(
            np.vstack((matrix, 1e-4*np.eye(state.size))),
            np.concatenate((rhs, np.zeros(state.size))), rcond=1e-10)[0]
        size = np.max(abs(mapping @ step))
        return step * min(1., radius/max(size, np.finfo(float).tiny))

    # Convert the same Jacobian to actual logarithmic nodal temperatures.
    matrix = np.linalg.solve(mapping.T, jacobian.T).T
    rhs = -residual
    if method in ("box-row", "temperature-penalty"):
        scale = np.maximum(np.max(abs(matrix), axis=1), np.finfo(float).tiny)
        matrix = matrix / scale[:, None]
        rhs = rhs / scale
    if method == "temperature-penalty":
        temperature_step = np.linalg.lstsq(
            np.vstack((matrix, 1e-4*np.eye(state.size))),
            np.concatenate((rhs, np.zeros(state.size))), rcond=1e-10)[0]
        temperature_step *= min(1., radius/max(np.max(abs(temperature_step)), np.finfo(float).tiny))
    elif method in ("box", "box-row"):
        result = lsq_linear(matrix, rhs, bounds=(-radius, radius),
                            method="bvls", tol=1e-10)
        if not result.success:
            raise RuntimeError(f"Bounded linear subproblem failed: {result.message}")
        temperature_step = result.x
    else:
        raise ValueError(f"Unknown experimental step method {method}")
    return np.linalg.solve(mapping, temperature_step)
