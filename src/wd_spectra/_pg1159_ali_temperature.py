"""TMAP-style hybrid temperature correction for the full-NLTE PG1159 structure.

The shared Newton path eliminates the populations and differentiates the flux
profile with a fixed-radiation tangent.  Near the PG 1159-035 root that tangent
misses the non-local population-radiation coupling and drives single cells into
spurious cold or hot states.  TMAP instead iterates

    formal solution -> local (per-depth) correction with an approximate
    lambda operator -> population update

so the non-local coupling is carried by the repeated formal solutions rather
than by a global Jacobian (Werner & Dreizler 1999; Werner et al. 2003; TMAP
User's Guide).  This module supplies the temperature step of that loop.

* Upper cells (emission below the stellar flux): local radiative equilibrium
  ``int (kappa J - eta) = 0`` linearized in the cell temperature, with the
  populations, charge and radiation re-solved together at each depth with
  ``J = J0 + Lambda* (S - S0)`` (the per-depth linearization of TMAP with the
  populations eliminated locally).
* Deeper cells: the plane-parallel Unsöld-Lucy correction in flux-mean optical
  depth (Lucy 1964), in its NLTE form with the actual emissivity:
  ``dB = [kappa_J J - int eta + kappa_J dJ] / kappa_P`` with
  ``dJ = 3 int_0^tau (H0 - H) dtau' + 2 (H0 - H(0))``.  TMAP's guide notes that
  Unsöld-Lucy is reliable in the inner atmosphere but not in the line-forming
  outer layers, which is why those use the local lambda-operator form.

All quantities are per unit mass on the structure wavelength grid.
"""
from dataclasses import replace
import logging

import numpy as np

from ._compat import trapezoid
from ._mass_feautrier import mass_width
from ._pg1159_transfer import transfer_field
from .constants import STEFAN_BOLTZMANN
from .opacity import optical_depth_from_mass_opacity
from .radiative_transfer import radiation_field

LOGGER = logging.getLogger(__name__)


def formal_state(equations, atmosphere, populations):
    """Coefficients, formal field and diagonal lambda operator at a state."""
    model = equations.model
    c = model.transfer_coefficients(atmosphere, equations.wave, populations)
    _, field, _ = transfer_field(
        atmosphere, c, n_angle=model.helium_model.population_n_angle, check_source=False
    )
    total = c.true_absorption + c.scattering
    source = (c.thermal_emissivity + c.scattering * field.mean_intensity) / total
    tau = optical_depth_from_mass_opacity(atmosphere.column_mass, total)
    increment = np.diff(tau, axis=-1)
    floor = 1e-14 * np.maximum(abs(tau[:, 1:]), 1e-300)
    if np.any(increment <= floor):
        tau = np.concatenate(
            (tau[:, :1], tau[:, :1] + np.cumsum(np.maximum(increment, floor), axis=-1)),
            axis=-1,
        )
    operator = radiation_field(
        tau, source, n_angle=model.helium_model.population_n_angle,
        calculate_diagonal_lambda=True,
    ).diagonal_lambda
    return c, field, operator


def local_heating_derivative(equations, atmosphere, populations, operator, mean, width, step=1e-3):
    """d(cell heating)/d ln T from the local ALO-linearized system.

    Every depth is perturbed together.  At each perturbed temperature the rate
    equations are re-solved with the radiation following the local source
    function through the diagonal operator, J = J0 + Lambda* (S - S0), to
    local self-consistency (populations, charge and J together, as in TMAP's
    per-depth linearization).  The cell heating is then evaluated with that
    same locally coupled J on the structure grid.
    """
    from ._pg1159_structure import rate_response_material

    model = equations.model
    wave = equations.wave
    c0 = model.transfer_coefficients(atmosphere, wave, populations)
    chi0 = c0.true_absorption + c0.scattering
    source0 = (c0.thermal_emissivity + c0.scattering * mean) / chi0
    heating = []
    for sign in (1.0, -1.0):
        t = atmosphere.temperature * np.exp(sign * step)
        a, p = rate_response_material(model, atmosphere, t, atmosphere.gas_pressure, populations,
                                      local_lambda_operator=True)
        c = model.transfer_coefficients(a, wave, p)
        chi = c.true_absorption + c.scattering
        local_mean = (mean + operator * (c.thermal_emissivity / chi - source0)) / np.maximum(
            1.0 - operator * c.scattering / chi, 1e-12)
        heating.append(width * trapezoid(
            (c.true_absorption * local_mean - c.thermal_emissivity)[:, :-1], wave, axis=0))
    return (heating[0] - heating[1]) / (2.0 * step)


def temperature_correction(
    equations,
    atmosphere,
    populations,
    *,
    band_last=None,
    band_first=None,
    local_maximum_rosseland_depth=1.0,
    maximum_step=0.05,
    derivative_step=1e-3,
    evaluation=None,
    previous=None,
    local_deadband=0.0,
    probe_step=None,
):
    """Return (d ln T per depth, diagnostics) for one hybrid correction.

    ``evaluation`` is the private diagnostics of ``equations.evaluate`` at this
    state; its transfer coefficients and radiation field are reused instead of
    a second formal solution.  ``previous`` is the ``diagnostics`` of the
    preceding correction.  With it, the upper-cell derivative is the measured
    per-cell secant of the relative heating along the actual iterates.  At
    tau 1e-5..1e-3 the ALO-local derivative overestimates the collective
    response by 10-40x (PG 1424 scan), so its steps are far too short, and in
    the first band cells it can even have the wrong sign.  The secant is used
    with either sign.  The ALO derivative
    is used only when no usable secant exists, unless ``probe_step`` is given:
    such cells then move by ``probe_step`` in ln T in the direction that
    reduces their imbalance, which seeds the secant without the two extra
    full-depth rate solves.  Cells with |relative heating| < ``local_deadband``
    are left unchanged.
    """
    wave = equations.wave
    target = equations.target  # sigma Teff^4
    if evaluation is not None and "_radiation_field" in evaluation:
        c, field = evaluation["_transfer_coefficients"], evaluation["_radiation_field"]
        operator = None
    else:
        c, field, operator = formal_state(equations, atmosphere, populations)
    kappa, eta, sigma = c.true_absorption, c.thermal_emissivity, c.scattering
    chi = kappa + sigma
    mean = field.mean_intensity
    nd = atmosphere.n_depth
    width = 4.0 * np.pi * mass_width(atmosphere.column_mass)  # cells 0..nd-2
    heating = width * trapezoid(kappa[:, :-1] * mean[:, :-1] - eta[:, :-1], wave, axis=0)
    emission = width * trapezoid(eta[:, :-1], wave, axis=0)
    relative = heating / np.maximum(abs(emission), 1e-30 * target)
    flux = trapezoid(field.interface_flux, wave, axis=0)  # interfaces 0..nd-1
    tau_ross = np.asarray(atmosphere.rosseland_optical_depth)
    x = np.log(np.asarray(atmosphere.temperature, dtype=float))
    if band_last is None:
        # Local (differential) RE through the line-forming atmosphere; the
        # Unsöld-Lucy flux form only where TMAP finds it reliable (deep).
        band_last = int(np.argmax(np.r_[tau_ross[:-1] >= local_maximum_rosseland_depth, True]))
    if band_first is None:
        # Cells above the certificate's local-energy window are
        # cancellation-prone; they follow the first gated cell.
        from ._pg1159_structure import _local_energy_mask
        band_first = int(np.argmax(np.r_[_local_energy_mask(atmosphere, equations.model), True]))

    # Upper cells: local radiative equilibrium, d(relative heating)/d ln T
    # from the secant of the last two iterates where it is usable.
    cells = np.arange(min(band_last, nd - 1))
    slope = np.full(nd - 1, np.nan)
    source = np.full(nd - 1, "alo", dtype=object)
    if previous is not None:
        dx = x[:-1] - previous["log_temperature"][:-1]
        dr = relative - previous["relative_heating"]
        with np.errstate(divide="ignore", invalid="ignore"):
            secant = dr / dx
        remembered = previous.get("slope")
        # A secant is refreshed only when both the step and the change in
        # relative heating are resolved; near balance the heating change is
        # at the evaluation noise (~deadband) and the quotient is meaningless.
        fresh = (abs(dx) > 1e-3) & (abs(dr) > max(local_deadband, 1e-6))
        slope = np.where(fresh, secant, remembered if remembered is not None else np.nan)
        source[:] = np.where(fresh, "secant", "kept")
    # The measured secant is used whatever its sign.  Locally, hotter means
    # more net cooling (negative slope), but at tau ~ 1e-5 the collective
    # response of the first band cells can be positive (a thermally unstable
    # branch); their root is then on the hot side, and forcing the local sign
    # only cools them indefinitely.  Only the ALO / probe seed assumes the
    # local sign.
    usable = np.isfinite(slope) & (slope != 0.0) & (source != "alo")
    if probe_step is not None and not np.all(usable[cells]):
        # d ln T = -relative / slope  ==  -sign(relative) * probe_step
        probe_slope = -abs(relative) / probe_step
        slope = np.where(usable, slope, probe_slope)
        source = np.where(usable, source, "probe")
    elif not np.all(usable[cells]):
        if operator is None:
            _, _, operator = formal_state(equations, atmosphere, populations)
        alo = local_heating_derivative(
            equations, atmosphere, populations, operator, mean, width, derivative_step
        ) / np.maximum(abs(emission), 1e-30 * target)
        slope = np.where(usable, slope, alo)
        source = np.where(usable, source, "alo")
    local_step = np.zeros(nd)
    with np.errstate(divide="ignore", invalid="ignore"):
        local_step[cells] = -relative[cells] / slope[cells]
    local_step[~np.isfinite(local_step)] = 0.0
    local_step[cells[abs(relative[cells]) < local_deadband]] = 0.0

    # Deeper cells: Unsöld-Lucy in flux-mean optical depth.
    planck_total = STEFAN_BOLTZMANN * atmosphere.temperature ** 4 / np.pi
    from .spectrum import planck_lambda_angstrom

    planck = planck_lambda_angstrom(wave[:, None], atmosphere.temperature[None, :])
    kappa_p = trapezoid(kappa * planck, wave, axis=0) / planck_total
    mean_total = trapezoid(mean, wave, axis=0)
    kappa_j = trapezoid(kappa * mean, wave, axis=0) / np.maximum(mean_total, 1e-300)
    emission_total = trapezoid(eta, wave, axis=0)
    node_flux = np.empty_like(field.interface_flux)
    node_flux[:, :-1] = 0.5 * (field.interface_flux[:, :-1] + field.interface_flux[:, 1:])
    node_flux[:, -1] = field.interface_flux[:, -1]
    node_flux_total = trapezoid(node_flux, wave, axis=0)
    chi_f = trapezoid(chi * node_flux, wave, axis=0) / np.where(
        abs(node_flux_total) > 0, node_flux_total, 1.0
    )
    chi_f = np.where(chi_f > 0, chi_f, trapezoid(chi * planck, wave, axis=0) / planck_total)
    tau_f = np.r_[chi_f[0] * atmosphere.column_mass[0],
                  chi_f[0] * atmosphere.column_mass[0]
                  + np.cumsum(0.5 * (chi_f[1:] + chi_f[:-1]) * np.diff(atmosphere.column_mass))]
    h0 = target / (4.0 * np.pi)
    delta_h = h0 - flux / (4.0 * np.pi)  # at interfaces
    # H is constant between nodes i-1 and i, equal to its interface-i value.
    dtau = np.diff(np.r_[0.0, tau_f])
    delta_j = 3.0 * np.cumsum(delta_h * dtau) + 2.0 * delta_h[0]
    delta_b = (kappa_j * mean_total - emission_total + kappa_j * delta_j) / kappa_p
    lucy_step = delta_b / (4.0 * planck_total)

    step = np.where(np.arange(nd) < band_last, local_step, lucy_step)
    if previous is not None:
        previous_step = previous.get(
            "step", np.clip(previous["raw_step"], -maximum_step, maximum_step))
        # A cell whose step reverses direction (its imbalance changed sign,
        # or its noisy secant did) is limited to half the previous step, so
        # oscillations of the collectively coupled first band cells contract.
        flipped = np.zeros(nd, dtype=bool)
        flipped[cells] = (np.sign(step[cells]) * np.sign(previous_step[cells])) < 0
        limit = 0.5 * abs(previous_step)
        step = np.where(flipped, np.clip(step, -limit, limit), step)
    step[:band_first] = step[band_first]
    raw = step.copy()
    step = np.clip(step, -maximum_step, maximum_step)
    diagnostics = {
        "maximum_flux_residual": float(np.max(abs(flux / target - 1.0))),
        "surface_flux_residual": float(flux[0] / target - 1.0),
        "relative_heating": relative,
        "log_temperature": x,
        "slope": slope,
        "slope_source": source,
        "raw_step": raw,
        "step": step,
        "band_last": int(band_last),
        "band_first": int(band_first),
    }
    return step, diagnostics


# Line-core velocity stencil for the refinement's structure grid.  The PG1424
# preset's nine samples leave the ~9 km/s oxygen thermal core with three
# points, which biases the energy balance of the outermost cells by up to
# 1.5e-2; this 17-point stencil agrees with a 29-point one to ~1e-4.
REFINEMENT_LINE_VELOCITY_SAMPLES_KMS = (
    -1200.0, -600.0, -300.0, -150.0, -80.0, -40.0, -20.0, -10.0, 0.0,
    10.0, 20.0, 40.0, 80.0, 150.0, 300.0, 600.0, 1200.0,
)


def refine_upper_atmosphere(
    atmosphere,
    populations,
    model,
    wave,
    *,
    maximum_iterations=30,
    maximum_step=0.2,
    probe_step=0.02,
    local_deadband=2e-5,
    local_maximum_rosseland_depth=1e-2,
    material_tolerance=1e-3,
    coarse_material_tolerance=3e-3,
    coarse_step=0.03,
    residual_stop=6e-4,
    residual_stop_step=0.03,
    stationary_step=0.01,
    flux_tolerance=1e-2,
    local_energy_tolerance=3e-3,
    include_radiative_acceleration=False,
    hydrostatic_tolerance=1e-3,
    iteration_callback=None,
):
    """Converge the upper atmosphere to local radiative equilibrium.

    Hybrid TMAP-style iteration on a full-NLTE PG1159 structure: at each
    temperature the populations are closed by plain Lambda iteration (to
    ``material_tolerance``); cells above tau_Ross = ``local_maximum_rosseland_depth``
    take a local-RE step from the measured per-cell secant
    (:func:`temperature_correction`), deeper cells an Unsöld-Lucy flux step.
    It stops when the flux and local-energy gates pass and either every step
    is below ``stationary_step``, or every band cell's relative heating is
    below ``residual_stop`` with every step below ``residual_stop_step``.
    While the last step exceeded ``coarse_step`` the populations are closed
    only to ``coarse_material_tolerance``: large temperature steps are not
    worth closing tightly, and a stop is accepted only at the fine tolerance.
    With ``include_radiative_acceleration`` the gas pressure is iterated
    alongside: each step sets ln P to its hydrostatic target for
    ``g - g_rad`` at the current state, and a stop also requires
    ``|ln P/P_target| < hydrostatic_tolerance``.

    Returns ``(atmosphere, populations, info)``.
    """
    from ._pg1159_structure import PG1159Equations
    from .nonlinear import RecoverableEvaluationError

    stage = replace(
        model,
        population_nlte_fraction=1.0,
        metal_population_damping=1.0,
        helium_population_damping=1.0,
        metal_population_acceleration_depth=0,
        use_pg1159_response_jacobian=True,
        use_population_ali=False,
        metal_population_relative_tolerance=material_tolerance,
    )
    equations = PG1159Equations(
        atmosphere, stage, wave, radiative_acceleration=include_radiative_acceleration,
        certification_stage=False, material_tolerance_ceiling=material_tolerance,
        resolved_material_closure=True,
    )
    equations.population_ali = False
    equations.anchor = populations
    nd = atmosphere.n_depth
    x = equations.initial_state()   # ln T, then ln P with radiative acceleration
    previous = None
    band_first = band_last = None
    history = []
    converged = False
    step = None
    current = None
    fine_stage = stage
    coarse_stage = replace(stage, metal_population_relative_tolerance=coarse_material_tolerance)
    fine = True
    base = x
    for iteration in range(maximum_iterations):
        # Close coarsely at the start and while the temperature is still
        # moving by more than coarse_step; the anchor populations carry over
        # between tolerances.
        fine = previous is not None and float(np.max(abs(previous["raw_step"]))) <= coarse_step
        tolerance = material_tolerance if fine else coarse_material_tolerance
        equations.model = fine_stage if fine else coarse_stage
        equations.material_tolerance_ceiling = tolerance
        # x = base + step; an inadmissible trial state halves the step.
        for halving in range(4):
            try:
                evaluation = equations.evaluate(x, False)
                break
            except RecoverableEvaluationError:
                if step is None or halving == 3:
                    raise
                step = 0.5 * step
                x = base + step
                LOGGER.info("PG1159 refinement: inadmissible step, halving")
        a, p, d = evaluation.payload
        equations.anchor = p
        current = (a, p, d)
        flux = float(d["maximum_all_depth_total_flux_residual"])
        surface = abs(float(d["surface_flux_ratio"]) - 1.0)
        local = float(d["maximum_relative_cell_energy_balance_residual"])
        hydrostatic = (
            float(np.max(abs(evaluation.residual[-nd:])))
            if include_radiative_acceleration else 0.0
        )
        gates = (flux < flux_tolerance and surface < flux_tolerance
                 and local < local_energy_tolerance and hydrostatic < hydrostatic_tolerance)
        stationary = False
        if previous is not None:
            band = slice(previous["band_first"], previous["band_last"])
            largest_step = float(np.max(abs(previous["raw_step"])))
            stationary = fine and (
                largest_step < stationary_step
                or (
                    float(np.max(abs(previous["relative_heating"][band]))) < residual_stop
                    and largest_step < residual_stop_step
                )
            )
        history.append(dict(iteration=iteration, all_depth_flux=flux, surface_flux=surface,
                            local_energy=local, hydrostatic=hydrostatic,
                            gates=bool(gates), stationary=bool(stationary)))
        LOGGER.info("PG1159 refinement %d: flux %.4g, surface %.4g, local energy %.4g, hydrostatic %.3g, "
                    "gates %s, stationary %s, closure %.0e",
                    iteration, flux, surface, local, hydrostatic, gates, stationary, tolerance)
        if iteration_callback is not None:
            iteration_callback(iteration, a, p, d)
        if gates and stationary:
            converged = True
            break
        step, diagnostics = temperature_correction(
            equations, a, p, band_last=band_last, band_first=band_first, evaluation=d,
            previous=previous, maximum_step=maximum_step, local_deadband=local_deadband,
            probe_step=probe_step, local_maximum_rosseland_depth=local_maximum_rosseland_depth,
        )
        band_first, band_last = diagnostics["band_first"], diagnostics["band_last"]
        previous = diagnostics
        if include_radiative_acceleration:
            # Hydrostatic fixed point: move ln P to its target at this state.
            step = np.r_[step, -evaluation.residual[-nd:]]
        base = x
        x = base + step
    a, p, d = current
    info = dict(converged=converged, iterations=len(history), history=tuple(history),
                maximum_step=maximum_step, probe_step=probe_step, local_deadband=local_deadband,
                local_maximum_rosseland_depth=local_maximum_rosseland_depth,
                material_tolerance=material_tolerance,
                coarse_material_tolerance=coarse_material_tolerance, coarse_step=coarse_step,
                residual_stop=residual_stop, residual_stop_step=residual_stop_step,
                stationary_step=stationary_step)
    return a, p, info
