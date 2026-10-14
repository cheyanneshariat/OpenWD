"""Lower-boundary mesh adaptation inside one atmosphere calculation.

Only an already stationary solve is extended. No external atmosphere is read,
no neighboring stellar model is used, and no material closure is changed.
Appended temperatures are provisional guesses, never certified solutions.
"""

from dataclasses import replace
import logging
import numpy as np

_LOGGER = logging.getLogger(__name__)


class LowerBoundaryTooShallow(Exception):
    """Raised by a solver whose first phase already shows a leaking bottom.

    Carries the first-phase atmosphere, its absorption-escape bound and the
    iterations spent, so the caller can deepen the mesh instead of finishing
    a full solve on a domain that cannot be certified.
    """

    def __init__(self, atmosphere, escape_bound, iterations=0):
        super().__init__(
            f"lower-boundary absorption escape {escape_bound:.3g} after the first solver phase"
        )
        self.atmosphere = atmosphere
        self.escape_bound = float(escape_bound)
        self.iterations = int(iterations)


def append_lower_domain(
    atmosphere, pressure_factor=2.0, *, maximum_new_cells=None
):
    """Preserve existing nodes and append one bounded hydrostatic interval.

    Extrapolation is only an initial guess for the new cells. The next full
    atmosphere solve reconstructs their EOS, opacities, convection and transfer.
    Median logarithmic spacing avoids propagating a compressed bottom cell.
    """
    p, t, tau = (
        np.asarray(x, dtype=float)
        for x in (
            atmosphere.gas_pressure,
            atmosphere.temperature,
            atmosphere.rosseland_optical_depth,
        )
    )
    if (
        p.ndim != 1
        or p.size < 2
        or any(
            x.shape != p.shape or np.any(~np.isfinite(x)) or np.any(x <= 0)
            for x in (p, t, tau)
        )
        or np.any(np.diff(p) <= 0)
        or not np.isfinite(pressure_factor)
        or pressure_factor <= 1
    ):
        raise ValueError(
            "domain extension requires positive ordered physical nodes"
        )
    lp = np.log(p)
    spacing = float(np.median(np.diff(lp)))
    intervals = max(1, int(np.ceil(np.log(pressure_factor) / spacing)))
    if (
        maximum_new_cells is not None
        and (
            isinstance(maximum_new_cells, bool)
            or not isinstance(maximum_new_cells, int)
            or maximum_new_cells < 1
        )
    ):
        raise ValueError("maximum new cells must be a positive integer or None")
    offsets = np.linspace(0.0, np.log(pressure_factor), intervals + 1)[1:]
    if maximum_new_cells is not None:
        offsets = offsets[:maximum_new_cells]
    gradient = float((np.log(t[-1]) - np.log(t[-2])) / (lp[-1] - lp[-2]))
    # A negative lower slope is not a justified diffusion extrapolation.
    if gradient < 0:
        raise ValueError("cannot extend a temperature-inverted lower boundary")
    new_p = np.r_[p, p[-1] * np.exp(offsets)]
    new_t = np.r_[t, t[-1] * np.exp(gradient * offsets)]
    # The Rosseland coordinate is provisional, too; it is recomputed on exit.
    new_tau = np.r_[tau, tau[-1] * np.exp(offsets)]
    new_m = np.r_[
        atmosphere.column_mass,
        atmosphere.column_mass[-1]
        + (new_p[p.size :] - p[-1]) / atmosphere.gravity,
    ]
    return dict(
        n_depth=len(new_p),
        initial_temperature=new_t,
        initial_gas_pressure=new_p,
        initial_column_mass=new_m,
        initial_rosseland_optical_depth=new_tau,
        resume_supplied_structure_in_formal_flux_phase=True,
    )


def solve_with_screened_boundary(
    solver,
    *args,
    maximum_domain_expansions=4,
    lower_domain_extension=None,
    deepen_on_any_screen_failure=False,
    restart_after_deepening=False,
    deepening_pressure_factor=2.0,
    early_deepening_escape_limit=None,
    **options,
):
    """Complete a cold/requested solve with bounded same-run domain adaptation.

    The stopping test uses the independently computed absorption-only escape
    bound at the relaxed lower boundary, not the original gray seed's depth.
    Re-solving an enlarged mesh is part of this calculation, not continuation
    from another model. Exhausting the bound returns an uncertified atmosphere.

    By default the mesh is deepened only when lower-boundary screening is the
    sole failed check, and the deeper solve resumes in the formal-flux phase.
    ``deepen_on_any_screen_failure`` also deepens when other checks failed with
    it: a bottom that leaks radiation can itself prevent equilibrium, as in
    cool low-gravity helium, whose transparent windows reach far below a
    Rosseland-scaled seed. ``restart_after_deepening`` re-runs the full phase
    sequence on the deeper mesh, because the extrapolated new cells start far
    from the convective gradient they need. ``deepening_pressure_factor`` sets
    the bottom-pressure increase per extension of the default extension.
    ``early_deepening_escape_limit`` asks the solver to stop after its first
    phase when the escape bound already exceeds this limit (far above the
    screening tolerance), and deepens immediately; it is not passed on the last
    allowed extension, which always completes a full solve.
    """
    if (
        isinstance(maximum_domain_expansions, bool)
        or not isinstance(maximum_domain_expansions, int)
        or maximum_domain_expansions < 0
    ):
        raise ValueError(
            "maximum domain expansions must be a nonnegative integer"
        )
    if lower_domain_extension is None:
        lower_domain_extension = append_lower_domain
    if not np.isfinite(deepening_pressure_factor) or deepening_pressure_factor <= 1.0:
        raise ValueError("deepening pressure factor must be finite and greater than one")
    if early_deepening_escape_limit is not None and (
        not np.isfinite(early_deepening_escape_limit) or early_deepening_escape_limit <= 0.0
    ):
        raise ValueError("early deepening escape limit must be None or finite and positive")
    if not callable(lower_domain_extension):
        raise ValueError("lower domain extension must be callable")
    callback = options.pop("iteration_callback", None)
    originally_cold = options.get("initial_temperature") is None
    segments = []
    offset = 0
    for expansion in range(maximum_domain_expansions + 1):

        def report(iteration, atmosphere, diagnostics):
            if callback is not None:
                callback(
                    offset + iteration,
                    atmosphere,
                    {**diagnostics, "domain_expansion": expansion},
                )

        call_options = dict(options)
        if early_deepening_escape_limit is not None and expansion < maximum_domain_expansions:
            call_options["early_lower_boundary_escape_limit"] = early_deepening_escape_limit
        try:
            result = solver(*args, **call_options, iteration_callback=report)
        except LowerBoundaryTooShallow as shallow:
            offset += shallow.iterations
            segments.append(
                dict(
                    n_depth=shallow.atmosphere.n_depth,
                    iterations=shallow.iterations,
                    thermal_sweeps=0,
                    absorption_escape_bound=shallow.escape_bound,
                    early_deepening=True,
                )
            )
            _LOGGER.info(
                "Deepening below %d layers after the first solver phase: "
                "lower-boundary absorption escape %.6g exceeds %.6g",
                shallow.atmosphere.n_depth,
                shallow.escape_bound,
                early_deepening_escape_limit,
            )
            options.update(
                lower_domain_extension(shallow.atmosphere)
                if lower_domain_extension is not append_lower_domain
                else append_lower_domain(shallow.atmosphere, deepening_pressure_factor)
            )
            if restart_after_deepening:
                options["resume_supplied_structure_in_formal_flux_phase"] = False
            continue
        meta = result.metadata
        iterations = int(meta.get("radiative_equilibrium_iterations", 0))
        offset += iterations
        segments.append(
            dict(
                n_depth=result.n_depth,
                iterations=iterations,
                thermal_sweeps=(meta.get("thermal_conditioning") or {}).get(
                    "iterations", 0
                ),
                absorption_escape_bound=meta.get(
                    "lower_boundary_absorption_escape_bound"
                ),
            )
        )
        failures = meta.get("equilibrium_certificate", {}).get("failures", [])
        if deepen_on_any_screen_failure:
            segments[-1]["failures"] = list(failures)
        screen_failed = (
            "boundary_screening" in failures
            if deepen_on_any_screen_failure
            else failures == ["boundary_screening"]
        )
        if not screen_failed or expansion == maximum_domain_expansions:
            break
        _LOGGER.info(
            "Extending the current calculation below %d layers: "
            "lower-boundary absorption escape %.6g exceeds %.6g",
            result.n_depth,
            meta["lower_boundary_absorption_escape_bound"],
            meta["lower_boundary_screening_tolerance"],
        )
        options.update(
            lower_domain_extension(result)
            if lower_domain_extension is not append_lower_domain
            else append_lower_domain(result, deepening_pressure_factor)
        )
        if restart_after_deepening:
            options["resume_supplied_structure_in_formal_flux_phase"] = False
    return replace(
        result,
        metadata={
            **result.metadata,
            "cold_start": originally_cold,
            "external_atmosphere_required": False,
            "adaptive_domain_segments": segments,
            "adaptive_domain_expansions": len(segments) - 1,
            "thermal_sweeps_including_domain_adaptation": sum(
                s["thermal_sweeps"] for s in segments
            ),
            "radiative_equilibrium_iterations_including_domain_adaptation": offset,
        },
    )
