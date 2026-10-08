"""Lower-boundary mesh adaptation inside one atmosphere calculation.

Only an already stationary solve is extended. No external atmosphere is read,
no neighboring stellar model is used, and no material closure is changed.
Appended temperatures are provisional guesses, never certified solutions.
"""

from dataclasses import replace
import logging
import numpy as np

_LOGGER = logging.getLogger(__name__)


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
    **options,
):
    """Complete a cold/requested solve with bounded same-run domain adaptation.

    The stopping test uses the independently computed absorption-only escape
    bound at the relaxed lower boundary, not the original gray seed's depth.
    Re-solving an enlarged mesh is part of this calculation, not continuation
    from another model. Exhausting the bound returns an uncertified atmosphere.
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

        result = solver(*args, **options, iteration_callback=report)
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
        if (
            failures != ["boundary_screening"]
            or expansion == maximum_domain_expansions
        ):
            break
        _LOGGER.info(
            "Extending the current calculation below %d layers: "
            "lower-boundary absorption escape %.6g exceeds %.6g",
            result.n_depth,
            meta["lower_boundary_absorption_escape_bound"],
            meta["lower_boundary_screening_tolerance"],
        )
        options.update(lower_domain_extension(result))
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
