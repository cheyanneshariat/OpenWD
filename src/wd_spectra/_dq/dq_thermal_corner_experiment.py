"""Opt-in EOS-corner-aware DQ thermal proposals with unchanged physical gates.

Use the ordinary proposal unless it fails the transient model target at an
EOS corner. Then compare the existing fixed-corner subspace and adjoining
table-piece proposals. The original shared outer solve still evaluates and
accepts full physical states. Supports saved-state and cold forcing drivers.
"""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import automatic_conditioning as controller
from .dq_eos_corner_subspace import corner_columns, refine_corner_subspace
from .dq_thermal_corner_trial import search_thermal_pieces


def repair_proposal(system, model, state, fun, jac, solved, bounds, relative_limit):
    """Retain ordinary successful proposals and original trial-model merit."""
    initial_norm = float(np.linalg.norm(fun(np.zeros_like(state))))
    original_norm = float(np.linalg.norm(fun(solved.x)))
    info = dict(mode='ordinary', initial_norm=initial_norm,
        original_norm=original_norm, selected_norm=original_norm,
        corners=[], atmosphere_certified=False)
    if initial_norm == 0. or original_norm < relative_limit*initial_norm:
        return solved, info
    columns = corner_columns(system, state[:system.n]+solved.x[:system.n])
    if not len(columns):
        return solved, info
    info['corners'] = columns.tolist()
    radius = float(bounds[1][0])
    evaluated = lambda d:SimpleNamespace(residual=fun(d), jacobian=jac(d))
    subspace, detail = refine_corner_subspace(system, state, solved, radius,
        evaluated, bounds=bounds)
    info['subspace'] = detail
    selected = solved

    def consider(candidate, mode):
        nonlocal selected
        delta = candidate.x
        rounding = 64*np.finfo(float).eps*np.maximum(1., abs(state))
        if (np.any(~np.isfinite(delta)) or np.any(delta < bounds[0]-rounding)
                or np.any(delta > bounds[1]+rounding)):
            return
        full = model.evaluate(state+delta, False)
        if not controller.nm.admissible_temperature_gradient(full,
                getattr(system, 'minimum_temperature_gradient_index', None)):
            return
        norm = float(np.linalg.norm(fun(delta)))
        if np.isfinite(norm) and norm < info['selected_norm']:
            selected = candidate
            info.update(mode=mode, selected_norm=norm)

    consider(subspace, 'corner-subspace')
    if info['selected_norm'] >= relative_limit*initial_norm:
        def model_rows(candidate, ev):
            delta = candidate-state
            return fun(delta), jac(delta)
        best, trials = search_thermal_pieces(system, model, state, model_rows,
            solved.x, radius)
        info['pieces'] = trials
        if best is not None:
            _, candidate, delta, _ = best
            candidate.x = delta
            consider(candidate, 'corner-pieces')
    return selected, info


@contextmanager
def corner_aware_thermal_proposals(emit):
    original_model = controller.nm.local_model
    original_ls = controller.least_squares
    current = {}

    def model(system, state, ev):
        result = original_model(system, state, ev)
        current.update(system=system, model=result, state=state.copy())
        return result

    def solve(fun, origin, *, jac, **options):
        original = original_ls(fun, origin, jac=jac, **options)
        selected, info = repair_proposal(current['system'], current['model'],
            current['state'], fun, jac, original, options['bounds'],
            controller.THERMAL_RELATIVE_NORM_LIMIT.get())
        emit(info)
        return selected

    with patch.object(controller.nm, 'local_model', model), \
            patch.object(controller, 'least_squares', solve):
        try:
            # The observer owns the last system/model, including its physical
            # radiation anchor. Release it between completed domain solves;
            # carrying that anchor into a deeper domain wastes hundreds of
            # MiB and cannot help any new-domain proposal.
            yield current.clear
        finally:
            current.clear()



