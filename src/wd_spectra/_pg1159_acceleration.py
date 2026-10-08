"""Population acceleration for the stiff PG1159 atom; no shared-solver changes."""
import numpy as np


class PopulationHistory(list):
    """Retain secant differences across small fixed-temperature EOS updates.

    A new EOS changes the map's residual offset. Retain only differences from
    within each segment, never the artificial jump between the two maps.
    """

    def __init__(self):
        super().__init__()
        self.retained = []

    def secants(self, depth):
        pairs = self.retained + [
            (b[0] - a[0], b[1] - a[1]) for a, b in zip(self[:-1], self[1:])
        ]
        return pairs[-(depth - 1):] if depth > 1 else []

    def restart(self, depth, *, retain):
        self.retained = self.secants(depth) if retain else []
        self.clear()

def population_update(
    log_population,
    log_fixed_point,
    history,
    *,
    depth,
    mixing,
    maximum_step,
    residual_weights=None
):
    """Type-II Anderson update using rank-revealing weighted least squares.

    Direct SVD avoids squaring the condition number of the history. The
    shared atom solver's ridge fit suppresses weak population modes that must
    be resolved accurately when populations are eliminated from the thermal
    equations. This PG-specific fit preserves the map, positivity decoding,
    convergence floors, and maximum extrapolation step.
    """
    residual = np.asarray(log_fixed_point - log_population)
    if depth < 2:
        return None, "history"
    history.append((log_population.copy(), residual.copy()))
    if len(history) > depth:
        del history[:-depth]
    pairs = (history.secants(depth) if isinstance(history, PopulationHistory)
             else [(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(history[:-1], history[1:])])
    if len(pairs) < 2:
        return None, "history"
    dx = np.column_stack([pair[0].ravel() for pair in pairs])
    df = np.column_stack([pair[1].ravel() for pair in pairs])
    f = residual.ravel()
    weights = (
        np.ones_like(f)
        if residual_weights is None
        else np.asarray(residual_weights).ravel()
    )
    if weights.shape != f.shape or np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError(
            "PG1159 acceleration weights must be finite, nonnegative and match the state"
        )
    weighted_df = weights[:, None] * df
    weighted_f = weights * f
    try:
        coefficient = np.linalg.lstsq(weighted_df, weighted_f, rcond=1e-8)[0]
    except np.linalg.LinAlgError:
        return None, "linear_solve"
    if np.any(~np.isfinite(coefficient)):
        return None, "coefficient"
    predicted = weighted_f - weighted_df @ coefficient
    if np.linalg.norm(predicted) >= np.linalg.norm(weighted_f):
        return None, "prediction"
    step = mixing * f - (dx + mixing * df) @ coefficient
    # Rows below the least-squares resolution carry no reliable extrapolation.
    # Advance those populations with the ordinary physical map, so their
    # unconstrained history cannot veto the resolved atom's bounded update.
    step = np.where(weights < 1e-8, mixing * f, step)
    important_step = step[weights >= 1.0]
    if np.any(~np.isfinite(step)) or (
        important_step.size and np.max(abs(important_step)) > maximum_step
    ):
        return None, "step"
    # Below the convergence floor, an ill-determined log increment must not
    # reject the well-resolved atom. Bound those increments individually;
    # wholly unresolved rows retain the ordinary physical-map update above.
    tails = (weights >= 1e-8) & (weights < 1.0)
    step[tails] = np.clip(step[tails], -maximum_step, maximum_step)
    return (
        np.ascontiguousarray(log_population + step.reshape(log_population.shape)),
        "accepted",
    )
