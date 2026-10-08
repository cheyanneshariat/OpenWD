"""Refresh stale Newton tangents only after hot-NLTE recovery activates.

Extra Jacobians are recorded separately from the common driver's counters.
The dormant solver path and all acceptance/certification criteria are retained.
"""
import logging

from ._hot_recovery import RejectedDirectionFallback

LOGGER = logging.getLogger(__name__)


class FreshTangentFallback(RejectedDirectionFallback):
    def __init__(self, proposals, equations):
        super().__init__(proposals)
        self.equations = equations
        self.extra_jacobian_evaluations = 0

    def direction(self, x, evaluation, jacobian, radius):
        if self.active and self.equations.jacobian_key != x.tobytes():
            # A projected accepted state can make the old tangent uphill.
            # Refresh at that state before building either the temperature
            # direction or its population-restoration operator.
            LOGGER.info('Refreshing the current hot recovery tangent')
            evaluation = self.proposals.evaluate(x, True)
            jacobian = evaluation.jacobian
            self.extra_jacobian_evaluations += 1
        return super().direction(x, evaluation, jacobian, radius)
