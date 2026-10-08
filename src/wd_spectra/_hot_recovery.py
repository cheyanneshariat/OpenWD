"""Bounded hot-NLTE recovery proposals for the shared nonlinear driver.

Population restoration handles curved statistical-equilibrium constraints.
For optically thin thermal turning points, a full-transfer root and coupled
neighbor adjustment provide a separate material proposal. The common driver
retains merit acceptance, mandatory Newton attempts and final certification.
Hooks remain dormant until two whole Newton directions have been rejected.
"""
import logging
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve, svd
from scipy.optimize import brentq

from ._compat import trapezoid
from .hot_nlte import transfer_field
from .nonlinear import NonlinearCorrection, RecoverableEvaluationError
from .nlte_core import NonphysicalPopulationError

LOGGER = logging.getLogger(__name__)


class ThermalPopulationCorrection:
    def __init__(self,evaluate,step_builder,step_tolerance):
        self.evaluate=evaluate
        self.step_builder=step_builder
        self.step_tolerance=step_tolerance
        self.anchor=None
        self.extra_residual_evaluations=0

    def direction(self,x,e,j,radius):
        proposal=self.step_builder(x,e,j,radius)
        self.anchor=(x,e,j,radius,proposal)
        return proposal

    def correction(self,x,e):
        nt=e.payload[0].n_depth
        if len(x)<=nt or e.payload[2].get('nlte_continuation_fraction',0.)==0.:
            return None
        anchor,_,j,radius,proposal=self.anchor
        if not np.array_equal(x,anchor):
            raise ValueError('Population correction requires the current Newton anchor')
        if not proposal.limited and max(abs(proposal.direction))<=self.step_tolerance:
            return None
        scale=np.maximum(abs(j[nt:,nt:]).max(axis=1),np.finfo(float).tiny)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',LinAlgWarning)
                factor=lu_factor(j[nt:,nt:]/scale[:,None])
            cross=lu_solve(factor,j[nt:,:nt]/scale[:,None])
            offset=lu_solve(factor,e.residual[nt:]/scale)
            schur=j[:nt,:nt]-j[:nt,nt:]@cross
            reduced=e.residual[:nt]-j[:nt,nt:]@offset
            u,s,vt=svd(schur,lapack_driver='gesvd')
        except (LinAlgWarning,np.linalg.LinAlgError):
            return None
        if np.any(~np.isfinite(s)) or not len(s) or s[0]==0.:
            return None
        rhs=u.T@(-reduced)
        def step(damping):
            dt=vt.T@(s/(s*s+damping)*rhs)
            return np.r_[dt,-offset-cross@dt]
        # Reserve half the box for nonlinear population restoration. If the
        # offset alone cannot fit, defer to the global Newton direction.
        if max(abs(offset))>=.5*radius:
            return None
        lo=0.;hi=max(s[0]**2,np.finfo(float).tiny)
        for _ in range(40):
            if max(abs(step(hi)))<=.5*radius:break
            hi*=10.
        else:return None
        for _ in range(50):
            mid=(lo+hi)*.5
            if max(abs(step(mid)))>.5*radius:lo=mid
            else:hi=mid
        trial_step=step(hi)
        def checked(state):
            self.extra_residual_evaluations+=1
            return self.evaluate(state,False)
        def trial_state(fraction):
            z=x+fraction*trial_step
            for _ in range(4):
                trial=checked(z)
                update=lu_solve(factor,-trial.residual[nt:]/scale)
                if np.any(~np.isfinite(update)):
                    raise RecoverableEvaluationError('Nonfinite population restoration')
                for damping in (1.,.5,.25,.125,.0625):
                    candidate=z.copy();candidate[nt:]+=damping*update
                    if max(abs(candidate-x))>radius:continue
                    try:new=checked(candidate)
                    except RecoverableEvaluationError:continue
                    if np.linalg.norm(new.residual[nt:])<np.linalg.norm(trial.residual[nt:]):
                        z=candidate;break
                else:break
            return z
        return NonlinearCorrection(trial_state)


class ProjectedThermalPopulationCorrection(ThermalPopulationCorrection):
    def direction(self,x,e,j,radius):
        self.projector_factor=None
        return super().direction(x,e,j,radius)

    def project(self,old,trial):
        anchor,e,j,radius,_=self.anchor
        if not np.array_equal(anchor,old):
            raise ValueError('Population projection requires the current Newton anchor')
        nt=e.payload[0].n_depth
        if len(old)<=nt or e.payload[2].get('nlte_continuation_fraction',0.)==0.:
            return trial
        if self.projector_factor is None:
            scale=np.maximum(abs(j[nt:,nt:]).max(axis=1),np.finfo(float).tiny)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error',LinAlgWarning)
                    factor=lu_factor(j[nt:,nt:]/scale[:,None])
            except (LinAlgWarning,np.linalg.LinAlgError):return trial
            self.projector_factor=(factor,scale)
        factor,scale=self.projector_factor
        # Match the central driver's allowance for subtraction roundoff in
        # log coordinates. A temperature step at radius .06 near log T=10
        # can decode as .0600000000000005; it must not disable all repairs.
        bound=radius+128*np.finfo(float).eps*max(1.,float(max(abs(old))),float(max(abs(trial))))
        def checked(state):
            self.extra_residual_evaluations+=1
            return self.evaluate(state,False)
        z=trial.copy()
        for _ in range(4):
            current=checked(z)
            update=lu_solve(factor,-current.residual[nt:]/scale)
            if np.any(~np.isfinite(update)):
                raise RecoverableEvaluationError('Nonfinite trial population repair')
            for damping in (1.,.5,.25,.125,.0625):
                candidate=z.copy();candidate[nt:]+=damping*update
                if max(abs(candidate-old))>bound:continue
                try:new=checked(candidate)
                except RecoverableEvaluationError:continue
                if np.linalg.norm(new.residual[nt:])<np.linalg.norm(current.residual[nt:]):
                    z=candidate;break
            else:break
        return z


def _bracket_root(function, lower, upper, **kwargs):
    root, result = brentq(function, lower, upper, full_output=True, disp=False, **kwargs)
    # An exhausted proposal can be skipped; errors in the physical evaluator
    # propagate unchanged instead of being caught as generic RuntimeError.
    if not result.converged:
        raise RecoverableEvaluationError('Thermal bracket did not settle')
    return root


class BracketedThermalPopulationCorrection(ProjectedThermalPopulationCorrection):
    def __init__(self, evaluate, step_builder, step_tolerance, equations):
        super().__init__(evaluate, step_builder, step_tolerance)
        self.equations = equations
        self.thermal_probes = 0

    def refine_neighbors(self, candidate, anchor, evaluation, jacobian, depth):
        return candidate

    def correction(self, x, e):
        eq = self.equations
        a, population, diagnostics = e.payload
        nt = a.n_depth
        anchor, _, j, radius, proposal = self.anchor
        if not np.array_equal(x, anchor):
            raise ValueError('Thermal bracketing requires the current Newton anchor')
        if (eq.fixed_temperature or eq.nlte_fraction != 1.
                or np.max(abs(e.residual[nt:]), initial=0.) > 1e-5
                or np.max(abs(proposal.direction[:nt]), initial=0.) < .05):
            return super().correction(x, e)
        depth = int(np.argmax(abs(proposal.direction[:nt])))
        # A local thermal balance is inappropriate for the imposed bottom
        # flux equation or a deep cell dominated by nonlocal transport.
        if depth == nt - 1 or a.rosseland_optical_depth[depth] > .1:
            return super().correction(x, e)
        c = eq.model.transfer_coefficients(a, eq.wave, population)
        _, field, _ = transfer_field(a, c, n_angle=eq.model.n_angle, check_source=False)
        del c
        t0 = x[:nt].copy()

        def local(log_temperature, *, state=False):
            self.thermal_probes += 1
            t = t0.copy()
            t[depth] = log_temperature
            try:
                aa, reference, groups, cache = eq.prepare(t)
                candidate, _ = eq.rate_residual(x, aa, reference, groups, field.mean_intensity)
                if state:
                    return np.r_[t, eq.coordinates(candidate)]
                coefficients = eq.model.transfer_coefficients(aa, eq.wave, candidate, _cache=cache)
                heating = trapezoid(coefficients.true_absorption[:, depth] * field.mean_intensity[:, depth]
                                    - coefficients.thermal_emissivity[:, depth], eq.wave)
                emission = trapezoid(coefficients.thermal_emissivity[:, depth], eq.wave)
                return float(heating / max(abs(emission), 1e-30))
            except NonphysicalPopulationError as exc:
                raise RecoverableEvaluationError(str(exc)) from exc
            finally:
                # A scan must not retain a complete material cache per probe.
                eq.prepared.clear()

        grid = t0[depth] + np.array([-.35, -.2, -.1, 0., .1, .2, .35])
        values = []
        for point in grid:
            try:
                values.append(local(point))
            except RecoverableEvaluationError:
                values.append(np.nan)
        brackets = [i for i in range(len(grid)-1)
                    if values[i] > 0. and values[i+1] < 0.]
        if not brackets:
            return super().correction(x, e)
        i = min(brackets, key=lambda k: abs(.5 * (grid[k]+grid[k+1]) - t0[depth]))
        root = _bracket_root(local, grid[i], grid[i+1], xtol=1e-7, rtol=1e-12, maxiter=30)
        LOGGER.info('Bracketed thermal proposal at depth %d: %.2f -> %.2f K',
                    depth, np.exp(t0[depth]), np.exp(root))
        scale = np.maximum(abs(j[nt:, nt:]).max(axis=1), np.finfo(float).tiny)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', LinAlgWarning)
                factorization = lu_factor(j[nt:, nt:] / scale[:, None])
        except (LinAlgWarning, np.linalg.LinAlgError) as exc:
            raise RecoverableEvaluationError('Singular bracketed population tangent') from exc

        def checked(z):
            self.extra_residual_evaluations += 1
            return self.evaluate(z, False)

        def restored(log_temperature):
            z = local(log_temperature, state=True)
            # This is a separate bracketed material correction, not a Newton
            # trust step. Its explicit temperature bound is 0.35 in log T;
            # the original Newton trust radius and stationary test stay intact.
            for _ in range(12):
                current = checked(z)
                if np.max(abs(current.residual[nt:]), initial=0.) < 1e-8:
                    break
                update = lu_solve(factorization, -current.residual[nt:] / scale)
                if np.any(~np.isfinite(update)):
                    raise RecoverableEvaluationError('Nonfinite bracketed population restoration')
                for damping in (1., .5, .25, .125, .0625):
                    candidate = z.copy()
                    candidate[nt:] += damping * update
                    if np.max(abs(candidate[nt:] - x[nt:])) > .5:
                        continue
                    try:
                        new = checked(candidate)
                    except RecoverableEvaluationError:
                        continue
                    if np.linalg.norm(new.residual[nt:]) < np.linalg.norm(current.residual[nt:]):
                        z = candidate
                        break
                else:
                    break
            if np.max(abs(z[nt:] - x[nt:])) > .5:
                raise RecoverableEvaluationError('Bracketed proposal exceeds population bound')
            result = checked(z)
            if np.max(abs(result.residual[nt:]), initial=0.) > 1e-7:
                raise RecoverableEvaluationError('Bracketed population restoration did not settle')
            return z, result

        # The frozen-field root is only a proposed bracket endpoint. Measure
        # and refine its thermal sign using the full transfer/population
        # equations: a local opacity change can move that root appreciably.
        full_cache = {float(t0[depth]): (x.copy(), e)}
        def full(log_temperature):
            key = float(log_temperature)
            if key not in full_cache:
                full_cache[key] = restored(key)
            return float(full_cache[key][1].residual[depth])
        f0, f1 = full(t0[depth]), full(root)
        if f0 * f1 > 0.:
            raise RecoverableEvaluationError('Frozen thermal bracket did not survive full transfer')
        coupled_root = _bracket_root(full, min(root, t0[depth]), max(root, t0[depth]),
                                    xtol=1e-6, rtol=1e-12, maxiter=20)
        LOGGER.info('Full-transfer thermal root at depth %d: %.2f K', depth, np.exp(coupled_root))

        def trial_state(fraction):
            temperature = float(t0[depth] + fraction * (coupled_root-t0[depth]))
            full(temperature)
            candidate = full_cache[temperature][0].copy()
            return self.refine_neighbors(candidate, x, e, j, depth)

        return NonlinearCorrection(trial_state)


class CoupledBracketedThermalPopulationCorrection(BracketedThermalPopulationCorrection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.neighbor_adjustments = 0

    def refine_neighbors(self, candidate, anchor, evaluation, jacobian, depth):
        nt = evaluation.payload[0].n_depth
        j = jacobian
        def checked(state):
            self.extra_residual_evaluations += 1
            return self.evaluate(state, False)
        base = checked(candidate)
        scale = np.maximum(abs(j[nt:, nt:]).max(axis=1), np.finfo(float).tiny)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', LinAlgWarning)
                factor = lu_factor(j[nt:, nt:]/scale[:, None])
            cross = lu_solve(factor, j[nt:, :nt]/scale[:, None])
            offset = lu_solve(factor, base.residual[nt:]/scale)
            schur = j[:nt, :nt] - j[:nt, nt:] @ cross
            rhs = -base.residual[:nt] + j[:nt, nt:] @ offset
            # Hold the selected root temperature fixed while correcting
            # the other thermal equations, including the bottom flux.
            schur[depth, :] = 0.
            schur[depth, depth] = 1.
            rhs[depth] = 0.
            rows = np.maximum(abs(schur).max(axis=1), np.finfo(float).tiny)
            dt, _, rank, _ = np.linalg.lstsq(schur/rows[:, None], rhs/rows, rcond=1e-10)
        except (LinAlgWarning, np.linalg.LinAlgError):
            return candidate
        if rank != nt or np.any(~np.isfinite(dt)):
            return candidate
        dt[depth] = 0.
        trial = candidate + np.r_[dt, -offset-cross@dt]
        trial[depth] = candidate[depth]
        if (max(abs(trial[:nt]-anchor[:nt])) > .35
                or max(abs(trial[nt:]-anchor[nt:])) > .5):
            return candidate
        try:
            for _ in range(12):
                current = checked(trial)
                if max(abs(current.residual[nt:])) < 1e-8:
                    break
                update = lu_solve(factor, -current.residual[nt:]/scale)
                if np.any(~np.isfinite(update)):
                    return candidate
                for damping in (1., .5, .25, .125, .0625):
                    proposed = trial.copy()
                    proposed[nt:] += damping*update
                    if max(abs(proposed[nt:]-anchor[nt:])) > .5:
                        continue
                    try:
                        measured = checked(proposed)
                    except RecoverableEvaluationError:
                        continue
                    if np.linalg.norm(measured.residual[nt:]) < np.linalg.norm(current.residual[nt:]):
                        trial = proposed
                        break
                else:
                    break
            final = checked(trial)
        except RecoverableEvaluationError:
            return candidate
        if (max(abs(final.residual[nt:])) <= 1e-7
                and np.linalg.norm(final.residual) < np.linalg.norm(base.residual)):
            self.neighbor_adjustments += 1
            return trial
        return candidate


class RejectedDirectionFallback:
    def __init__(self, proposals):
        self.proposals = proposals
        self.rejected_directions = 0
        self.active = False

    def direction(self, *args):
        return self.proposals.direction(*args)

    def correction(self, *args):
        return self.proposals.correction(*args) if self.active else None

    def project(self, old, trial):
        return self.proposals.project(old, trial) if self.active else trial

    def rejected(self, state, evaluation):
        self.rejected_directions += 1
        if self.rejected_directions == 2:
            self.active = True
            LOGGER.info('Activating hot material recovery after two rejected Newton directions')
        # Observe the rejection, then let the common driver refresh its
        # Jacobian and continue the same solve with its original budget.
        return False
