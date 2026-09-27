"""Research-only bounded Gauss-Newton trial model, in physical coordinates.

No atmosphere equations are changed. Unlike a tanh parameterization, the
physical temperature box has no vanishing coordinate derivative at its edge.
The linear subproblem keeps weak singular directions down to LAPACK's normal
rank threshold; it does not square the condition number through J.T @ J.
Every returned proposal still needs a fully reevaluated atmosphere trial.
"""
from types import SimpleNamespace
import numpy as np
from scipy.optimize import lsq_linear


def solve_bounded_model(model, size, radius, maximum_evaluations):
    if size < 2 or not np.isfinite(radius) or radius <= 0 or maximum_evaluations < 1:
        raise ValueError('invalid bounded nonlinear proposal dimensions/budget')
    delta=np.zeros(size)
    values,jacobian=model(delta)
    calls=1
    def merit(v):return float(np.linalg.norm(v))
    initial=merit(values)
    target=np.sqrt(np.finfo(float).eps)*max(1.,initial)
    status=0
    while calls < maximum_evaluations:
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(jacobian)):
            raise ValueError('nonfinite bounded trial model')
        current=merit(values)
        if current <= target:
            status=1
            break
        # Solve directly in nodal log temperatures. Uniform modes are not
        # excluded by a box on rotated coordinates or killed by tanh saturation.
        lower=-radius-delta
        upper=radius-delta
        linear=lsq_linear(jacobian,-values,bounds=(lower,upper),method='bvls',
                          tol=1e-12,lsq_solver='exact')
        if not linear.success:
            raise RuntimeError('bounded linear subproblem did not finish')
        direction=linear.x
        accepted=False
        factor=1.
        while calls < maximum_evaluations:
            trial=np.clip(delta+factor*direction,-radius,radius)
            if np.array_equal(trial,delta):break
            trial_values,trial_jacobian=model(trial)
            calls+=1
            if (np.all(np.isfinite(trial_values)) and np.all(np.isfinite(trial_jacobian))
                    and merit(trial_values)<current):
                delta,values,jacobian=trial,trial_values,trial_jacobian
                accepted=True
                break
            factor*=.5
        if not accepted:
            status=2  # constrained/stagnant proposal, NOT a root certificate
            break
    if merit(values)<=target:status=1
    return SimpleNamespace(x=delta,nfev=calls,status=status,
        residual_maximum=float(np.max(abs(values))),
        root_solved=bool(status==1),
        optimality=float(np.max(abs(jacobian.T@values))))
