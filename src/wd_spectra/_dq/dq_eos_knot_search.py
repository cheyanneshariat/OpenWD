"""Search adjoining EOS pieces when a cheap-model minimum sits on a corner.

These are starting points for the SAME bounded nonlinear trial problem,
not new equations or accepted atmospheres. Endpoints come only from the EOS
table and its thermodynamic window; there are no object/Teff choices. The
outer physical merit and convergence gates remain mandatory.
"""
import numpy as np
from scipy.optimize import least_squares


def corner_starts(system, state, delta, radius):
    owner=getattr(system,'eos_trial_owner',None)
    if owner is None:
        return []
    n=system.n;x=state[:n]+delta[:n]
    grid=np.log(owner.helium_reos3.temperature_grid)
    breaks=np.unique(np.r_[grid,grid-2e-4,grid+2e-4])
    starts=[]
    for i,t in enumerate(x):
        j=int(np.argmin(abs(breaks-t)))
        # Match the inner solver's relative xtol, not a stellar parameter.
        if abs(breaks[j]-t)>1e-10*max(1.,abs(t)):
            continue
        # Include every table/window piece reachable in this trust box.
        # Merely nudging past the first edge can return to the same corner
        # without traversing the narrow thermodynamic transition at all.
        lo=max(state[i]-radius,grid[0]+2e-4)
        hi=min(state[i]+radius,grid[-1]-2e-4)
        edges=np.r_[lo,breaks[(breaks>lo)&(breaks<hi)],hi]
        for midpoint in .5*(edges[:-1]+edges[1:]):
            dt=midpoint-t
            move=np.zeros_like(state);move[i]=dt
            move[n:]=system.gradient_operator[1:,i]*dt/system.gradient_scale
            # Keep compatibility changes together while entering the same
            # native-coordinate trust box. Never increase its radius.
            ratios=np.divide(radius-np.sign(move)*delta,abs(move),
                out=np.full_like(move,np.inf),where=move!=0)
            factor=min(1.,float(np.min(ratios)))
            if factor<=0:
                continue
            starts.append(np.clip(delta+factor*move,-radius,radius))
    return starts


def search_corners(system,state,solved,radius,evaluated):
    starts=corner_starts(system,state,solved.x,radius)
    best=solved
    trials=[]
    for start in starts:
        candidate=least_squares(lambda d:evaluated(d).residual,start,
            jac=lambda d:evaluated(d).jacobian,bounds=(-radius,radius),
            method='trf',x_scale='jac',max_nfev=1000,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        trials.append(dict(cost=float(candidate.cost),evaluations=candidate.nfev,
            success=bool(candidate.success)))
        if np.isfinite(candidate.cost) and candidate.cost<best.cost:
            best=candidate
    return best,dict(eos_corner_trials=trials,eos_corner_selected=best is not solved)
