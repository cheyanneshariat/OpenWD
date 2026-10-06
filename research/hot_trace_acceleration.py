"""Research acceleration with separate element histories and one shared SE map."""
import numpy as np
from wd_spectra._pg1159_acceleration import population_update


class ElementPopulationAcceleration:
    """Prevent one atom's many levels from dominating every secant fit.

    The caller still computes one coupled radiation field and the same raw
    statistical-equilibrium proposals and convergence residual. Failed
    extrapolations use the solver's damped fallback (arithmetic for rising,
    geometric for falling populations).
    """
    def __init__(self,rows):
        self.rows=dict(rows)
        self.histories={element:[] for element in rows}
        self.previous_norm={}
        self.calls=0
        self.reasons={element:{} for element in rows}
        self.last_reasons={}

    def __call__(self,old,target,history,*,depth,mixing,maximum_step,residual_weights=None):
        if old.shape!=target.shape or old.shape[0]!=sum(self.rows.values()):
            raise ValueError('acceleration element rows do not match the population arrays')
        if not history:
            for values in self.histories.values():values.clear()
            self.previous_norm.clear()
        # The caller may clear this marker after a globally growing defect.
        history[:]=[None]
        self.calls+=1;offset=0;parts=[];accepted=False
        for element,size in self.rows.items():
            sl=slice(offset,offset+size);offset+=size
            weights=None if residual_weights is None else residual_weights[sl]
            norm=float(np.max(abs(target[sl]-old[sl])*(1. if weights is None else weights)))
            if element in self.previous_norm and norm>2*self.previous_norm[element]:
                self.histories[element].clear()
            self.previous_norm[element]=norm
            result,reason=population_update(old[sl],target[sl],self.histories[element],
                depth=depth,mixing=mixing,maximum_step=maximum_step,residual_weights=weights)
            self.reasons[element][reason]=self.reasons[element].get(reason,0)+1
            self.last_reasons[element]=reason
            if result is None:
                # Same damped fallback as solve_hot_trace_metals: arithmetic
                # mean for rising populations, geometric (halfway in log) for
                # falling ones; the caller restores each element's particles.
                result=(target[sl] if mixing==1. else np.where(target[sl]>=old[sl],
                    np.logaddexp(np.log1p(-mixing)+old[sl],np.log(mixing)+target[sl]),
                    (1.-mixing)*old[sl]+mixing*target[sl]))
            else:accepted=True
            parts.append(result)
        return (np.concatenate(parts),'accepted') if accepted else (None,'elementwise_fallback')

    def diagnostics(self):
        return dict(calls=self.calls,last_reasons=self.last_reasons,reason_counts=self.reasons)
