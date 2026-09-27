"""Exact fixed-pressure REOS material response inside an ML2 proposal.

Only radiation and Rosseland-opacity changes remain tangent approximations.
The accepted atmosphere still evaluates ALL actual materials and transfer.
No tolerance relaxation at the EOS's piecewise-polynomial knots is needed.
"""
from dataclasses import replace
import numpy as np
from wd_spectra._material_response import temperature_response_probes


class DenseExactMaterials:
    def __init__(self,exact,logt,radius,*,bulk):
        self.exact,self.logt,self.radius=exact,np.asarray(logt),radius
        self.atmosphere,base=exact.fields(logt)
        self.isobar=bulk.fixed_pressure(self.atmosphere.gas_pressure)
        self.opacity=base[1]
        hot,cold,h=temperature_response_probes(
            lambda offset: exact.fields(logt+offset)[1][1],2e-4,centered=True)
        self.opacity_slope=(np.log(hot)-np.log(self.opacity if cold is None else cold))/(
            h if cold is None else 2*h)
        # Explicitly verify the new evaluator's base matches the real one.
        fields,_=self.fields(np.zeros_like(logt))
        np.testing.assert_allclose(fields,base,rtol=2e-13,atol=0)

    def fields(self,delta):
        t=np.exp(self.logt+delta)
        (rho,u,cp,q),(rho_j,u_j,cp_j,q_j)=self.isobar(t)
        ad=self.atmosphere.gas_pressure*q/(rho*t*cp)
        ad_j=ad*(q_j/q-rho_j/rho-1-cp_j/cp)
        opacity=self.opacity*np.exp(self.opacity_slope*delta)
        return np.asarray((rho,opacity,cp,q,ad)),np.asarray((rho_j,opacity*self.opacity_slope,cp_j,q_j,ad_j))

    def __call__(self,delta):
        if np.max(abs(delta)) > self.radius*(1+1e-12):
            raise ValueError('material proposal outside trust region')
        return self.evaluate(delta)

    def evaluate(self,delta):
        """Exact EOS evaluation, also during inner compatibility iterations.

        Unlike a trust-region interpolant this has no surrogate-domain bound;
        the physical REOS bounds still apply. The returned proposal is bounded
        independently and every accepted state is evaluated by the full solver.
        """
        fields,slopes=self.fields(delta)
        atmosphere=replace(self.atmosphere,temperature=np.exp(self.logt+delta),mass_density=fields[0])
        return self.exact.assemble(atmosphere,fields,slopes)
