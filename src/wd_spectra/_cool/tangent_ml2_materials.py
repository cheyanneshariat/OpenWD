"""Local material tangent for a nonlinear ML2 proposal, never an accepted EOS."""
import numpy as np


class TangentML2Materials:
    def __init__(self,payload):
        t=payload['convection_transport']
        self.base=tuple(t[k] for k in ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
        self.response=payload['ml2_coefficient_log_temperature_responses']

    def __call__(self,delta):
        a,b,c=self.base;da,db,dc=self.response
        nb=b*np.exp((db/b[:,None])@delta)
        nc=c*np.exp((dc/c[:,None])@delta)
        return (a+da@delta,nb,nc),(da,nb[:,None]*(db/b[:,None]),nc[:,None]*(dc/c[:,None]))
