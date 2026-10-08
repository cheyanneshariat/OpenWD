"""Research C1 REOS interpolation; not a thermodynamically fitted free energy.

Shape-preserving PCHIP avoids jumps of Cp and Q at the old linear T knots.
Rho and u remain table interpolants: full Maxwell consistency is NOT claimed.
Only the common pressure support of the selected isotherms is accepted.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
from wd_spectra.eos import HeliumThermodynamics
from .dense_helium_limits import DenseHeliumDomainError


class SmoothREOS3:
    def __init__(self, table, minimum_temperature, maximum_temperature):
        lower = max(0, np.searchsorted(table.temperature_grid, minimum_temperature)-2)
        upper = min(len(table.temperature_grid), np.searchsorted(table.temperature_grid, maximum_temperature)+3)
        self.temperature_grid = table.temperature_grid[lower:upper]
        self.pressure_by_temperature = table.pressure_by_temperature[lower:upper]
        self.p_min = max(p[0] for p in self.pressure_by_temperature)
        self.p_max = min(p[-1] for p in self.pressure_by_temperature)
        self.density_curves = [PchipInterpolator(np.log(p), np.log(r), extrapolate=False)
            for p,r in zip(self.pressure_by_temperature, table.density_by_temperature[lower:upper])]
        self.energy_curves = [PchipInterpolator(np.log(p), u*1e10, extrapolate=False)
            for p,u in zip(self.pressure_by_temperature, table.internal_energy_by_temperature[lower:upper])]
        self.source = table.source+'; research C1 PCHIP rho/u; analytic Cp,Q from same interpolants'

    def fields(self, pressure, temperature):
        return self.fixed_pressure(pressure)(temperature)[0]

    def fixed_pressure(self, pressure):
        """Precompute isobar polynomials and their exact log-T derivatives.

        This is evaluation of the SAME interpolated EOS, not an additional
        local surrogate. Particularly useful in stiff convective proposals.
        """
        p=np.asarray(pressure,float)
        if np.any(~np.isfinite(p)) or np.any(p < self.p_min) or np.any(p > self.p_max):
            raise DenseHeliumDomainError('smooth REOS3 common-support domain violation; no extrapolation')
        values_rho = np.stack([curve(np.log(p).ravel()) for curve in self.density_curves])
        values_u = np.stack([curve(np.log(p).ravel()) for curve in self.energy_curves])
        rho_curve = PchipInterpolator(np.log(self.temperature_grid), values_rho, axis=0, extrapolate=False)
        u_curve = PchipInterpolator(self.temperature_grid, values_u, axis=0, extrapolate=False)
        def evaluate(temperature):
            return self._isobar_fields(p,temperature,rho_curve,u_curve)
        return evaluate

    def _isobar_fields(self, pressure, temperature, rho_curve, u_curve):
        p,t = np.broadcast_arrays(np.asarray(pressure,float),np.asarray(temperature,float))
        if (np.any(~np.isfinite(p)) or np.any(~np.isfinite(t)) or np.any(p < self.p_min)
                or np.any(p > self.p_max) or np.any(t < self.temperature_grid[0])
                or np.any(t > self.temperature_grid[-1])):
            raise DenseHeliumDomainError('smooth REOS3 common-support domain violation; no extrapolation')
        # Each column has its own pressure and temperature. Evaluate the
        # corresponding polynomial without forming an N-by-N cross product.
        def diagonal(curve, x, derivative=0):
            x = x.ravel()
            i = np.clip(np.searchsorted(curve.x,x,side='right')-1,0,len(curve.x)-2)
            dx = x-curve.x[i]
            column=np.arange(len(x)) if curve.c.shape[-1] > 1 else np.zeros(len(x),int)
            c = curve.c[:,i,column]
            if derivative == 2:
                result=6*c[0]*dx+2*c[1]
            elif derivative == 1:
                result=(3*c[0]*dx+2*c[1])*dx+c[2]
            else:
                result=((c[0]*dx+c[1])*dx+c[2])*dx+c[3]
            return result.reshape(p.shape)
        rho = np.exp(diagonal(rho_curve,np.log(t)))
        u = diagonal(u_curve,t)
        expansion = -diagonal(rho_curve,np.log(t),True)
        cp = diagonal(u_curve,t,True)+p/rho*expansion/t
        q_slope=-diagonal(rho_curve,np.log(t),2)
        cp_slope=t*diagonal(u_curve,t,2)+p/(rho*t)*(q_slope+(expansion-1)*expansion)
        return (rho,u,cp,expansion),(-rho*expansion,t*diagonal(u_curve,t,1),cp_slope,q_slope)

    def evaluate(self, pressure, temperature):
        rho,u,_,_ = self.fields(pressure,temperature)
        return rho,u,np.ones_like(rho,dtype=bool)

    def thermodynamics(self, temperature, pressure):
        rho,_,cp,q = self.fields(pressure,temperature)
        ad = np.asarray(pressure)*q/(rho*np.asarray(temperature)*cp)
        if np.any(~np.isfinite(cp)) or np.any(cp <= 0) or np.any(q <= 0):
            raise ValueError('nonphysical smooth REOS thermal derivative')
        return HeliumThermodynamics(cp,q,ad)
