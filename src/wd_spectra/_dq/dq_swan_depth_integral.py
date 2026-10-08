"""Analytic spatial quadrature of thermally broadened, pressure-shifted lines.

For a cell with constant local line strength/width and a shift varying
linearly with column mass, integrate every Gaussian through the cell.
This is NOT wavelength smoothing: it integrates extinction along a ray.
The thermal/unshifted limit and line areas are retained without rescaling.
Temperature/abundance variation still requires ordinary cell refinement.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import ndtr

from .dqsolution_sampling import SparseResolvedSwan, TRUNCATION_SIGMA


class SpatiallyIntegratedSwan(SparseResolvedSwan):
    def cumulative(self, query_wavenumber, temperature):
        q=np.asarray(query_wavenumber,dtype=float)
        if q.ndim!=1 or np.any(~np.isfinite(q)):
            raise ValueError('Finite 1D wavenumbers required')
        # Negative wavenumbers have exactly zero support in this bounded list.
        positive=q>0
        out=np.zeros_like(q)
        if not np.any(positive):
            return out
        mapping=self._query_map(
            1e8/q[positive], maximum_temperature=temperature
        )
        velocity=self._velocity_fraction(temperature)
        strength=self._line_strength(temperature)
        prefix=np.r_[0.,np.cumsum(strength)]
        index=np.searchsorted(self.nu,q[positive],side='left')
        offset=mapping.fractional_offset
        # Correct a cheap zero-width prefix integral only near each boundary.
        # Beyond eight sigma the omitted Gaussian area is <1.3e-15 per line.
        inside=np.abs(offset)<=TRUNCATION_SIGMA*velocity
        corrections=np.zeros_like(offset)
        corrections[inside]=ndtr(offset[inside]/velocity)-(offset[inside]>0)
        operator=csr_matrix((corrections,mapping.indices,mapping.indptr),
            shape=(len(mapping.wavelength),len(self.nu)),copy=False)
        out[positive]=prefix[index]+operator@strength
        return out

    def cell_average(self, wavelength, temperature, shift_left, shift_right):
        w=np.asarray(wavelength,dtype=float)
        if (w.ndim!=1 or np.any(~np.isfinite(w)) or np.any(w<=0)
            or not np.isfinite(temperature) or temperature<self.table.temperature_K[0]
            or temperature>self.maximum_temperature
            or not np.isfinite(shift_left) or not np.isfinite(shift_right)):
            raise ValueError('Invalid spatial line quadrature state')
        low,high=sorted((float(shift_left),float(shift_right)))
        span=high-low
        q=1e8/w
        if span==0:
            shifted=q-low
            result=np.zeros_like(w);use=shifted>0
            result[use]=self.swan_column(1e8/shifted[use],temperature)
            return result
        # Tiny spatial shifts use direct Gaussian quadrature to avoid
        # cancellation of two almost equal prefix integrals. GL8 resolves
        # this branch (span < 0.01 of the narrowest list Doppler sigma).
        if span<.01*self.nu[0]*self._velocity_fraction(temperature):
            x,weights=np.polynomial.legendre.leggauss(8)
            result=np.zeros_like(w)
            for xj,wj in zip(x,weights):
                shifted=q-(low+.5*(xj+1)*span);use=shifted>0
                result[use]+=.5*wj*self.swan_column(1e8/shifted[use],temperature)
            return result
        result=(self.cumulative(q-low,temperature)-self.cumulative(q-high,temperature))/span
        # The CDF difference is nonnegative; roundoff is bounded by prefix
        # summation, not a profile/area renormalization.
        tolerance=2e-13*np.sum(self._line_strength(temperature))/span
        if np.min(result)<-tolerance:
            raise FloatingPointError('Negative spatially integrated opacity')
        return np.maximum(result,0.)
