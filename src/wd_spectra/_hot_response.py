"""Reuse frozen transfer responses across hot-atom material derivatives.

At fixed material and radiation coefficients, each depth-local derivative
column is linear in its source and extinction derivative. The bottom source
replaces the last direct-source derivative. Two basis solves therefore span
all material columns. Float64 bases retain the original transfer equations;
combining them changes floating-point operation order in the Jacobian only.
"""
import numpy as np
from ._mass_feautrier import MassResponseOperator, MassFactors, mass_width, validate_chunk


class HotResponseOperator(MassResponseOperator):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("reconstruct_intensity", True)
        super().__init__(*args, **kwargs)
        validate_chunk(self.chunk_size)
        if self.source.ndim != 2 or np.any(~np.isfinite(self.source)):
            raise ValueError('invalid source')
        nw, nd = self.source.shape
        if (self.wave.shape != (nw,) or nw < 2 or np.any(~np.isfinite(self.wave))
                or np.any(self.wave <= 0) or np.any(np.diff(self.wave) <= 0)):
            raise ValueError('invalid wavelengths')
        mass_width(self.mass)
        if self.mass.shape != (nd,):
            raise ValueError('mass grid mismatch')
        if any(v.shape != self.source.shape or np.any(~np.isfinite(v))
               for v in (self.tau, self.fraction, self.extinction)):
            raise ValueError('response fields must be finite and have identical shapes')
        if (np.any(self.extinction <= 0) or np.any(self.fraction < 0)
                or (not self.allow_stimulated_gain and np.any(self.fraction > 1))):
            raise ValueError('invalid material coefficients')
        self.bases = {}

    def _basis(self, start, stop):
        local = slice(start, stop)
        nc, nd = stop-start, len(self.mass)
        scalar = MassFactors(self.tau[local], np.zeros((nc, nd)), self.n_angle,
                             self.mass, self.extinction[local])
        rhs = np.zeros((nc, nd+1, self.n_angle))
        rhs[:, 1:] = self.source[local, :, None]
        _, jumps = scalar.solve(rhs, reconstruct_intensity=self.reconstruct_intensity)
        coupled = MassFactors(self.tau[local], self.fraction[local], self.n_angle,
                              self.mass, self.extinction[local])
        h = coupled.h
        # Relative extinction derivatives give well-scaled basis coefficients.
        dk = self.extinction[local]
        dh = np.zeros((nc, nd, nd))
        dh[:, 0, 0] = self.mass[0]*dk[:, 0]
        for i in range(1, nd):
            width = .5*(self.mass[i]-self.mass[i-1])
            dh[:, i, i-1] = width*dk[:, i-1]
            dh[:, i, i] = width*dk[:, i]
        opacity_rhs = np.zeros((nc, nd+1, self.n_angle, nd))
        opacity_rhs[:, 0] = (-coupled.mu[None, :, None]*dh[:, 0, None, :]
                             /h[:, 0, None, None]**2*jumps[:, 0, :, None])
        dv = np.zeros((nc, nd-1, nd))
        cells = np.arange(nd-1)
        dv[:, cells, cells] = 1.
        da = -coupled.a[:, 1:-1, :, None]*(dh[:, :-1, None, :]/h[:, :-1, None, None]+dv[:, :, None, :])
        dc = -coupled.c[:, 1:-1, :, None]*(dh[:, 1:, None, :]/h[:, 1:, None, None]+dv[:, :, None, :])
        opacity_rhs[:, 1:-1] = -da*jumps[:, :-1, :, None]+dc*jumps[:, 1:, :, None]
        source_rhs = np.zeros_like(opacity_rhs)
        for i in range(nd):
            source_rhs[:, i+1, :, i] = 1.
        source_response, source_jump = coupled.solve(source_rhs, reconstruct_intensity=self.reconstruct_intensity)
        opacity_response, opacity_jump = coupled.solve(opacity_rhs, reconstruct_intensity=self.reconstruct_intensity)
        fw = 4*np.pi*coupled.weight*coupled.mu**2
        basis = (
            np.einsum('wdrk,r->wdk', source_response[:, 1:], coupled.weight),
            np.einsum('wdrk,r->wdk', opacity_response[:, 1:], coupled.weight),
            np.einsum('wdrk,r->wdk', source_jump/h[:, :, None, None], fw),
            np.einsum('wdrk,r->wdk', opacity_jump/h[:, :, None, None]
                      -jumps[:, :, :, None]*dh[:, :, None, :]/h[:, :, None, None]**2, fw),
        )
        for array in basis:
            array.flags.writeable = False
        return basis

    def apply(self, direct, db, dk, *, return_auxiliary_response=True, mean_response_consumer=None):
        if any(np.shape(v) != self.source.shape or np.any(~np.isfinite(v)) for v in (direct, db, dk)):
            raise ValueError('response fields must be finite and have identical shapes')
        nw, nd = self.source.shape
        spacing = np.diff(self.wave, prepend=self.wave[0], append=self.wave[-1])
        weights = .5*(spacing[:-1]+spacing[1:])
        integrated = np.zeros((nd, nd))
        mean_response = np.empty((nw, nd, nd)) if return_auxiliary_response else None
        source_response = np.empty_like(mean_response) if return_auxiliary_response else None
        for start in range(0, nw, self.chunk_size):
            stop = min(nw, start+self.chunk_size)
            local = slice(start, stop)
            if start not in self.bases:
                self.bases[start] = self._basis(start, stop)
            mean_source, mean_opacity, flux_source, flux_opacity = self.bases[start]
            ds = np.array(direct[local], copy=True)
            ds[:, -1] = db[local, -1]
            relative_dk = dk[local]/self.extinction[local]
            mean = mean_source*ds[:, None, :]+mean_opacity*relative_dk[:, None, :]
            if return_auxiliary_response:
                mean_response[local] = mean
                sr = self.fraction[local, :, None]*mean
                sr[:, np.arange(nd), np.arange(nd)] += direct[local]
                source_response[local] = sr
            if mean_response_consumer is not None:
                mean.flags.writeable = False
                mean_response_consumer(start, stop, mean)
            flux = flux_source*ds[:, None, :]+flux_opacity*relative_dk[:, None, :]
            integrated += np.einsum('wdk,w->dk', flux, weights[local])
        return integrated, mean_response, source_response
