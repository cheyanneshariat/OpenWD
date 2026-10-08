"""DQ research proposals on finite-material ML2 coordinates.

Only trial temperatures and their linearization use these coordinates.
Every accepted residual is the original, rebuilt physical radiation/ML2
residual. No coordinate flux replaces a calculated flux. The shared
solver and physical convergence gates are retained. The DQ physical phase
uses consistent least-squares globalization and unmodified analytic refreshes;
public non-DQ dispatch and numerical-policy defaults are unchanged.
"""
from collections import OrderedDict
import numpy as np


class ExactMaterialCache:
    """Bounded reuse of identical material requests, never interpolation."""
    def __init__(self,materials,maximum_entries=32,maximum_bytes=16*1024**2):
        if maximum_entries<1 or maximum_bytes<1:raise ValueError('Positive cache bounds required')
        self.materials=materials;self.maximum_entries=maximum_entries;self.maximum_bytes=maximum_bytes
        self.cache=OrderedDict();self.nbytes=0

    @staticmethod
    def frozen(value):
        if isinstance(value,(tuple,list)):
            items=[ExactMaterialCache.frozen(v) for v in value]
            return tuple(item[0] for item in items),sum(item[1] for item in items)
        result=np.array(value,copy=True);result.flags.writeable=False
        return result,result.nbytes

    def __call__(self,logt,derivative=False):
        x=np.asarray(logt)
        key=(x.dtype.str,x.shape,x.tobytes(),bool(derivative))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        result,size=self.frozen(self.materials(logt,derivative))
        if size<=self.maximum_bytes:
            while self.cache and (len(self.cache)>=self.maximum_entries or self.nbytes+size>self.maximum_bytes):
                _,(_,removed)=self.cache.popitem(last=False);self.nbytes-=removed
            self.cache[key]=(result,size);self.nbytes+=size
        return result






