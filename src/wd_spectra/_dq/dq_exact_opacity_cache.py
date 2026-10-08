"""Bounded, exact-local-state opacity reuse for the isolated DQ experiment.

Every constitutive state field enters the key. No rounded temperatures,
densities, wavelength shifts, or interpolated cached opacities are used.
Atmosphere arrays have depth first; MetalLTEState arrays have depth last.
"""
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace

import numpy as np


def subset(value, indices, depths, axis=0):
    if isinstance(value, np.ndarray):
        if value.shape[axis] != depths:
            raise ValueError('Unexpected layer axis in constitutive state')
        return np.take(value, indices, axis=axis)
    if is_dataclass(value):
        return replace(value, **{f.name: ({} if f.name == 'metadata' else
            subset(getattr(value, f.name), indices, depths, axis)) for f in fields(value)})
    if isinstance(value, Mapping):
        return {k: subset(v, indices, depths, axis) for k, v in value.items()}
    return value


def identity(value):
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if is_dataclass(value):
        return tuple((f.name, identity(getattr(value, f.name)))
            for f in fields(value) if f.name != 'metadata')
    if isinstance(value, Mapping):
        return tuple((k, identity(v)) for k, v in value.items())
    return value


class ExactLayerOpacityCache:
    def __init__(self, max_grids=2, max_columns=192):
        if max_grids < 1 or max_columns < 1:
            raise ValueError('Cache limits must be positive')
        self.max_grids, self.max_columns = max_grids, max_columns
        self.grids = OrderedDict()
        self.evaluated_columns = self.reused_columns = 0

    def evaluate(self, a, carbon, c2, wave, include_c2, compute):
        w = np.asarray(wave, dtype=float)
        key = (identity(w), bool(include_c2))
        if key not in self.grids:
            self.grids[key] = OrderedDict()
        self.grids.move_to_end(key)
        while len(self.grids) > self.max_grids:
            self.grids.popitem(last=False)
        cache = self.grids[key]
        keys = []
        for j in range(a.n_depth):
            keys.append((identity(subset(a, [j], a.n_depth)),
                identity(subset(carbon, [j], a.n_depth, -1)), float(c2[j])))
        # Gather existing columns before inserting misses, so bounded eviction
        # cannot remove a column needed in this very call.
        out = np.empty((len(w), a.n_depth))
        missing = []
        for j, state in enumerate(keys):
            if state in cache:
                out[:, j] = cache[state]
                cache.move_to_end(state)
                self.reused_columns += 1
            else:
                missing.append(j)
        if missing:
            aa = subset(a, missing, a.n_depth)
            cc = subset(carbon, missing, a.n_depth, -1)
            values = compute(aa, cc, c2[missing], w, include_c2=include_c2)
            self.evaluated_columns += len(missing)
            for k, j in enumerate(missing):
                out[:, j] = values[:, k]
                column = values[:, k].copy()
                column.setflags(write=False)
                cache[keys[j]] = column
                while len(cache) > self.max_columns:
                    cache.popitem(last=False)
        return out
