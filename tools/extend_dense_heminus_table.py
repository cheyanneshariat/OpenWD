"""Extend the dense-He He- free-free table below its coolest isotherm.

Cool pure-He (dense-db) upper layers reach ~1000 K, far below the table's
2412 K edge, where the held factor's zero T derivative stalls the structure
solve. Cooler isotherms are computed on the same uniform log-T spacing with
the builder's density knots, wavelengths and Rogers-Young fluid.

At low T the self-consistent fluid solve stops converging at high density
(cold, compressed He is near or beyond its melting line, where no fluid
closure applies). Each cool isotherm is kept up to its last converged knot;
its density limit is stored as ``maximum_valid_log1p_density`` and
DenseHeliumCorrectionTable refuses requests beyond it. Unconverged knots are
filled by quadratic continuation along density only to construct spline
coefficients. Sequential; each isotherm is checkpointed in OUTDIR.

usage: python tools/extend_dense_heminus_table.py EXISTING.npz OUTDIR MINIMUM_K
"""
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

from wd_spectra.constants import BOHR_RADIUS, HELIUM_MASS, LIGHT_SPEED, PLANCK
from wd_spectra.dense_helium_continuum import HARTREE_ERG, helium_minus_correction
from wd_spectra._dq.dq_dense_continuum_inputs import dielectric
from wd_spectra._dq.dq_rogers_young_fluid import RadialGrid, solve_consistent_rogers_young

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("builder", ROOT / "tools/build_dq_dense_continuum.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def partial_isotherm(t, x, grid):
    """B.isotherm, kept up to the last density knot whose fluid solve converges."""
    energy = PLANCK*LIGHT_SPEED/(B.WAVELENGTHS*1e-8*HARTREE_ERG)
    ratio = np.ones((len(x), len(B.WAVELENGTHS)))
    eps = np.ones_like(ratio)
    alpha = np.full(len(x), np.nan)
    s0, s0_virial, smax = np.ones(len(x)), np.ones(len(x)), np.ones(len(x))
    closure = quadrature = 0.
    gamma, guess, last = None, 0.1, 0
    for j in range(1, len(x)):
        try:
            for xi in np.linspace(x[j-1], x[j], B.SUBSTEPS+1)[1:]:
                n = B.DENSITY_SCALE*np.expm1(xi)/HELIUM_MASS*1e-24   # atoms/A^3
                fluid = solve_consistent_rogers_young(grid, t, n, gamma, guess)
                gamma, guess = fluid.state.gamma, fluid.alpha
        except RuntimeError:
            break
        state = fluid.state
        structure = CubicSpline(np.r_[0, grid.k*BOHR_RADIUS*1e8], np.r_[fluid.structure_zero, state.structure],
                                bc_type=((1, 0.), (2, 0.)), extrapolate=False)
        result = helium_minus_correction(B.WAVELENGTHS, t, structure_factor=structure, dielectric=1.)
        ratio[j] = result.factor
        eps[j] = dielectric(t, B.DENSITY_SCALE*np.expm1(x[j])/HELIUM_MASS, energy, short_range="constant")
        alpha[j], s0[j], s0_virial[j] = fluid.alpha, fluid.structure_zero, fluid.virial_structure_zero
        smax[j] = state.structure.max()
        closure = max(closure, state.residual)
        quadrature = max(quadrature, result.quadrature_relative_change)
        last = j
    return dict(ratio=ratio, dielectric=eps, alpha=alpha, s0=s0, s0_virial=s0_virial, smax=smax,
                closure=closure, quadrature=quadrature, last_knot=last)


def continue_along_density(values, last):
    """Quadratic continuation of log values beyond knot ``last`` (spline support only)."""
    v = np.array(values, dtype=float)
    if last < 2:
        raise ValueError("an isotherm needs at least three converged density knots")
    for j in range(last+1, v.shape[0]):
        v[j] = 3*v[j-1]-3*v[j-2]+v[j-3]
    return v


def main():
    existing, outdir, minimum = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    with np.load(existing, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    meta = json.loads(str(old["metadata_json"]))
    if meta.get("dielectric_dressing", "propagation") != "propagation":
        raise SystemExit("existing table is not the propagation dressing")
    x, logt = old["log1p_density"], old["log_temperature"]
    step = logt[1]-logt[0]
    grid = RadialGrid(2047, 24.)
    started = time.monotonic()
    parts, temperatures = [], []
    k = 1
    while np.exp(logt[0]-k*step) >= minimum:
        t = float(np.exp(logt[0]-k*step))
        path = outdir/f"isotherm-cool-{k:02d}.npz"
        if not path.exists():
            values = partial_isotherm(t, x, grid)
            np.savez(path.with_suffix(".tmp.npz"), temperature=t, **values)
            path.with_suffix(".tmp.npz").rename(path)
            print(json.dumps(dict(temperature=t, elapsed_seconds=time.monotonic()-started,
                                  closure=float(values["closure"]), quadrature=float(values["quadrature"]),
                                  maximum_valid_density_g_cm3=float(B.DENSITY_SCALE*np.expm1(x[values["last_knot"]])))),
                  flush=True)
        with np.load(path) as z:
            parts.insert(0, {key: z[key] for key in z.files})
        temperatures.insert(0, t)
        k += 1
    if not parts:
        raise SystemExit("no isotherm below the existing table edge was requested")
    last = np.array([int(p["last_knot"]) for p in parts])
    ratio = np.stack([np.exp(continue_along_density(np.log(p["ratio"]), j)) for p, j in zip(parts, last)])
    eps = np.stack([np.exp(continue_along_density(np.log(p["dielectric"]), j)) for p, j in zip(parts, last)])
    limit = np.r_[x[last], np.full(len(logt), x[-1])]
    previous = old.get("maximum_valid_log1p_density")
    if previous is not None:
        limit[len(parts):] = previous
    meta["isotherms"] = ([dict(temperature=t, closure=float(p["closure"]), quadrature=float(p["quadrature"]),
                               s0_top=float(p["s0"][j]), smax_top=float(p["smax"][j]),
                               maximum_valid_density_g_cm3=float(B.DENSITY_SCALE*np.expm1(x[j])))
                          for t, p, j in zip(temperatures, parts, last)] + meta.get("isotherms", []))
    meta["cool_extension"] = (f"{temperatures[0]:.0f}-{temperatures[-1]:.0f} K added by "
                              "tools/extend_dense_heminus_table.py; requests above each isotherm's "
                              "converged fluid density are refused")
    lt = np.r_[np.log(temperatures), logt]
    assert np.allclose(np.diff(lt), step, rtol=1e-10)
    def stacked(part_key, table_key):
        return np.concatenate([np.stack([np.where(np.arange(len(x)) <= j, p[part_key], np.nan)
                                         for p, j in zip(parts, last)]), old[table_key]])
    np.savez_compressed(outdir/"correction.npz", log_temperature=lt, log1p_density=x,
                        log_wavelength=old["log_wavelength"], density_scale_g_cm3=old["density_scale_g_cm3"],
                        factor=np.concatenate([ratio, old["correlation_ratio"]]),
                        correlation_ratio=np.concatenate([ratio, old["correlation_ratio"]]),
                        dielectric=np.concatenate([eps, old["dielectric"]]),
                        alpha=stacked("alpha", "alpha"), structure_zero=stacked("s0", "structure_zero"),
                        structure_zero_virial=stacked("s0_virial", "structure_zero_virial"),
                        maximum_valid_log1p_density=limit, metadata_json=json.dumps(meta))
    print("wrote", outdir/"correction.npz", flush=True)


if __name__ == "__main__":
    main()
