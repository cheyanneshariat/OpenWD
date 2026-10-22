"""Build the dense-He He- free-free correction table (no star fitting).

delta_ff(T, rho, lambda) = int I0(k) S(k) dk / int I0(k) dk, Iglesias,
Rogers & Saumon (2002) Eqs. 3.2-3.4, with S(k) from the thermodynamically
self-consistent Rogers-Young closure for the Ross-Young (1986) exp-6 fluid.

Dielectric dressing (``--dielectric``):
  propagation  (default) no factor in delta. The ray kernel already divides
               the extinction by the refractive index N; this is the 1/N of
               Iglesias Eq. 4.6 and of free-carrier absorption by neutral
               scatterers in a transparent dielectric.
  blouin       delta additionally divided by |epsilon(omega)|^2 = N^4 as in
               Blouin, Dufour & Allard (2018) Eq. 4 (net 1/N^5 with the ray
               kernel). Retained for reproduction only.
Both the bare correlation ratio and epsilon are stored, so either can be
reconstructed from one build. Isotherms are computed sequentially and each
is checkpointed; rerunning with the same --output resumes.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import CubicSpline

from wd_spectra.constants import BOHR_RADIUS, HELIUM_MASS, LIGHT_SPEED, PLANCK
from wd_spectra.dense_helium_continuum import HARTREE_ERG, helium_minus_correction
from wd_spectra._dq.dq_dense_continuum_inputs import SOURCES, dielectric
from wd_spectra._dq.dq_rogers_young_fluid import RadialGrid, solve_consistent_rogers_young
from wd_spectra._dq.provenance import digest

ROOT = Path(__file__).resolve().parents[1]
TEMPERATURES = np.geomspace(2000, 40000, 33)[4:]          # 2908--40000 K, as v1
DENSITY_SCALE = 1e-3
_X = np.linspace(0, np.log1p(1.2/DENSITY_SCALE), 49)
WAVELENGTHS = np.geomspace(1000, 100000, 81)
SUBSTEPS = 4   # density continuation steps per knot; knots are unchanged


def density_knots(extra):
    return np.r_[_X, _X[-1]+(_X[1]-_X[0])*np.arange(1, extra+1)]


def isotherm(t, x, grid):
    rho = DENSITY_SCALE*np.expm1(x)
    energy = PLANCK*LIGHT_SPEED/(WAVELENGTHS*1e-8*HARTREE_ERG)
    ratio = np.ones((len(x), len(WAVELENGTHS)))
    eps = np.ones_like(ratio)
    alpha = np.full(len(x), np.nan)
    s0 = np.ones(len(x))
    s0_virial = np.ones(len(x))
    smax = np.ones(len(x))
    closure = quadrature = 0.
    gamma, guess = None, 0.1
    for j in range(1, len(x)):
        for xi in np.linspace(x[j-1], x[j], SUBSTEPS+1)[1:]:
            n = DENSITY_SCALE*np.expm1(xi)/HELIUM_MASS*1e-24   # atoms/A^3
            fluid = solve_consistent_rogers_young(grid, t, n, gamma, guess)
            gamma, guess = fluid.state.gamma, fluid.alpha
        state = fluid.state
        structure = CubicSpline(np.r_[0, grid.k*BOHR_RADIUS*1e8], np.r_[fluid.structure_zero, state.structure],
                                bc_type=((1, 0.), (2, 0.)), extrapolate=False)
        result = helium_minus_correction(WAVELENGTHS, t, structure_factor=structure, dielectric=1.)
        ratio[j] = result.factor
        eps[j] = dielectric(t, rho[j]/HELIUM_MASS, energy, short_range="constant")
        alpha[j], s0[j], s0_virial[j] = fluid.alpha, fluid.structure_zero, fluid.virial_structure_zero
        smax[j] = state.structure.max()
        closure = max(closure, state.residual)
        quadrature = max(quadrature, result.quadrature_relative_change)
    return dict(ratio=ratio, dielectric=eps, alpha=alpha, s0=s0, s0_virial=s0_virial,
                smax=smax, closure=closure, quadrature=quadrature)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-density-knots", type=int, default=6,
                        help="knots beyond 1.2 g/cm3 at the v1 spacing (6 -> 2.913 g/cm3)")
    parser.add_argument("--dielectric", choices=("propagation", "blouin"), default="propagation")
    parser.add_argument("--radial-points", type=int, default=2047)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    x = density_knots(args.extra_density_knots)
    grid = RadialGrid(args.radial_points, 24.)
    started = time.monotonic()
    parts = []
    for i, t in enumerate(TEMPERATURES):
        path = args.output/f"isotherm-{i:02d}.npz"
        if not path.exists():
            values = isotherm(t, x, grid)
            np.savez(path.with_suffix(".tmp.npz"), **values)
            path.with_suffix(".tmp.npz").rename(path)
            print(json.dumps(dict(isotherm=i, temperature=t, elapsed_seconds=time.monotonic()-started,
                                  closure=values["closure"], quadrature=values["quadrature"],
                                  s0_top=values["s0"][-1], smax_top=values["smax"][-1])), flush=True)
        with np.load(path) as z:
            parts.append({k: z[k] for k in z.files})
    ratio = np.stack([p["ratio"] for p in parts])
    eps = np.stack([p["dielectric"] for p in parts])
    factor = ratio if args.dielectric == "propagation" else ratio/eps**2
    implementation = {p: digest(ROOT/p) for p in (
        "tools/build_dq_dense_continuum.py", "src/wd_spectra/dense_helium_continuum.py",
        "src/wd_spectra/_dq/dq_rogers_young_fluid.py", "src/wd_spectra/_dq/dq_dense_continuum_inputs.py",
        "src/wd_spectra/_cool/dense_helium_fluid_experiment.py")}
    metadata = dict(
        schema="dense-He-ff-v1", sources=dict(SOURCES, closure="https://doi.org/10.1103/PhysRevA.30.999"),
        parameters_fitted=False, qualified=False,
        approximation="Iglesias et al. 2002 Born ratio, Eqs. 3.2-3.4",
        fluid="Ross & Young 1986 exp-6, OZ with Rogers-Young closure; alpha from "
              "virial/compressibility self-consistency at every state (no data fitted)",
        dielectric_dressing=args.dielectric,
        dielectric_note=("factor excludes 1/|epsilon|^2; transport applies 1/N" if args.dielectric == "propagation"
                         else "factor = ratio/|epsilon|^2 (Blouin 2018 Eq. 4)"),
        refractivity="Masili Tables V, VI + Hattig/Maroulis pair polarizability; constant short range",
        wavelength_scope="1000--100000 A",
        maximum_density_g_cm3=float(DENSITY_SCALE*np.expm1(x[-1])),
        density_continuation_substeps=SUBSTEPS, radial_points=args.radial_points, radial_extent_angstrom=24.,
        implementation_sha256=implementation,
        isotherms=[dict(temperature=float(t), closure=float(p["closure"]), quadrature=float(p["quadrature"]),
                        s0_top=float(p["s0"][-1]), smax_top=float(p["smax"][-1]))
                   for t, p in zip(TEMPERATURES, parts)])
    np.savez_compressed(args.output/"correction.npz", log_temperature=np.log(TEMPERATURES),
        log1p_density=x, log_wavelength=np.log(WAVELENGTHS), density_scale_g_cm3=DENSITY_SCALE,
        factor=factor, correlation_ratio=ratio, dielectric=eps,
        alpha=np.stack([p["alpha"] for p in parts]), structure_zero=np.stack([p["s0"] for p in parts]),
        structure_zero_virial=np.stack([p["s0_virial"] for p in parts]),
        metadata_json=json.dumps(metadata))
    (args.output/"report.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"completed {args.output/'correction.npz'} in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
