#!/usr/bin/env python3
"""Build the bundled first-principles He- free-free table (no stellar data).

usage: python tools/build_he_minus_partial_wave_table.py WORK_DIR [--assemble-only]

Physics in tools/he_minus_partial_wave.py: e-He continuum dipole matrix
elements in an l-dependent pseudopotential fitted to Nesbet (1979) phase
shifts with the p-wave weighted tightly (parameters below), thermally averaged
via Kirchhoff's law.  One temperature per checkpoint in WORK_DIR; rerunning
resumes.  The assembled table goes to
src/wd_spectra/data/runtime/cache/he-minus-ff/he_minus_ff_partial_wave.npz
with the comparison against John (1994) at lambda >= 0.5063 um recorded in its
metadata.  OpenWD uses only the wavelength *shape* below 0.5063 um, scaled to
John's edge value, so the table never replaces John where John is tabulated.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import he_minus_partial_wave as heff  # noqa: E402

PWAVE_PARAMS = np.array([-0.11462272644270027, 0.3743916962167844, 0.5812466750562241,
                         2.5120719742164304, 0.4776639108051892])
THETA = np.round(np.arange(0.5, 3.6001, 0.1), 3)        # John (1994): 1400-10080 K
WAVE = np.unique(np.r_[np.geomspace(900.0, 151878.0, 90), 5063.0])
OUT = ROOT / "src/wd_spectra/data/runtime/cache/he-minus-ff/he_minus_ff_partial_wave.npz"


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    heff.PARAMS = PWAVE_PARAMS
    grid = {"theta": THETA.tolist(), "wavelength_angstrom": WAVE.tolist(),
            "params": heff.PARAMS.tolist(), "which": "pwave"}
    gfile = work / "grid.json"
    if gfile.exists() and json.loads(gfile.read_text()) != grid:
        raise SystemExit("work dir grid/params differ from this builder")
    gfile.write_text(json.dumps(grid))
    if "--assemble-only" not in sys.argv:
        for th in THETA:
            f = work / f"theta_{th:.3f}.npy"
            if f.exists():
                continue
            t0 = time.time()
            k = heff.k_ff_grid(5040.0 / th, WAVE)
            np.save(work / f"theta_{th:.3f}.tmp.npy", k)
            (work / f"theta_{th:.3f}.tmp.npy").rename(f)
            print(f"theta {th:.3f} done {time.time() - t0:.1f}s", flush=True)
    table = np.array([np.load(work / f"theta_{th:.3f}.npy") for th in THETA])
    from wd_spectra.helium import helium_minus_free_free_coefficient as k
    sel = WAVE >= 5063.0
    ratio = np.array([table[i, sel] / k(WAVE[sel] / 1e4, 5040.0 / th, prescription="john1994")
                      for i, th in enumerate(THETA)])
    source = (ROOT / "tools/he_minus_partial_wave.py").read_bytes()
    meta = {
        "schema": "he-minus-ff-partial-wave-v1",
        "method": "e-He continuum dipole matrix elements (acceleration form; convergent "
                  "s-p form for the s-only Pauli term), Kirchhoff thermal average, "
                  "stimulated emission included; cm^4 dyn^-1 per n(He I) P_e",
        "potential": "V1 = B1 exp(-b1 r^2) - alpha/(2 (r^2+d^2)^2) (l>=1); V0 = V1 + C exp(-g r^2)",
        "potential_parameters_au": heff.PARAMS.tolist(),
        "phase_shift_reference": "Nesbet 1979, Phys. Rev. A 20, 58; scattering length 1.1835 a0",
        "alpha_He_a0^3": heff.ALPHA_HE,
        "numerics": "Numerov h=0.02 a0 to 200 a0, l<=24, 48-point Gauss-Laguerre in E/kT; "
                    "converged to 1e-4 in l and quadrature",
        "john1994_ratio_lambda_ge_0.5063um": {"min": float(ratio.min()), "max": float(ratio.max()),
                                              "median": float(np.median(ratio))},
        "builder_source_sha256": hashlib.sha256(source).hexdigest(),
        "use": "OpenWD uses only the shape below 0.5063 um, scaled to John (1994) at the edge",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, theta=THETA, wavelength_micron=WAVE / 1e4, coefficient_cm4_dyn=table,
                        metadata_json=np.asarray(json.dumps(meta)))
    print("wrote", OUT, json.dumps(meta["john1994_ratio_lambda_ge_0.5063um"]))


if __name__ == "__main__":
    main()
