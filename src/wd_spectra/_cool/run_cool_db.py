"""Reproduce the explicitly qualified dense-He cold-start recipe and audit it.

This is NOT compute_db's default EOS. Local material-domain guards remain
active at every Teff. No saved atmosphere is loaded and no EOS is substituted.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def commands(temperature, output, interaction_table, *, logg=8.):
    output = Path(output).resolve()
    common = [sys.executable, "-m", "wd_spectra._cool.extended_thermal_wavelength_experiment",
        "--logg", str(logg),
        "--thermal-maximum-wavelength", "1e8", "--structure-angles", "8",
        "--direct-spectrum", "--mass-conservative-transfer", "--smooth-heminus-join",
        "--pseudo-time-sweeps", "80", "--pseudo-time-implicit-convection",
        "--pseudo-time-until-local-balance", "--pseudo-time-output",
        str(output / f"pseudo-time-{temperature}.jsonl"),
        "--interaction-table", str(Path(interaction_table).resolve()),
        "--least-squares-merit", "--exact-dense-materials", "--exact-convection-tangent",
        "--discrete-transport-seed", "--augmented-convection-proposal",
        "--positive-radiative-rates", "--velocity-scaled-compatibility",
        str(temperature), "--physical-only", "--stable-transfer",
        "--step-method", "nonlinear-convection-current-energy",
        "--max-iterations", "40", "--no-continuations", "--mesh", "optical",
        "--n-depth", "80", "--nonlinear-materials", "--inverse-ml2-step",
        "--output-root", str(output)]
    audit = [sys.executable, "-m", "wd_spectra._cool.audit_dense_spectrum",
        "--expected-logg", str(logg),
        str(output / str(temperature)), "--interaction-table", str(Path(interaction_table).resolve()),
        "--output", str(output / "audit" / str(temperature)), "--wavelength-count", "8000",
        "--maximum-wavelength", "1e9", "--angles", "8", "16", "--skip-linear"]
    return common, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("temperature", type=int)
    parser.add_argument("--logg", type=float, default=8., help="requested log10 surface acceleration in cgs")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-unqualified", action="store_true",
        help="retain and return a completed exploratory result with failed qualification (never replace it)")
    parser.add_argument("--interaction-table", type=Path,
        default=HERE.parent / "data/runtime/cool-models/molecular-hnc-electron-domain-table.npz")
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error("temperature must be positive")
    if not math.isfinite(args.logg):
        parser.error('logg must be finite')
    if not args.interaction_table.is_file():
        parser.error("interaction table is missing; no substitute table is used")
    case = args.output_root / str(args.temperature)
    audit = args.output_root / "audit" / str(args.temperature)
    if case.exists() or audit.exists():
        parser.error("choose a new output directory; existing results are never overwritten")
    from wd_spectra.models import DBConfig
    DBConfig(effective_temperature=args.temperature, logg=args.logg, quality='production')
    print("Experimental pure-He physics; reference cold points are 5000 and 8000 K at log g=8. "
          f"Requested log g={args.logg:g}; other parameters remain unvalidated and must pass the same numerical audit.", flush=True)
    for command in commands(args.temperature, args.output_root, args.interaction_table, logg=args.logg):
        subprocess.run(command, check=True)
    qualified = json.loads((audit / "qualification.json").read_text())
    if qualified.get("numerically_qualified_for_declared_experimental_physics") is not True:
        message="Numerical qualification FAILED. Exploratory outputs retained; no replacement used."
        if not args.allow_unqualified:
            raise SystemExit(message)
        print(message,flush=True)
        return
    print("Declared-equation numerical checks passed; this is not full-physics or depth-grid validation.", flush=True)


if __name__ == "__main__":
    main()
