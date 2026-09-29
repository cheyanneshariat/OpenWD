"""Propose a bounded Teff/logg step from three actual weak-field models.

This is a local sensitivity experiment, not a posterior or an uncertainty
estimate. Halpha and Hbeta choose the step; Hgamma and Hdelta are held out.
The proposed parameters must be recomputed with a fresh atmosphere.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from wd_spectra.validation.magnetic_da import weak_field_profile_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("temperature_trial", type=Path)
    parser.add_argument("gravity_trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    directories = [args.baseline, args.temperature_trial, args.gravity_trial]
    scores = [json.loads((p / "scores.json").read_text()) for p in directories]
    if any(s["convergence"] != "converged" for s in scores):
        raise SystemExit("All three input atmospheres must be converged")
    target = scores[0]["target"]
    if any(s["target"] != target for s in scores):
        raise SystemExit("Input targets differ")
    configs = [json.loads((p / "request.json").read_text())["config"] for p in directories]
    keys = ("effective_temperature", "logg")
    differences = np.array([[c[k] - configs[0][k] for k in keys] for c in configs[1:]])
    if not (differences[0, 0] and differences[1, 1]) or differences[0, 1] or differences[1, 0]:
        raise SystemExit("Trials must change only temperature or only gravity, respectively")
    for c in configs[1:]:
        if any(c[k] != configs[0][k] for k in c if k not in keys):
            raise SystemExit("Non-fitted settings must match")
    profiles = []
    for p in directories:
        wave, flux = np.loadtxt(p / "model/spectrum.txt", unpack=True)
        profiles.append(weak_field_profile_arrays(target, wave, flux, scores[0]["epoch_velocities_kms"]))
    # Scaled coordinates avoid mixing kelvin and dex in the least-squares solve.
    scales = np.array([1000., .1])
    jacobians, residuals = {}, {}
    for line, (wave, observed, baseline) in profiles[0].items():
        residuals[line] = baseline - observed
        jacobians[line] = np.column_stack([
            (np.interp(wave, p[line][0], p[line][2]) - baseline) / delta * scale
            for p, delta, scale in zip(profiles[1:], differences.diagonal(), scales)
        ])
    training = ("Halpha", "Hbeta")
    fit = lsq_linear(
        np.vstack([jacobians[k] / np.sqrt(residuals[k].size) for k in training]),
        np.concatenate([-residuals[k] / np.sqrt(residuals[k].size) for k in training]),
        bounds=([-2., -3.], [2., 3.]),
    )
    proposal = {
        "target": target,
        "inputs": [str(p.resolve()) for p in directories],
        "training_lines": training,
        "held_out_lines": ["Hgamma", "Hdelta"],
        "epoch_velocities_kms": scores[0]["epoch_velocities_kms"],
        "proposed_parameters": {k: float(configs[0][k] + dx) for k, dx in zip(keys, fit.x * scales)},
        "linear_prediction_rms": {
            k: float(np.sqrt(np.mean((residuals[k] + jacobians[k] @ fit.x) ** 2)))
            for k in residuals
        },
        "bound_active": fit.active_mask.tolist(),
        "requires_fresh_model_verification": True,
        "interpretation": "Local exploratory parameter fit; no parameter uncertainties inferred",
    }
    args.output.write_text(json.dumps(proposal, indent=2))
    print(json.dumps(proposal, indent=2))


if __name__ == "__main__":
    main()
