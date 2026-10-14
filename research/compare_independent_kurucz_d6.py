"""Compare saved J1637 spectra with one frozen baseline continuum transfer.

No atmosphere solve, abundance adjustment or candidate continuum fit occurs.
The bundled paper digitization has no errors: report squared residuals, not chi2.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from wd_spectra.validation.hollands_d6 import (
    HOLLANDS2025_PANEL_WINDOWS, convolve_to_resolving_power,
    load_hollands2025_digitization, DEFAULT_DIGITIZATION,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with np.load(args.baseline, allow_pickle=False) as saved:
        wave = saved['wavelength']
        baseline = convolve_to_resolving_power(wave, saved['surface_flux'], 2500)
    with np.load(args.candidate, allow_pickle=False) as saved:
        np.testing.assert_array_equal(wave, saved['wavelength'])
        candidate = convolve_to_resolving_power(wave, saved['surface_flux'], 2500)
    published = load_hollands2025_digitization()
    scores = {}
    for target in ('observed', 'koester'):
        panels = []
        for panel, (lower, upper) in enumerate(HOLLANDS2025_PANEL_WINDOWS, 1):
            w = published[f'panel{panel}_{target}_wavelength']
            f = published[f'panel{panel}_{target}_normalized_flux']
            order = np.argsort(w); w, f = w[order], f[order]
            valid = np.isfinite(f) & (f > .05) & (w >= wave[0]) & (w <= wave[-1])
            if target == 'observed':
                valid &= f < 1.35
            w, f = w[valid], f[valid]
            old, new = np.interp(w, wave, baseline), np.interp(w, wave, candidate)
            x = 2 * (w - lower) / (upper - lower) - 1
            ratio = np.log(f / old)
            edges = np.linspace(x.min(), x.max(), 41)
            bx, by = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                take = (x >= lo) & (x <= hi)
                if np.count_nonzero(take) >= 3:
                    bx.append(np.median(x[take])); by.append(np.median(ratio[take]))
            coefficients = np.polynomial.polynomial.polyfit(bx, by, 2)
            transfer = np.exp(np.polynomial.polynomial.polyval(x, coefficients))
            old_sum = float(np.sum((old * transfer - f)**2))
            new_sum = float(np.sum((new * transfer - f)**2))
            panels.append(dict(panel=panel, pixels=len(w), baseline_squared_residual=old_sum,
                candidate_squared_residual=new_sum, relative_change=new_sum / old_sum - 1,
                baseline_continuum_coefficients=coefficients.tolist()))
        old_sum = sum(p['baseline_squared_residual'] for p in panels)
        new_sum = sum(p['candidate_squared_residual'] for p in panels)
        scores[target] = dict(panels=panels, baseline_squared_residual=old_sum,
            candidate_squared_residual=new_sum, relative_change=new_sum / old_sum - 1)
    record = dict(scope='fixed historical atmosphere, R=2500, baseline continuum transfer reused',
        score_kind='unweighted squared residual, not chi-square',
        candidate_continuum_refitted=False, parameters_or_abundances_changed=False,
        inputs={role:dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for role,p in (('baseline',args.baseline),('candidate',args.candidate),
                               ('digitization',DEFAULT_DIGITIZATION))}, scores=scores)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k:v['relative_change'] for k,v in scores.items()}))


if __name__ == '__main__':
    main()
