"""Strict structure-grid certification; no inherited or surface-only success.

Independent quadrature/depth/domain/physics validation remains separate.
This checks recorded numerical evidence; it does not re-solve the atmosphere.
"""
import json
from pathlib import Path
import numpy as np


def convergence_failures(summary,metadata,direct,history):
    failures=[]
    a=metadata.get('atmosphere_metadata',{})
    def small(value,tolerance):
        try:return bool(np.isfinite(value) and 0<=value<tolerance)
        except (TypeError,ValueError):return False
    if summary.get('converged') is not True or a.get('radiative_equilibrium_converged') is not True:
        failures.append('atmosphere does not report convergence')
    if metadata.get('config',{}).get('include_molecules') is not True:
        failures.append('molecular chemistry was not selected')
    for name in ('maximum_all_depth_total_flux_residual','maximum_relative_cell_energy_balance_residual'):
        if not small(summary.get(name),.003):failures.append(name)
    if not small(a.get('radiative_equilibrium_maximum_log_temperature_correction'),3e-4):
        failures.append('measured temperature correction')
    segments=a.get('nonlinear_solver_segments',[])
    if not segments or not str(segments[-1].get('phase','')).startswith('formal-radiative-flux'):
        failures.append('no final physical-flux phase')
    final=history[-1].get('diagnostics',{}) if history else {}
    if final.get('converged') is not True or direct.get('converged') is not True:
        failures.append('final solver/transfer record does not report convergence')
    if not str(final.get('solver_phase','')).startswith('formal-radiative-flux'):
        failures.append('final telemetry is not a physical-flux phase')
    proposal_measured=(a.get('diagnostic_requires_non_trust_limited_correction') is True or
                      final.get('line_search_factor')==0.)
    radius=final.get('trust_radius',0.)
    if not (proposal_measured and small(radius,np.inf) and radius>=3e-4 and
            small(final.get('maximum_log_temperature_correction'),3e-4)):
        failures.append('no non-trust-limited correction certificate')
    if not small(direct.get('source_closure_scaled_error'),1e-10):
        failures.append('independent prescribed-source closure')
    ratio=direct.get('spectrum_integral_over_sigma_teff4')
    if not small(ratio,np.inf) or not small(abs(ratio-1),.003):
        failures.append('matched structure-grid surface flux')
    if not small(summary.get('bottom_absorption_escape_bound'),.003):
        failures.append('bottom absorption escape bound')
    return failures


def check_result(root):
    root=Path(root)
    read=lambda name:json.loads((root/name).read_text())
    history=[json.loads(row) for row in (root/'iterations.jsonl').read_text().splitlines() if row.strip()]
    failures=convergence_failures(read('summary.json'),read('metadata.json'),read('direct-transfer.json'),history)
    report=dict(structure_grid_convergence_verified=not failures,failures=failures,
                independent_mesh_validation=False,full_physics_validation=False,root=str(root))
    print(json.dumps(report),flush=True)
    return not failures


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    raise SystemExit(0 if check_result(parser.parse_args().root) else 1)
