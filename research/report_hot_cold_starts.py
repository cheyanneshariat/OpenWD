"""Report cold-start evidence separately from saved-model equilibrium checks."""
import argparse,json,math,os
from pathlib import Path

CASES=(
    ('do-50000-standard-fresh-release',50000.,8.,None,'standard',32,8),
    ('do-60000-standard-baseline',60000.,8.,None,'standard',32,8),
    ('do-70000-standard-baseline',70000.,8.,None,'standard',32,8),
    ('dao-60000-standard-baseline',60000.,8.,2.,'standard',32,8),
    ('gd153-standard-baseline',40204.,7.82,6.,'standard',32,8),
    ('gd153-production-h8-baseline',40204.,7.82,6.,'production',8,8),
    ('gd153-production-h20-baseline',40204.,7.82,6.,'production',8,20),
)

CERTIFICATE_FIELDS = dict(
    all_depth_flux='maximum_all_depth_total_flux_residual',
    local_energy='maximum_relative_cell_energy_balance_residual',
    temperature_stationarity='maximum_unrestricted_log_temperature_correction',
    source_closure='electron_scattering_source_final_maximum_relative_residual',
    boundary_screening='lower_boundary_absorption_escape_bound')


def measured_equilibrium(model):
    """Recheck finite recorded measurements; a copied True flag is insufficient."""
    atmosphere = model.get('atmosphere_metadata', {})
    certificate = atmosphere.get('equilibrium_certificate', {})
    if (certificate.get('verified') is not True
            or atmosphere.get('radiative_equilibrium_solver_converged') is not True
            or atmosphere.get('nlte_populations_converged') is not True
            or atmosphere.get('temperature_correction_measured') is not True):
        return False
    def below(value, tolerance):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool)
                and math.isfinite(value) and math.isfinite(tolerance)
                and 0. <= value < tolerance)
    if not below(atmosphere.get('nlte_maximum_relative_population_change'),
                 model.get('config', {}).get('population_tolerance')):
        return False
    for name, field in CERTIFICATE_FIELDS.items():
        check = certificate.get('checks', {}).get(name, {})
        if (check.get('passed') is not True or check.get('measured') is not True
                or check.get('value') != atmosphere.get(field)
                or not below(atmosphere.get(field), check.get('tolerance'))):
            return False
    return True


def model_declares_restart(model):
    metadata = model.get('model_metadata', {})
    return (metadata.get('cold_start') is False
            or metadata.get('initialization', {}).get('previous_model_supplied') is True)


def read_optional_json(path):
    # Another cold worker may be publishing a progress record right now.
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def public_preset_qualified(record, provenance, model, case):
    """A cold research experiment does not qualify the ordinary public preset."""
    if (not record['cold_start_pass'] or not provenance or not model
            or not measured_equilibrium(model)
            or provenance.get('experiment') is not None
            # Early instrumented runs predate the model-level label; their
            # explicit cold-start provenance already establishes this in
            # record['cold_start_pass']. A contradictory label is disallowed.
            or model_declares_restart(model)):
        return False
    _,teff,logg,ratio,quality,he,h=case
    expected=dict(effective_temperature=teff,logg=logg,quality=quality,
                  maximum_helium_ii_level=he,log_hydrogen_to_helium=ratio,
                  population_maximum_iterations=120,population_tolerance=1e-4,
                  hydrogenic_collision_model='tlusty-mihalas',
                  helium_i_profile='Tremblay26.txt',helium_ii_interpolation='series-adaptive')
    if ratio is not None:expected['maximum_hydrogen_level']=h
    unchanged_config=all(config.get(key)==value
               for config in (provenance.get('config',{}),model.get('config',{}))
               for key,value in expected.items())
    checks=model.get('atmosphere_metadata',{}).get('equilibrium_certificate',{}).get('checks',{})
    tolerances=dict(all_depth_flux=3e-3,local_energy=3e-3,temperature_stationarity=3e-4,
                    source_closure=1e-6,boundary_screening=3e-3)
    return unchanged_config and all(checks.get(key,{}).get('tolerance')==tolerance
                                    for key,tolerance in tolerances.items())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/hot-nlte-migration/cold-audit'))
    parser.add_argument('--require-complete',action='store_true',
                        help='Exit nonzero unless every required public cold case passes')
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    report=dict(scope=__doc__.strip(),cases=[])
    lines=['# Cold-start qualification', '',
           'Only a completed invocation with no saved atmosphere/population inputs and a verified',
           'final equilibrium certificate counts as a cold-start pass. Internal initialization',
           'at the requested stellar parameters is allowed. Saved-model continuations and',
           'fixed-state residual checks are diagnostics, not cold-start passes.', '',
           'For running models, elapsed time is the latest recorded progress; completed',
           'times include fresh initialization. Interrupted models did not necessarily',
           'exhaust their iteration budget.', '',
           '| Required public preset | Status | Qualified | Recorded elapsed (s) |',
           '| --- | --- | --- | --- |']
    cases=list(CASES)
    for path in sorted(args.root.glob('*/cold-start-provenance.json')):
        if path.parent.name in {case[0] for case in cases} or 'instrumentation-failed' in path.parent.name:
            continue
        saved=read_optional_json(path)
        if saved is None:continue
        config=saved['config']
        cases.append((path.parent.name,config['effective_temperature'],config['logg'],
                      config.get('log_hydrogen_to_helium'),config['quality'],
                      config['maximum_helium_ii_level'],config.get('maximum_hydrogen_level',8)))
    for name,teff,logg,ratio,quality,he,h in cases:
        directory=args.root/name
        record=dict(name=name,effective_temperature=teff,logg=logg,log_hydrogen_to_helium=ratio,
                    quality=quality,maximum_helium_ii_level=he,maximum_hydrogen_level=h,
                    status='not run',cold_start_pass=False)
        if directory.exists():record['status']='running or interrupted; no completed cold certificate'
        provenance=directory/'cold-start-provenance.json'
        saved=read_optional_json(provenance)
        model=None
        if saved is not None:
            record.update(status=saved['status'],elapsed_seconds=saved['elapsed_seconds'])
            record['experiment']=saved.get('experiment')
            metadata=directory/'metadata.json'
            model=read_optional_json(metadata)
            if model is not None:
                certificate=model['atmosphere_metadata'].get('equilibrium_certificate',{})
                record['certificate']=certificate
                # A saved certificate can precede the end of run_model.
                # Only the harness's terminal provenance marks completion.
                record['cold_start_pass']=bool(record['status']=='converged' and saved['cold_start']
                    and saved['initial_model_supplied'] is False and saved['checkpoint_inputs']==[]
                    and not model_declares_restart(model)
                    and measured_equilibrium(model))
        record['public_preset_qualified']=public_preset_qualified(
            record,saved,model,(name,teff,logg,ratio,quality,he,h))
        progress=directory/'numerical-diagnostics/latest-progress.json'
        progress_record=read_optional_json(progress)
        if progress_record is not None:
            record['latest_progress']=progress_record
            if record['status']=='running':
                record['elapsed_seconds']=progress_record.get('elapsed_seconds',record.get('elapsed_seconds'))
        report['cases'].append(record)
        seconds=record.get('elapsed_seconds')
        if len(report['cases']) == len(CASES)+1:
            lines += ['', 'Supplementary runs and historical attempts (excluded from the required total):', '',
                      '| Case | Status | Cold-start pass | Recorded elapsed (s) |',
                      '| --- | --- | --- | --- |']
        required=name in {row[0] for row in CASES}
        qualifies=record['public_preset_qualified'] if required else record['cold_start_pass']
        passed=('Yes' if qualifies else 'No' if record['status'] in ('converged','failed','interrupted') else 'Pending')
        lines.append(f"| {name} | {record['status']} | {passed} | "
                     +(f'{seconds:.1f}' if seconds is not None else '—')+' |')
    report['public_preset_cold_passes']=sum(case['public_preset_qualified'] for case in report['cases']
                                         if case['name'] in {row[0] for row in CASES})
    report['required_public_cases']=len(CASES)
    lines[2:2]=[f"Public-preset cold passes: {report['public_preset_cold_passes']}/{len(CASES)}. "
                'Supplementary experiments do not count toward this total.','']
    for name,contents in [('report.json',json.dumps(report,indent=2)+'\n'),('REPORT.md','\n'.join(lines)+'\n')]:
        temporary=args.root/f'{name}.{os.getpid()}.tmp'
        temporary.write_text(contents)
        temporary.replace(args.root/name)
    print('\n'.join(lines))
    if args.require_complete and report['public_preset_cold_passes']!=len(CASES):
        raise SystemExit(1)


if __name__=='__main__':main()
