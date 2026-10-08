"""Prevent warm, altered-parameter or experimental results qualifying release."""
from copy import deepcopy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from report_hot_cold_starts import CERTIFICATE_FIELDS,main,public_preset_qualified


class ColdQualificationTests(unittest.TestCase):
    def setUp(self):
        self.case=('dao-test',60000.,8.,2.,'standard',32,8)
        config=dict(effective_temperature=60000.,logg=8.,quality='standard',
                    log_hydrogen_to_helium=2.,maximum_helium_ii_level=32,
                    maximum_hydrogen_level=8,population_maximum_iterations=120,
                    population_tolerance=1e-4,hydrogenic_collision_model='tlusty-mihalas',
                    helium_i_profile='Tremblay26.txt',helium_ii_interpolation='series-adaptive')
        self.record=dict(cold_start_pass=True)
        self.provenance=dict(config=deepcopy(config),experiment=None)
        self.model=dict(config=deepcopy(config),model_metadata=dict(cold_start=True))
        tolerances=dict(all_depth_flux=3e-3,local_energy=3e-3,temperature_stationarity=3e-4,
                        source_closure=1e-6,boundary_screening=3e-3)
        self.checks={key:dict(tolerance=value,value=value/10,passed=True,measured=True)
                     for key,value in tolerances.items()}
        self.model['atmosphere_metadata']=dict(
            equilibrium_certificate=dict(checks=self.checks,verified=True),
            radiative_equilibrium_solver_converged=True,nlte_populations_converged=True,
            temperature_correction_measured=True,nlte_maximum_relative_population_change=1e-5,
            **{field:self.checks[name]['value'] for name,field in CERTIFICATE_FIELDS.items()})

    def qualified(self):
        return public_preset_qualified(self.record,self.provenance,self.model,self.case)

    def test_matching_public_cold_pass(self):
        self.assertTrue(self.qualified())

    def test_research_hook_does_not_qualify_public_path(self):
        self.provenance['experiment']={'scope':'research proposal'}
        self.assertFalse(self.qualified())

    def test_saved_model_not_cold(self):
        self.model['model_metadata']['cold_start']=False
        self.assertFalse(self.qualified())

    def test_older_metadata_uses_verified_instrumentation_provenance(self):
        self.model['model_metadata'].pop('cold_start')
        self.assertTrue(self.qualified())

    def test_missing_or_failed_certificate(self):
        self.record['cold_start_pass']=False
        self.assertFalse(self.qualified())

    def test_copied_certificate_does_not_hide_nonstationary_temperature(self):
        atmosphere=self.model['atmosphere_metadata']
        atmosphere['maximum_unrestricted_log_temperature_correction']=0.45
        self.assertFalse(self.qualified())
        self.checks['temperature_stationarity']['value']=0.45
        self.assertFalse(self.qualified())

    def test_small_thermal_residuals_do_not_hide_unconverged_populations(self):
        self.model['atmosphere_metadata']['nlte_maximum_relative_population_change']=0.02
        self.assertFalse(self.qualified())

    def test_unmeasured_or_nonfinite_diagnostics_do_not_qualify(self):
        atmosphere=self.model['atmosphere_metadata']
        atmosphere['temperature_correction_measured']=False
        self.assertFalse(self.qualified())
        atmosphere['temperature_correction_measured']=True
        for name,field in CERTIFICATE_FIELDS.items():
            value=atmosphere[field]
            for invalid in (None,float('nan'),float('inf'),-1.,True):
                with self.subTest(check=name,value=invalid):
                    atmosphere[field]=invalid
                    self.checks[name]['value']=invalid
                    self.assertFalse(self.qualified())
            atmosphere[field]=value
            self.checks[name]['value']=value

    def test_relaxed_certificate_threshold_cannot_qualify(self):
        for key in self.checks:
            with self.subTest(check=key):
                previous=self.checks[key]['tolerance']
                self.checks[key]['tolerance']*=10
                self.assertFalse(self.qualified())
                self.checks[key]['tolerance']=previous

    def test_changed_parameters_or_resolution_in_either_record(self):
        for container in (self.provenance,self.model):
            for key in container['config']:
                with self.subTest(record=container is self.model,key=key):
                    previous=container['config'][key]
                    container['config'][key]='different'
                    self.assertFalse(self.qualified())
                    container['config'][key]=previous

    def test_saved_certificate_during_running_invocation_is_not_a_cold_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            directory=root/'dao-60000-standard-baseline'
            directory.mkdir()
            provenance=dict(self.provenance,status='running',elapsed_seconds=10.,
                            cold_start=True,initial_model_supplied=False,checkpoint_inputs=[])
            self.model['atmosphere_metadata']['equilibrium_certificate']['verified']=True
            (directory/'cold-start-provenance.json').write_text(json.dumps(provenance))
            (directory/'metadata.json').write_text(json.dumps(self.model))
            with patch.object(sys,'argv',['report','--root',str(root),'--require-complete']):
                with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as raised:
                    main()
            self.assertEqual(raised.exception.code,1)
            report=json.loads((root/'report.json').read_text())
            self.assertEqual(report['public_preset_cold_passes'],0)
            record=next(item for item in report['cases'] if item['name']==directory.name)
            self.assertEqual(record['status'],'running')
            self.assertFalse(record['cold_start_pass'])

    def test_warm_metadata_cannot_be_overridden_by_cold_provenance(self):
        for metadata in (dict(cold_start=False),
                         dict(cold_start=True,initialization=dict(previous_model_supplied=True))):
            with self.subTest(metadata=metadata),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                directory=root/'dao-60000-standard-baseline'
                directory.mkdir()
                provenance=dict(self.provenance,status='converged',elapsed_seconds=10.,
                                cold_start=True,initial_model_supplied=False,checkpoint_inputs=[])
                model=deepcopy(self.model)
                model['model_metadata']=metadata
                (directory/'cold-start-provenance.json').write_text(json.dumps(provenance))
                (directory/'metadata.json').write_text(json.dumps(model))
                with patch.object(sys,'argv',['report','--root',str(root)]):
                    with contextlib.redirect_stdout(io.StringIO()):main()
                report=json.loads((root/'report.json').read_text())
                record=next(item for item in report['cases'] if item['name']==directory.name)
                self.assertFalse(record['cold_start_pass'])
                self.assertFalse(record['public_preset_qualified'])


if __name__=='__main__':unittest.main()
