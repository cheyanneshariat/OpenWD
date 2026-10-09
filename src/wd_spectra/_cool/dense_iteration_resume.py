"""Explicit continuation of an UNCONVERGED same-physics iteration.

Unlike TemperaturePredictor this does not certify its source, scale Teff,
resample a grid, change pressure limits, or project temperatures. The new
solver must measure its own correction and satisfy every physical gate.
"""
import hashlib
import io
import json
from pathlib import Path
import numpy as np


class IterationResume:
    reuse_pressure=True
    def __init__(self,directory,interaction_table,*,pseudo_time_snapshot=False):
        from .dense_helium_molecular_experiment import PHYSICS
        self.directory=Path(directory).resolve()
        self.sha=hashlib.sha256(Path(interaction_table).read_bytes()).hexdigest()
        options=json.loads((self.directory/'experiment-options.json').read_text())
        from .heminus_join_experiment import HARD_JOIN,SMOOTH_JOIN
        self.heminus_join_policy=(SMOOTH_JOIN if options.get('experimental_thermal_driver_options',{}).get('smooth_heminus_join',False)
                                 else HARD_JOIN)
        if pseudo_time_snapshot:
            if options.get('experimental_material_certificate_version')!=2:
                raise ValueError('pseudo-time resume requires the version-2 run material certificate')
            self.path=Path(options['experimental_thermal_driver_options']['pseudo_time_output']).resolve().with_suffix('.npz')
        else:
            self.path=self.directory/'experimental-latest-iteration.npz'
        # Read once: telemetry may be running concurrently. The recorded
        # SHA must describe exactly the bytes from which the state was read.
        raw=self.path.read_bytes()
        self.file_sha=hashlib.sha256(raw).hexdigest()
        self.teff=options['temperature']
        self.logg=float(options.get('logg',8.))
        if options['log_h_he'] is not None or options['experimental_physics']!=PHYSICS:
            raise ValueError('iteration source composition/physics mismatch')
        table=Path(options['experimental_dense_options']['interaction_table']).resolve()
        if table!=Path(interaction_table).resolve():
            raise ValueError('iteration source interaction-table path mismatch')
        with np.load(io.BytesIO(raw)) as data:
            if pseudo_time_snapshot:
                if not bool(data['initialization_only']) or bool(data['static_convergence_claim']):
                    raise ValueError('pseudo-time snapshot must be explicitly uncertified initialization')
            elif str(data['experimental_physics'])!=PHYSICS:
                raise ValueError('iteration snapshot physics mismatch')
            self.temperature=data['experimental_temperature'].copy()
            self.pressure=data['experimental_gas_pressure'].copy()
            self.tau=data['experimental_rosseland_optical_depth'].copy()
        # Earlier hierarchy checkpoints record the material SHA explicitly;
        # do not infer identity from a matching filename alone.
        version=options.get('experimental_material_certificate_version',1)
        if version==2:
            # New runs certify their table before the first iteration,
            # including warm starts which never build a discrete seed.
            certificate=self.directory/'experiment-options.json'
            if options['experimental_material_table_sha256']!=self.sha:
                raise ValueError('iteration source material certificate mismatch')
        elif version==1:
            certificate=self.directory/f'experimental-discrete-seed-{len(self.temperature)}.npz'
            with np.load(certificate) as data:
                if str(data['interaction_table_sha256'])!=self.sha or str(data['experimental_physics'])!=PHYSICS:
                    raise ValueError('iteration source material certificate mismatch')
        else:
            raise ValueError('unsupported experimental material certificate version')
        self.certificate_sha=hashlib.sha256(certificate.read_bytes()).hexdigest()
        self.kind='pseudo-time initializer' if pseudo_time_snapshot else 'iteration'
        self.description=(f'EXPLICIT continuation of UNCONVERGED {self.kind} from {self.path}; '
            'identical Teff, pressure grid and temperature values; NOT a cold start or convergence claim')
    def seed(self,teff,n_depth,bottom_tau,table=None,mesh='pressure',*,logg=8.):
        if teff!=self.teff or logg!=self.logg or n_depth!=len(self.temperature) or table is not None:
            raise ValueError('iteration resume requires identical Teff, gravity, layer count and EOS')
        from . import check_cool_db_transport_seed as runner
        return runner.atmosphere_at(teff,self.pressure.copy(),self.temperature.copy(),self.tau.copy(),logg=logg)
    def __call__(self,seed,options):
        np.testing.assert_array_equal(seed.temperature,self.temperature)
        np.testing.assert_array_equal(seed.gas_pressure,self.pressure)
        options={**options,'metadata':{**options.get('metadata',{}),
            'experimental_unconverged_iteration_source':str(self.path),
            'experimental_unconverged_iteration_source_sha256':self.file_sha,
            'experimental_unconverged_iteration_material_certificate_sha256':self.certificate_sha,
            'experimental_iteration_source_interaction_table_sha256':self.sha,
            'experimental_iteration_resume_is_cold_start':False,
            'experimental_iteration_source_kind':self.kind,
            'experimental_iteration_resume_is_convergence':False}}
        print(self.description,flush=True)
        return seed,options
