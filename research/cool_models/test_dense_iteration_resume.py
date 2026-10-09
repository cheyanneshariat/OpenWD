import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
from dense_iteration_resume import IterationResume
from dense_helium_molecular_experiment import PHYSICS


def fixture(tmp_path,logg=None):
    table=tmp_path/'table';table.write_bytes(b'unique interaction table')
    sha=hashlib.sha256(table.read_bytes()).hexdigest()
    np.savez(tmp_path/'experimental-latest-iteration.npz',experimental_physics=PHYSICS,
        experimental_temperature=np.array([3000.,4000.]),experimental_gas_pressure=np.array([1e7,1e8]),
        experimental_rosseland_optical_depth=np.array([1e-6,.1]))
    np.savez(tmp_path/'experimental-discrete-seed-2.npz',experimental_physics=PHYSICS,interaction_table_sha256=sha)
    options=dict(temperature=8000,log_h_he=None,experimental_physics=PHYSICS,
                 experimental_dense_options=dict(interaction_table=str(table)))
    if logg is not None: options['logg']=logg
    (tmp_path/'experiment-options.json').write_text(json.dumps(options))
    return table


@pytest.mark.parametrize('logg', [7.75, 8.0])
def test_resume_retains_exact_state_and_marks_unconverged(tmp_path,monkeypatch,logg):
    table=fixture(tmp_path,logg);source=IterationResume(tmp_path,table)
    import check_cool_db_transport_seed as runner
    monkeypatch.setattr(runner,'atmosphere_at',lambda teff,p,t,tau,*,logg:
        SimpleNamespace(gas_pressure=p,temperature=t,rosseland_optical_depth=tau,logg=logg))
    seed=source.seed(8000,2,100,logg=logg)
    result,options=source(seed,{})
    assert result is seed
    assert seed.logg == logg
    np.testing.assert_array_equal(seed.temperature,[3000.,4000.])
    assert not options['metadata']['experimental_iteration_resume_is_convergence']
    assert not options['metadata']['experimental_iteration_resume_is_cold_start']
    with pytest.raises(ValueError,match='identical'):source.seed(7500,2,100,logg=logg)
    with pytest.raises(ValueError,match='identical'):source.seed(8000,4,100,logg=logg)
    with pytest.raises(ValueError,match='identical'):source.seed(8000,2,100,logg=logg+.25)


def test_resume_refuses_changed_material_file(tmp_path):
    table=fixture(tmp_path);table.write_bytes(b'changed table')
    with pytest.raises(ValueError,match='certificate'):IterationResume(tmp_path,table)


def test_version_two_certificate_supports_warm_iterations_without_discrete_seed(tmp_path):
    table=fixture(tmp_path);path=tmp_path/'experiment-options.json'
    options=json.loads(path.read_text())
    options.update(experimental_material_certificate_version=2,
        experimental_material_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest())
    path.write_text(json.dumps(options))
    (tmp_path/'experimental-discrete-seed-2.npz').unlink()
    source=IterationResume(tmp_path,table)
    np.testing.assert_array_equal(source.temperature,[3000.,4000.])
    options['experimental_material_table_sha256']='wrong'
    path.write_text(json.dumps(options))
    with pytest.raises(ValueError,match='certificate'):IterationResume(tmp_path,table)


def test_explicit_pseudo_time_snapshot_is_not_called_a_converged_iteration(tmp_path):
    table=fixture(tmp_path);path=tmp_path/'experiment-options.json';snapshot=tmp_path/'pseudo.npz'
    np.savez(snapshot,experimental_temperature=np.array([3100.,4200.]),
        experimental_gas_pressure=np.array([1e7,1e8]),experimental_rosseland_optical_depth=np.array([1e-6,.1]),
        initialization_only=True,static_convergence_claim=False)
    options=json.loads(path.read_text())
    options.update(experimental_material_certificate_version=2,
        experimental_material_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
        experimental_thermal_driver_options=dict(pseudo_time_output=str(snapshot.with_suffix('.jsonl'))))
    path.write_text(json.dumps(options))
    source=IterationResume(tmp_path,table,pseudo_time_snapshot=True)
    np.testing.assert_array_equal(source.temperature,[3100.,4200.])
    assert source.kind=='pseudo-time initializer'
    assert 'UNCONVERGED pseudo-time initializer' in source.description
