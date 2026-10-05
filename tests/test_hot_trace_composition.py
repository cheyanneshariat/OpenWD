"""Atomic supplement unit checks and multielement detailed balance."""
from pathlib import Path
import sys
import gzip
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from hot_trace_composition import (OuterShellFit, outer_shell_fits,
                                  convert_iron_group_lines, load_composition_data)
from wd_spectra.atmosphere import gray_hydrogen_atmosphere
from wd_spectra.hot_trace_metals import (fixed_electron_metal_reference,solve_hot_trace_metals,
                                       TraceMetalConvergenceWarning)
from wd_spectra.light_metal_nlte import (reduced_light_metal_wavelength,
    solve_reduced_light_metal_levels_nlte, light_metal_bound_free_nlte_coefficients)
from wd_spectra.metals import read_pg1159_atomic_database,read_verner_photoionization_database
from wd_spectra.models.common import ModelData
from wd_spectra.nlte_core import NLTETransferCoefficients
from wd_spectra.spectrum import planck_lambda_angstrom


def test_outer_shell_formula_and_domain():
    fit=OuterShellFit('P',3,3,1,50.,10.,2.,4.,6.,.5)
    # At E=100 eV: y=10, Q=3.5, sigma0=2 Mb.
    expected=2e-18*((10-1)**2+.25)*10**-3.5*(1+np.sqrt(2.5))**-6
    np.testing.assert_allclose(fit.cross_section([49.,100.,100001.]),[0.,expected,0.],rtol=1e-14)
    with pytest.raises(ValueError):fit.cross_section([0.])


def test_iron_group_format_preserves_parity_order_and_gf(tmp_path):
    source=tmp_path/'test.gz';target=tmp_path/'test.gf'
    with gzip.open(source,'wt') as f:
        f.write('100.0000 28.04 -0.500 150000.000 2.0 50000.000 1.0 8.00 -6.50 -7.00 0\n')
        f.write('100.0000 6.02 -0.500 0.000 2.0 100000.000 1.0 8.00 -6.50 -7.00 0\n')
    assert convert_iron_group_lines(source,target)=={'28.04':1}
    s=target.read_text()
    assert float(s[11:18])==-.5 and float(s[24:36])==150000.
    assert float(s[51:63])==50000. and float(s[85:91])==-6.5


def test_partial_warm_start_includes_new_element_lte_in_callback():
    data=ModelData.default();db=read_pg1159_atomic_database(data.stout,elements=('C','N'))
    photo=read_verner_photoionization_database(data.verner_photoionization,elements=('C','N'))
    atmosphere=gray_hydrogen_atmosphere(52500.,7.53,n_depth=3,tau_max=2.)
    abundance={'C':-7.,'N':-7.};ref=fixed_electron_metal_reference(atmosphere,db,abundance)
    counts={'C':{2:3,3:3,4:1},'N':{2:3,3:3,4:1}}
    grid=np.unique(np.concatenate([reduced_light_metal_wavelength(db,e,n,n_continuum_wavelength=120)
                           for e,n in counts.items()]))
    b=planck_lambda_angstrom(grid[:,None],atmosphere.temperature)
    seed=solve_reduced_light_metal_levels_nlte(atmosphere,db,ref,photo,grid,.4*b,'C',counts['C'])
    def background(w):
        bb=planck_lambda_angstrom(w[:,None],atmosphere.temperature)
        return NLTETransferCoefficients(w,np.ones_like(bb),bb,np.zeros_like(bb),{})
    def callback(i,d,states,synthesize):
        assert set(states)=={'C','N'}
        np.testing.assert_array_equal(states['N'].population_density,states['N'].lte_population_density)
        return True
    with pytest.warns(TraceMetalConvergenceWarning):
        result=solve_hot_trace_metals(atmosphere,background,abundance,np.linspace(1237.,1244.,40),
            atomic_database=db,photoionization_database=photo,levels_per_charge=counts,
            initial_populations={'C':seed},population_wavelength=grid,
            require_convergence=False,state_callback=callback,tolerance=1e-10)
    assert set(result.populations)=={'C','N'}
    with pytest.raises(ValueError,match='explicit levels_per_charge'):
        solve_hot_trace_metals(atmosphere,background,{'N':-7.},grid,
                              atomic_database=db,photoionization_database=photo)


@pytest.mark.parametrize('element,counts',[
 ('N',{2:8,3:8,4:8,5:1}),('O',{2:8,3:8,4:8,5:1}),
 ('Al',{2:8,3:8,4:1}),('P',{2:8,3:8,4:8,5:1}),('S',{2:8,3:8,4:8,5:1})])
def test_added_atoms_preserve_planck_detailed_balance(element,counts):
    directory=Path(__file__).resolve().parents[1]/'results/hot-daz/multimetal-data'
    if not (directory/'verner95.dat').exists():pytest.skip('downloaded research atomic data not present')
    db,photo,_=load_composition_data(ModelData.default(),directory,iron_group=False)
    atmosphere=gray_hydrogen_atmosphere(52500.,7.53,n_depth=3,tau_max=2.)
    ref=fixed_electron_metal_reference(atmosphere,db,{element:-7.})
    wave=reduced_light_metal_wavelength(db,element,counts,n_continuum_wavelength=120)
    b=planck_lambda_angstrom(wave[:,None],atmosphere.temperature)
    state=solve_reduced_light_metal_levels_nlte(atmosphere,db,ref,photo,wave,b,element,counts)
    np.testing.assert_allclose(state.population_density,state.lte_population_density,rtol=3e-6)
    unity={(element,s.charge):np.ones(atmosphere.n_depth) for s in db.ion_stages(element)}
    selected={s.charge:counts.get(s.charge,0) if s.charge<max(counts) else 0
              for s in db.ion_stages(element)}
    a,j=light_metal_bound_free_nlte_coefficients(atmosphere,wave,db,ref,photo,unity,
        elements=(element,),levels_per_charge=selected,include_explicit_kramers=True)
    np.testing.assert_allclose(j,a*b,rtol=3e-14,atol=1e-100)


def test_iron_group_lte_opacity_obeys_kirchhoff():
    from wd_spectra.light_metal_nlte import hot_metal_line_nlte_coefficients
    directory=Path(__file__).resolve().parents[1]/'results/hot-daz/multimetal-data'
    if not (directory/'gfFUV99.dat.gz').exists():pytest.skip('downloaded research atomic data not present')
    db,photo,_=load_composition_data(ModelData.default(),directory)
    atmosphere=gray_hydrogen_atmosphere(52500.,7.53,n_depth=3,tau_max=2.)
    ref=fixed_electron_metal_reference(atmosphere,db,{'Fe':np.log10(5e-6),'Ni':np.log10(1.01e-6)})
    unity={(e,s.charge):np.ones(atmosphere.n_depth) for e in ('Fe','Ni') for s in db.ion_stages(e)}
    wave=np.unique(np.r_[np.linspace(1306.3,1306.9,80),np.geomspace(30.,900.,60)])
    a,j=hot_metal_line_nlte_coefficients(atmosphere,wave,db,ref,unity,elements=('Fe','Ni'),
        include_static_linear_stark=False,maximum_lines=None)
    b,k=light_metal_bound_free_nlte_coefficients(atmosphere,wave,db,ref,photo,unity,elements=('Fe','Ni'))
    planck=planck_lambda_angstrom(wave[:,None],atmosphere.temperature)
    assert np.max(a)>0 and np.max(b)>0
    np.testing.assert_allclose(j,a*planck,rtol=1e-12,atol=1e-100)
    np.testing.assert_allclose(k,b*planck,rtol=1e-12,atol=1e-100)
    for element in ('Fe','Ni'):
        np.testing.assert_allclose(ref.ion_number_density[element].sum(axis=0),
                                   ref.element_number_density[element],rtol=1e-14)


def test_population_restart_restores_all_saved_elements(tmp_path):
    from explore_hot_trace import load_guesses
    saved={}
    for element in ('C','Si','N','O','Al','P','S','Ni'):
        saved[element+'_level_key']=np.array([(element,3,1),(element,4,1)])
        saved[element+'_population_density']=np.array([[2.,4.],[6.,8.]])
        saved[element+'_lte_population_density']=np.ones((2,2))
    np.savez(tmp_path/'populations.npz',**saved)
    result=load_guesses(tmp_path)
    assert set(result)=={'C','Si','N','O','Al','P','S','Ni'}
    for element,state in result.items():
        np.testing.assert_array_equal(state.population_density,saved[element+'_population_density'])
        assert state.level_key==((element,3,1),(element,4,1))
        np.testing.assert_array_equal(state.level_departure_coefficient[element,3,1],[2.,4.])
    assert set(load_guesses(tmp_path,elements=('C','O')))=={'C','O'}


def test_promoted_nickel_lines_and_lte_remainder_recover_full_lte():
    from wd_spectra.light_metal_nlte import hot_metal_line_nlte_coefficients
    from wd_spectra.hot_trace_metals import _atom_selection
    directory=Path(__file__).resolve().parents[1]/'results/hot-daz/multimetal-data'
    if not (directory/'gfFUV99.dat.gz').exists():pytest.skip('downloaded research atomic data not present')
    db,photo,_=load_composition_data(ModelData.default(),directory)
    atmosphere=gray_hydrogen_atmosphere(52500.,7.53,n_depth=3,tau_max=2.)
    ref=fixed_electron_metal_reference(atmosphere,db,{'Ni':np.log10(1.01e-6)})
    unity={('Ni',s.charge):np.ones(atmosphere.n_depth) for s in db.ion_stages('Ni')}
    wave=np.unique(np.r_[np.linspace(1306.3,1306.9,80),np.linspace(1402.3,1402.9,80)])
    closed,_=_atom_selection(db,'Ni',{3:140,4:140,5:1})
    all_keys={('Ni',s.charge,l.lower_index,l.upper_index) for s in db.ion_stages('Ni') for l in s.transitions}
    def opacity(keys):
        return hot_metal_line_nlte_coefficients(atmosphere,wave,db,ref,unity,elements=('Ni',),
            minimum_oscillator_strength=0.,maximum_lines=None,include_static_linear_stark=False,
            transition_keys=keys)
    whole=opacity(None);explicit=opacity(closed);remainder=opacity(all_keys-closed)
    assert np.max(explicit[0])>0 and np.max(remainder[0])>0
    for total,a,b in zip(whole,explicit,remainder):
        np.testing.assert_allclose(total,a+b,rtol=2e-13,atol=1e-100)
