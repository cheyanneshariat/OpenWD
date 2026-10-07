import numpy as np
from wd_spectra.stark import default_lyman_stark_table
from stark_density_edge_experiment import consistent_stark_density_edge


def test_density_clipping_context_matches_default_and_is_isolated():
    line=default_lyman_stark_table()[(1,2)]
    w=1215.6713+np.linspace(-.2,.2,2001)
    low=10.**line.log_electron_density[0]
    original=line.wavelength_profile(w,1215.6713,4000.,1e4)
    edge=line.wavelength_profile(w,1215.6713,4000.,low)
    interior=line.wavelength_profile(w,1215.6713,8000.,3e13)
    with consistent_stark_density_edge():
        np.testing.assert_array_equal(line.wavelength_profile(w,1215.6713,4000.,1e4),edge)
        np.testing.assert_array_equal(line.wavelength_profile(w,1215.6713,8000.,3e13),interior)
    np.testing.assert_array_equal(line.wavelength_profile(w,1215.6713,4000.,1e4),original)
    # The formerly experimental edge policy is now the public default.
    # The context must remain idempotent and restore the original method.
    np.testing.assert_array_equal(original,edge)


def test_density_edge_is_continuous_from_inside_and_outside():
    line=default_lyman_stark_table()[(1,2)]
    w=1215.6713+np.linspace(-.2,.2,2001)
    edge=10.**line.log_electron_density[0]
    with consistent_stark_density_edge():
        p=[line.wavelength_profile(w,1215.6713,4000.,edge*f) for f in (1-1e-8,1.,1+1e-8)]
    np.testing.assert_array_equal(p[0],p[1])
    np.testing.assert_allclose(p[2],p[1],rtol=1e-6,atol=1e-12)
