"""Explicit research join of the existing He-minus free-free prescriptions.

John (1994), MNRAS 269, 871, Table 2 starts at 0.5063 micron. Its
conclusions distinguish the less certain optical interval from the better
determined lambda > 1 micron infrared. Neither limit is a physical edge.
Blend in that overlap, without extrapolating the cool table or rescaling
either dataset. The blend is our numerical prescription, not John's fit.
The historical automatic temperature limits and explicit source modes are
unchanged. This does not repair their separate temperature-boundary joins.
"""
from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch
import numpy as np
from wd_spectra import helium

HARD_JOIN='historical-hard-domain-switch'
SMOOTH_JOIN='cubic-log-wavelength-overlap-0.5063-to-1-micron-v1'
METADATA_KEY='experimental_heminus_wavelength_join'


def coefficient_with_smooth_join(original,wavelength_micron,temperature,*,prescription='automatic'):
    # Let the original function perform validation and retain explicit source
    # prescriptions bit for bit, including their documented extrapolations.
    result=original(wavelength_micron,temperature,prescription=prescription)
    if prescription!='automatic':return result
    wave,temp=np.broadcast_arrays(np.asarray(wavelength_micron,float),np.asarray(temperature,float))
    lower=float(helium._JOHN_WAVELENGTH_UM[0]);upper=1.
    selected=((wave>=lower)&(wave<upper)&(temp>=5040./helium._JOHN_THETA[-1])
              &(temp<=5040./helium._JOHN_THETA[0]))
    if not np.any(selected):return result
    old=original(wave,temp,prescription='john1968')
    coordinate=np.clip(np.log(wave/lower)/np.log(upper/lower),0.,1.)
    weight=coordinate**2*(3.-2.*coordinate)
    return np.where(selected,(1.-weight)*old+weight*result,result)


def active_policy():
    return getattr(helium.helium_minus_free_free_coefficient,'experimental_join_policy',HARD_JOIN)


def validate_continuation_policy(source,target):
    if source not in (HARD_JOIN,SMOOTH_JOIN) or target not in (HARD_JOIN,SMOOTH_JOIN):
        raise ValueError('unknown source or target opacity join')
    if source==SMOOTH_JOIN and target==HARD_JOIN:
        raise ValueError('resuming a smooth-opacity run requires --smooth-heminus-join; no silent opacity reversion')
    return source!=target


@contextmanager
def heminus_join_scope(policy=SMOOTH_JOIN):
    if policy not in (HARD_JOIN,SMOOTH_JOIN):raise ValueError('unknown He-minus wavelength join policy')
    if policy==HARD_JOIN:
        if active_policy()!=HARD_JOIN:raise ValueError('cannot silently restore historical opacity inside a smooth scope')
        yield
        return
    if active_policy()!=HARD_JOIN:raise ValueError('He-minus join scope must not be nested')
    original=helium.helium_minus_free_free_coefficient
    @wraps(original)
    def joined(*args,**kwargs):return coefficient_with_smooth_join(original,*args,**kwargs)
    joined.experimental_join_policy=SMOOTH_JOIN
    with patch.object(helium,'helium_minus_free_free_coefficient',joined):yield
