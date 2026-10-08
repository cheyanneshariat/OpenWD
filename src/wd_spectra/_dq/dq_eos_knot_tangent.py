"""DQ-local analytic thermal responses; no nested EOS finite differences."""
import numpy as np
from wd_spectra._material_response import temperature_response_probes
from .dq_reos_secant_response import thermodynamic_response


def eos_probe_policy(material,evaluate,step,*,centered):
    if not centered:return temperature_response_probes(evaluate,step,centered=False)
    # Only opacity/chemistry/radiation use these small centered probes.
    # cp, Q, rho and nabla_ad responses below differentiate the table itself.
    h=min(abs(step),np.cbrt(np.finfo(float).eps))
    material.last_eos_derivative_step=h
    result=temperature_response_probes(evaluate,h,centered=True)
    material.eos_response_probes=result
    return result


def corrected_evaluation(owner, materials, ev):
    """Replace thermal tangents, retaining every physical value and equation."""
    from wd_spectra.nonlinear import NonlinearEvaluation
    from wd_spectra.adaptive_structure import _ml2_flux_coefficient_response
    from wd_spectra.constants import STEFAN_BOLTZMANN
    p=ev.payload
    if p.get('analytic_reos_thermal_response'):
        return ev
    if not (p.get('energy_balance_is_physical_flux') and p.get('local_energy_equations_enforced')):
        raise ValueError('Analytic DQ EOS responses require physical flux and energy equations')
    a=p['atmosphere'];transport=p['convection_transport']
    high,low,h=owner.eos_response_probes
    if low is None:
        raise ValueError('DQ analytic EOS response requires centered material probes')
    hot,cold=high[0],low[0]
    # Retain the exact objects/rounding used by the physical opacity probes.
    # Rebuilding them as exp(log(T)+h) causes unnecessary opacity cache misses.
    np.testing.assert_array_equal(hot.temperature,a.temperature*np.exp(h))
    np.testing.assert_array_equal(cold.temperature,a.temperature*np.exp(-h))
    np.testing.assert_array_equal(hot.gas_pressure,a.gas_pressure)
    np.testing.assert_array_equal(cold.gas_pressure,a.gas_pressure)
    fields=np.asarray([a.mass_density,transport['rosseland'],transport['specific_heat'],
        transport['expansion'],transport['node_adiabatic_gradient']])
    slopes=np.zeros_like(fields)
    slopes[1]=(materials.rosseland_opacity(hot)-materials.rosseland_opacity(cold))/(2*h)
    hp=np.asarray(a.metadata['dq_helium_host_pressure'])
    q=(np.log(hot.metadata['dq_helium_host_pressure'])-
       np.log(cold.metadata['dq_helium_host_pressure']))/(2*h)
    response=thermodynamic_response(owner.helium_reos3,a.temperature,hp,q)
    # Mixture mass per He nucleus is fixed; chemistry changes the host
    # pressure, included through q. Thermodynamics uses that same host law.
    response[0] *= a.mass_density/owner.helium_reos3.evaluate(hp,a.temperature)[0]
    slopes[[0,2,3,4]]=response
    materials.index_response_step=h
    values,tangent=materials.assemble(a,fields,slopes)
    expected=tuple(p['convection_transport'][k] for k in
        ('adiabatic_gradient','ml2_radiative_loss','ml2_flux_coefficient'))
    for new,old in zip(values,expected):
        np.testing.assert_allclose(new,old,rtol=3e-12,atol=0)
    fcj=transport['actual_convective_flux_gradient_derivative'][:,None]*p['interface_gradient_operator']
    fcj=fcj+_ml2_flux_coefficient_response(p['temperature_gradient'],expected,tangent)
    cellj=p['radiative_cell_energy_log_temperature_jacobian']+np.diff(fcj,axis=0)
    scalej=p['thermal_cell_emission_log_temperature_jacobian']+fcj[:-1]+fcj[1:]
    target=STEFAN_BOLTZMANN*a.effective_temperature**4
    jac=(p['radiative_flux_log_temperature_jacobian']+fcj)/target
    scale=p['cell_energy_scale'];defect=p['cell_energy_balance_defect_in_stellar_flux']*target
    jac[:-1]-=cellj/scale[:,None]
    jac[:-1]+=(defect/scale**2)[:,None]*scalej
    payload={**p,'ml2_coefficient_log_temperature_responses':tangent,
        'convective_flux_log_temperature_jacobian':fcj,
        'cell_energy_log_temperature_jacobian':cellj,
        'cell_energy_scale_log_temperature_jacobian':scalej,
        'analytic_reos_thermal_response':True}
    return NonlinearEvaluation(ev.residual,jac@p['log_temperature_from_state'],payload)
