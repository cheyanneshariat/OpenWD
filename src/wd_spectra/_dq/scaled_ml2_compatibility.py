"""A local ML2 compatibility norm that remains meaningful through onset.

An initially stable layer must not retain its large stability-gap tolerance
after the auxiliary velocity becomes convective. The C1 scale below uses
the stellar-flux excess on the convective branch, and the current stable
gap on the negative-velocity branch. No physical equation is replaced.
"""
import numpy as np
from wd_spectra._ml2_auxiliary import ml2_auxiliary_compatibility,ml2_scaled_coefficients


def compatibility_normalization(y,coefficients,target,thermal_scale=None):
    """Positive local scale, also usable as a frozen Newton-step weight."""
    _,loss,coefficient=coefficients
    a,b=ml2_scaled_coefficients(loss,coefficient,target)
    if thermal_scale is None:
        v2=np.ones_like(y)
    else:
        t=np.asarray(thermal_scale)
        if t.shape!=y.shape or np.any(~np.isfinite(t)) or np.any(t<=0):
            raise ValueError('thermal scale must be positive, finite and match the interfaces')
        v2=np.cbrt(t/(1+t)+np.maximum(y,0.)**3)**2
    gap=v2+np.minimum(y,0.)**2
    return a*np.sqrt(gap)+b*gap


def scaled_compatibility(gradient,y,coefficients,responses,operator,target,*,
                         thermal_scale=None,thermal_scale_jacobian=None,normalization=None):
    defect,tangent,y_tangent=ml2_auxiliary_compatibility(
        gradient,y,coefficients,responses,operator,target)
    if normalization is not None:
        norm=np.asarray(normalization)
        if norm.shape!=defect.shape or np.any(~np.isfinite(norm)) or np.any(norm<=0):
            raise ValueError('frozen compatibility normalization must be positive finite and match the interfaces')
        return defect/norm,None if tangent is None else tangent/norm[:,None],y_tangent/norm
    ad,loss,coefficient=coefficients
    a,b=ml2_scaled_coefficients(loss,coefficient,target)
    negative=np.minimum(y,0.)
    if thermal_scale is None:
        velocity=np.ones_like(y)
        velocity_y=np.zeros_like(y)
        velocity_j=None
    else:
        thermal=np.asarray(thermal_scale)
        if thermal.shape!=y.shape or np.any(thermal<=0) or np.any(~np.isfinite(thermal)):
            raise ValueError('local compatibility scale requires positive finite interface emission')
        # Smoothly restrict the emission scale by Fstar: both local energy
        # and absolute stellar-flux conservation must remain meaningful.
        thermal=thermal/(1+thermal)
        positive=np.maximum(y,0.)
        velocity=np.cbrt(thermal+positive**3)
        velocity_y=positive**2/velocity**2
        if responses is not None:
            if thermal_scale_jacobian is None:
                raise ValueError('differentiated thermal norm requires its temperature response')
            velocity_j=(thermal_scale_jacobian/(1+np.asarray(thermal_scale))[:,None]**2
                /(3*velocity[:,None]**2))
    stable=velocity**2+negative**2
    root=np.sqrt(stable)
    scale=a*root+b*stable
    residual=defect/scale
    sy=(a/root+2*b)*(negative+velocity*velocity_y)
    y_j=(y_tangent-residual*sy)/scale
    if responses is None: return residual,None,y_j
    _,loss_j,coefficient_j=responses
    material_velocity=np.cbrt(target/coefficient)
    log_vj=-coefficient_j/(3*coefficient[:,None])
    aj=loss_j*material_velocity[:,None]+a[:,None]*log_vj
    bj=2*b[:,None]*log_vj
    scale_j=aj*root[:,None]+bj*stable[:,None]
    if thermal_scale is not None:
        scale_j+=(a/root+2*b)[:,None]*velocity[:,None]*velocity_j
    return residual,(tangent-residual[:,None]*scale_j)/scale[:,None],y_j
