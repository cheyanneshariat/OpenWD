"""Frozen pre-optimization algebra, solely for bitwise kernel regressions."""

import numpy as np

from numba import njit

from wd_spectra._dq.dq_refractive_ray_bundles import bundles

@njit(cache=True)
def ray_solve(depth,conductance,rhs):
    """Stable difference elimination with a supplied Dirichlet last row.

    rhs has one row per crossed material cell plus the bottom reservoir.
    Return material intensities and outward face flux divided by 4*pi*dq/2.
    """
    count=len(depth);ncol=rhs.shape[1]
    reduced=rhs.copy();g=np.empty(count);p=np.empty(count)
    g[0]=depth[0]+conductance[0];p[0]=g[0]+conductance[1]
    for i in range(1,count):
        ratio=conductance[i]/p[i-1]
        g[i]=depth[i]+ratio*g[i-1]
        p[i]=g[i]+conductance[i+1]
        reduced[i]+=ratio*reduced[i-1]
    value=np.empty_like(reduced);value[-1]=reduced[-1]
    face_flux=np.empty((count+1,ncol))
    for i in range(count-1,-1,-1):
        jump=(g[i]*value[i+1]-reduced[i])/p[i]
        value[i]=value[i+1]-jump
        face_flux[i+1]=conductance[i+1]*jump
    face_flux[0]=conductance[0]*value[0]
    return value[:-1],face_flux

@njit(cache=False)
def known_response(mass,k,index,dk,dindex,source,mean,direct,dbottom,angles,weights):
    """Tangent forcing excluding the common scattering feedback through dJ.

    Compile against the current imported geometry; do not reuse a disk
    cache that cannot track edits to that separate research module.
    """
    size=len(mass);cells=size-1
    dj=np.zeros((cells,size));df=np.zeros((size,size));dv=np.zeros((cells,size))
    for first,last,depth,g,dd,dg,weight,dweight in bundles(
            mass,k,index,dk,dindex,angles,weights,True):
        count=last-first+1
        base_rhs=np.zeros((count+1,1));base_rhs[:-1,0]=depth*source[first:last+1]
        base_rhs[-1,0]=source[-1]
        p,v=ray_solve(depth,g,base_rhs)
        rhs=np.zeros((count+1,size))
        for j in range(count):
            i=first+j
            rhs[j]=dd[j]*(source[i]-p[j,0])
            rhs[j,i]+=depth[j]*direct[i]
        rhs[0]-=dg[0]*p[0,0]
        for j in range(1,count):
            jump=v[j,0]/g[j]
            rhs[j-1]+=dg[j]*jump
            rhs[j]-=dg[j]*jump
        if g[-1]>0:rhs[-2]+=dg[-1]*(v[-1,0]/g[-1])
        rhs[-1,-1]=dbottom
        dp,dflux=ray_solve(depth,g,rhs)
        dflux[0]+=dg[0]*p[0,0]
        for j in range(1,count+1):
            if g[j]>0:dflux[j]+=dg[j]*(v[j,0]/g[j])
        for j in range(count):
            i=first+j
            volume_derivative=dweight*depth[j]+weight*dd[j]
            dv[i]+=volume_derivative
            dj[i]+=weight*depth[j]*dp[j]+volume_derivative*(p[j,0]-mean[i])
        df[first:last+2]+=4*np.pi*(weight*dflux+dweight[None,:]*v)
    return dj,df,dv
