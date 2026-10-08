"""Conservative research transfer for refracting, plane-parallel atmospheres.

This is a finite-volume discretization of Kowalski & Saumon (2004), not a
claim to reproduce their numerical code. I'=I/n^2; chi=chi0/n; p=n sin(theta)
is invariant. A fixed bundle of rays is integrated with dq/2, q=p^2, so the
same ray weights carry flux across every crossed physical mass face.

Each material cell has constant opacity/source and linearly varying n^2.
Path optical depths are exact for that reconstruction, including turns.
Scattering couples all rays through the path-volume-averaged reduced J.
Consequently the computed (not repaired) flux divergence equals the direct
cell heating, including scattering cancellation, to rounding precision.

The last supplied node sets the constant source of an LTE reservoir with
outgoing I'=B, not the optically thick approximation P'=B. Rays turning
inside its finite buffer are propagated exactly. The top face is at zero
mass with n=1. No composition module or production solver imports this.
"""
import numpy as np
from numba import njit
from .dq_refractive_ray_bundles import bundles

REFERENCE='https://doi.org/10.1086/386280'


@njit(cache=True)
def geometry(mass,index):
    size=len(mass)
    faces=np.empty(size);n2=np.empty(size)
    faces[0]=0.;n2[0]=1.
    for i in range(1,size):
        faces[i]=.5*(mass[i-1]+mass[i])
        n2[i]=.5*(index[i-1]**2+index[i]**2)
    return faces,n2


@njit(cache=True)
def ray_cells(mass,extinction,index,faces,n2,q,first):
    """Cell path lengths and face conductances for one invariant ray."""
    count=len(mass)-1-first
    depth=np.empty(count)
    for j in range(count):
        i=first+j
        lo=max(n2[i],q);hi=n2[i+1]
        length=faces[i+1]-faces[i]
        if q>n2[i]:length*=(hi-q)/(hi-n2[i])
        depth[j]=2*extinction[i]*length/(np.sqrt(lo-q)+np.sqrt(hi-q))
    conductance=np.empty(count+1)
    conductance[0]=0. if q>=1. else 1/(1+.5*depth[0])
    for j in range(1,count):
        conductance[j]=2/(depth[j-1]+depth[j])
    # Optical resistance from the last cell face to the thermal reservoir.
    y0=np.sqrt(n2[-1]-q);y1=np.sqrt(index[-1]**2-q)
    buffer=2*extinction[-1]*(mass[-1]-faces[-1])/(y0+y1)
    conductance[-1]=1/(1.+.5*depth[-1])
    return depth,conductance


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
        for j in range(ncol):
            reduced[i,j]+=ratio*reduced[i-1,j]
    value=np.empty_like(reduced);value[-1]=reduced[-1]
    face_flux=np.empty((count+1,ncol))
    for i in range(count-1,-1,-1):
        for j in range(ncol):
            jump=(g[i]*value[i+1,j]-reduced[i,j])/p[i]
            value[i,j]=value[i+1,j]-jump
            face_flux[i+1,j]=conductance[i+1]*jump
    for j in range(ncol):
        face_flux[0,j]=conductance[0]*value[0,j]
    return value[:-1],face_flux


@njit(cache=False)
def operators(mass,extinction,index,angles,weights):
    """Reduced-J and physical-flux response to cell sources and bottom B.

    Do not persist this caller: Numba does not invalidate its disk cache
    when the imported ray-bundle implementation changes.
    """
    size=len(mass);cells=size-1
    mean=np.zeros((cells,size));flux=np.zeros((size,size));volume=np.zeros(cells)
    zero=np.zeros(size)
    for first,last,depth,conductance,dd,dg,weight,dweight in bundles(
            mass,extinction,index,zero,zero,angles,weights,False):
        count=last-first+1
        rhs=np.zeros((count+1,count+1))
        for j in range(count):rhs[j,j]=depth[j]
        rhs[-1,-1]=1.
        value,ray_flux=ray_solve(depth,conductance,rhs)
        for j in range(count):
            volume[first+j]+=weight*depth[j]
            mean[first+j,first:last+1]+=weight*depth[j]*value[j,:-1]
            mean[first+j,-1]+=weight*depth[j]*value[j,-1]
        flux[first:last+2,first:last+1]+=4*np.pi*weight*ray_flux[:,:-1]
        flux[first:last+2,-1]+=4*np.pi*weight*ray_flux[:,-1]
    for j in range(cells):mean[j]/=volume[j]
    return mean,flux,volume


@njit(cache=False)
def prescribed_operators(mass,extinction,index,source,angles,weights):
    """Independent direct source sweep, without constructing a Lambda matrix."""
    size=len(mass);cells=size-1
    mean=np.zeros(cells);flux=np.zeros(size);volume=np.zeros(cells);zero=np.zeros(size)
    for first,last,depth,g,dd,dg,weight,dweight in bundles(
            mass,extinction,index,zero,zero,angles,weights,False):
        count=last-first+1
        rhs=np.zeros((count+1,1));rhs[:-1,0]=depth*source[first:last+1]
        rhs[-1,0]=source[-1]
        value,ray_flux=ray_solve(depth,g,rhs)
        for j in range(count):
            volume[first+j]+=weight*depth[j]
            mean[first+j]+=weight*depth[j]*value[j,0]
        flux[first:last+2]+=4*np.pi*weight*ray_flux[:,0]
    return mean/volume,flux,volume


def formal_field(mass,extinction,source,index,*,n_angle=4):
    mass,(k,_,b,n),angles,weights=validate(mass,extinction,0.,source,index,n_angle)
    mean=np.empty_like(b);flux=np.empty_like(b);volume=np.empty_like(b[:,:-1])
    for iw in range(len(b)):
        j,f,v=prescribed_operators(mass,k[iw],n[iw],b[iw],angles,weights)
        mean[iw]=np.r_[j,b[iw,-1]];flux[iw]=f;volume[iw]=v
    energy=4*np.pi*volume*(mean[:,:-1]-b[:,:-1])
    scale=4*np.pi*volume*b[:,:-1]+abs(flux[:,:-1])+abs(flux[:,1:])
    return dict(reduced_mean_intensity=mean,flux=flux,
        maximum_conservation_error=float(np.max(abs(np.diff(flux,axis=1)-energy)/np.maximum(scale,np.finfo(float).tiny))))


def validate(mass,absorption0,scattering0,planck,index,n_angle):
    mass=np.asarray(mass,float)
    values=np.broadcast_arrays(*[np.asarray(x,float) for x in (absorption0,scattering0,planck,index)])
    a,s,b,n=[np.array(x,order='C',copy=True) for x in values]
    if (mass.ndim!=1 or len(mass)<3 or np.any(mass<=0) or np.any(np.diff(mass)<=0)
        or a.ndim!=2 or a.shape[1]!=len(mass)
        or isinstance(n_angle,(bool,np.bool_)) or not isinstance(n_angle,(int,np.integer)) or n_angle<2
        or any(np.any(~np.isfinite(x)) for x in (mass,a,s,b,n))
        or np.any(a<0) or np.any(s<0) or np.any(a+s<=0) or np.any(b<0)
        or np.any(n<1)):
        raise ValueError('Positive ordered mass, nonnegative material fields and finite n>=1 required')
    nodes,weights=np.polynomial.legendre.leggauss(n_angle)
    return mass,(a,s,b,n),(nodes+1)/2,weights/2


def solve(mass,absorption0,scattering0,planck,refractive_index,*,n_angle=4):
    mass,(a,s,b,n),angles,weights=validate(mass,absorption0,scattering0,planck,refractive_index,n_angle)
    mean=np.empty_like(b);flux=np.empty_like(b);volume=np.empty_like(b[:,:-1])
    closure=0.;boundary_flux=np.empty(len(b))
    for iw in range(len(b)):
        lam,fl,v=operators(mass,a[iw]+s[iw],n[iw],angles,weights)
        eps=a[iw,:-1]/(a[iw,:-1]+s[iw,:-1])
        matrix=np.eye(len(mass)-1)-lam[:,:-1]*(1-eps)[None,:]
        j=np.linalg.solve(matrix,lam[:,:-1]@(eps*b[iw,:-1])+lam[:,-1]*b[iw,-1])
        source=np.r_[eps*b[iw,:-1]+(1-eps)*j,b[iw,-1]]
        boundary_mean=np.linalg.solve(matrix,lam[:,-1])
        boundary_flux[iw]=(fl[0,-1]+fl[0,:-1]@((1-eps)*boundary_mean))*b[iw,-1]
        mean[iw]=np.r_[j,b[iw,-1]]
        flux[iw]=fl@source;volume[iw]=v
        scale=np.maximum(np.maximum(abs(j),b[iw,:-1]),np.finfo(float).tiny)
        closure=max(closure,float(np.max(abs(j-lam@source)/scale)))
    eps=a[:,:-1]/(a[:,:-1]+s[:,:-1])
    emission=4*np.pi*volume*eps*b[:,:-1]
    heating=4*np.pi*volume*eps*(mean[:,:-1]-b[:,:-1])
    defect=np.diff(flux,axis=1)-heating
    scale=np.maximum(emission+abs(flux[:,:-1])+abs(flux[:,1:]),np.finfo(float).tiny)
    return dict(reduced_mean_intensity=mean,flux=flux,cell_heating=heating,
        cell_thermal_emission=emission,path_volume=volume,
        boundary_surface_flux=boundary_flux,
        maximum_source_error=closure,maximum_conservation_error=float(np.max(abs(defect)/scale)),
        reference=REFERENCE,atmosphere_reconverged=False)
