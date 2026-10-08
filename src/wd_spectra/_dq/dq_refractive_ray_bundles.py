"""Invariant-ray cells for arbitrary positive, piecewise-linear n^2 profiles.

Separate disconnected allowed regions at each q=p^2. Rays can turn at the
upper OR lower edge of a region, including trapped interior ray families.
The opacity/source reconstruction is the same as the monotonic-volume
experiment. Analytic derivatives include moving turns and quadrature nodes.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def face_geometry(mass,index):
    size=len(mass);faces=np.empty(size);n2=np.empty(size)
    faces[0]=0.;n2[0]=1.
    for i in range(1,size):
        faces[i]=.5*(mass[i-1]+mass[i])
        n2[i]=.5*(index[i-1]**2+index[i]**2)
    return faces,n2


@njit(cache=True)
def segment_depth_value(length,k,left,right,q):
    if left<q:return 2*k*length*np.sqrt(right-q)/(right-left)
    if right<q:return 2*k*length*np.sqrt(left-q)/(left-right)
    return 2*k*length/(np.sqrt(left-q)+np.sqrt(right-q))


@njit(cache=True)
def segment_depth(length,k,dk,column,left,right,dleft,dright,q,dq):
    if left<q:
        depth=2*k*length*np.sqrt(right-q)/(right-left)
        derivative=depth*((dright-dq)/(2*(right-q))-(dright-dleft)/(right-left))
    elif right<q:
        depth=2*k*length*np.sqrt(left-q)/(left-right)
        derivative=depth*((dleft-dq)/(2*(left-q))-(dleft-dright)/(left-right))
    else:
        y0=np.sqrt(left-q);y1=np.sqrt(right-q)
        depth=2*k*length/(y0+y1)
        derivative=-depth*((dleft-dq)/(2*y0)+(dright-dq)/(2*y1))/(y0+y1)
    if column>=0:derivative[column]+=depth*dk/k
    return depth,derivative


@njit(cache=True)
def bundles(mass,k,index,dk,dindex,angles,weights,need_derivative):
    # A typed list (bounded to one wavelength) avoids Numba 0.58's cached
    # cross-module generator-lowering failure on a fresh Python process.
    result=[]
    size=len(mass);cells=size-1;cols=size if need_derivative else 1
    faces,n2=face_geometry(mass,index)
    levels=np.empty(size+1);levels[:-1]=n2;levels[-1]=index[-1]**2
    dl=np.zeros((size+1,cols))
    if need_derivative:
        for i in range(1,size):
            dl[i,i-1]=index[i-1]*dindex[i-1]
            dl[i,i]=index[i]*dindex[i]
        dl[-1,-1]=2*index[-1]*dindex[-1]
    order=np.argsort(levels)
    zero=np.zeros(cols)
    for region in range(size+1):
        low=0. if region==0 else levels[order[region-1]]
        high=levels[order[region]]
        if high<=low:continue
        dlow=zero if region==0 else dl[order[region-1]]
        dhigh=dl[order[region]]
        for t,w in zip(angles,weights):
            q=high-(high-low)*t*t;dq=dhigh-(dhigh-dlow)*t*t
            weight=(high-low)*t*w;dweight=(dhigh-dlow)*t*w
            first=0
            while first<cells:
                if max(n2[first],n2[first+1])<=q:
                    first+=1
                    continue
                last=first
                while last+1<cells and n2[last+1]>q:last+=1
                count=last-first+1
                depth=np.empty(count);dd=np.empty((count,cols)) if need_derivative else np.empty((0,0))
                for j in range(count):
                    i=first+j
                    if need_derivative:
                        depth[j],dd[j]=segment_depth(faces[i+1]-faces[i],k[i],dk[i],i,
                            n2[i],n2[i+1],dl[i],dl[i+1],q,dq)
                    else:
                        depth[j]=segment_depth_value(faces[i+1]-faces[i],k[i],n2[i],n2[i+1],q)
                g=np.empty(count+1);dg=np.empty((count+1,cols)) if need_derivative else np.empty((0,0))
                g[0]=1/(1+.5*depth[0]) if first==0 and q<1 else 0.
                if need_derivative:dg[0]=-.5*g[0]**2*dd[0]
                for j in range(1,count):
                    g[j]=2/(depth[j-1]+depth[j])
                    if need_derivative:dg[j]=-.5*g[j]**2*(dd[j-1]+dd[j])
                if last<cells-1 or n2[-1]<=q:
                    g[-1]=0.
                    if need_derivative:dg[-1]=zero
                else:
                    if need_derivative:
                        buffer,dbuffer=segment_depth(mass[-1]-faces[-1],k[-1],dk[-1],
                            size-1,n2[-1],levels[-1],dl[-2],dl[-1],q,dq)
                    else:
                        buffer=segment_depth_value(mass[-1]-faces[-1],k[-1],n2[-1],levels[-1],q)
                        dbuffer=zero
                    if levels[-1]>q:
                        # I'_out=B in the LTE reservoir, not P'=B.
                        # The latter is only valid at large optical depth;
                        # in a transparent slab it injects twice pi*B.
                        # Constant-source LTE propagation through the buffer
                        # leaves I'_out=B, so the face condition is P'+v=B.
                        resistance=1.+.5*depth[-1]
                        if need_derivative:derivative=.5*dd[-1]
                    else:
                        # A turn inside the constant-B reservoir returns
                        # I'_out=B+(I'_in-B)*exp(-2*buffer). Eliminate that
                        # exact finite LTE segment: P'+coth(buffer)*v=B.
                        tangent=np.tanh(buffer)
                        resistance=.5*depth[-1]+1/tangent
                        if need_derivative:
                            derivative=.5*dd[-1]-(1-tangent*tangent)/tangent**2*dbuffer
                    g[-1]=1/resistance
                    if need_derivative:dg[-1]=-g[-1]**2*derivative
                result.append((first,last,depth,g,dd,dg,weight,dweight))
                first=last+1
    return result
