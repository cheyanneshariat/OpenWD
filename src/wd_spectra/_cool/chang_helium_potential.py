"""Chang (2002), Texas Tech dissertation, chapter 2, Tables 2.2--2.3.

Public source: https://hdl.handle.net/2346/9503
QCISD(T)/aug-cc-pVTZ He2+/He3+ ground surfaces. Atomic units internally.

The printed switching signs in Eqs 2.20/2.26 contradict their stated
short/long-range limits. We implement the stated limits explicitly: the
negative polarization term turns ON at long range; the three-body polynomial
turns OFF below r1=1.3 bohr. This interpretation must reproduce the independently
reported minimum and asymptote before use. It is not a claim to have recovered
the author's source code or the different Scifoni potential used by Kowalski.
"""
import numpy as np
from scipy.special import expit
from numpy.polynomial.legendre import leggauss

BOHR_ANGSTROM = .529177210903
HARTREE_EV = 27.211386245988

# l m n, coefficient in atomic units. Visually checked against pp.20--22.
_DATA = '''
000 20.580389898270120
100 3.3861066903624448
010 -7.9943258643728399
200 -4.6941338498776695
020 -13.845234141521200
002 -12.806671336152224
110 9.1275467161043551
300 -24.047706629139096
030 6.4136370342346227
210 59.645413237976832
120 -31.692933449801366
102 -107.36783927197288
012 43.961267940365154
400 4.6105740503585331
040 12.993810789492178
004 -85.799336921764976
310 -62.709166084469459
130 -62.023587594996904
220 121.43238341775377
202 -181.70353724580170
022 -42.687642218424728
112 181.21452315281556
500 44.986495181718858
050 -1.4632618989391608
410 -129.79657974579266
140 -2.6218228104219152
320 104.27116736590155
230 -15.590768096729247
104 -253.81157013376313
014 -8.0481332879474969
302 2.0619950447782678
032 -69.954481695916428
212 -258.30635201350890
122 292.54798699523349
600 -22.009966761511766
060 -6.4526321335076542
006 28.900496961610422
510 123.47282118635520
150 35.060367822277961
420 -231.56502112023102
240 -104.32449847999422
330 200.51747177883132
204 -37.228018295356243
024 155.91039464997405
402 120.76680953896120
042 3.7633658022015992
114 -512.46265382838681
312 -318.48063673593185
132 -9.1608016243364094
222 112.27558428822887
700 -14.344153639782899
070 -.88476852006011264
610 43.983047115855939
160 6.1750433469225392
520 -51.105539123737188
250 -2.2155179793769122
430 39.794975607010159
340 -21.721455691917058
106 125.99772281931540
016 -111.98891885277750
304 108.74387343331379
034 19.367637437735887
502 -111.41260047161114
052 29.886912845163291
214 -422.30820031112859
124 -58.811934521159202
412 463.17437851766812
142 -171.21119235912255
322 -657.68990091654769
232 365.19778226040938
800 7.7120178270526587
080 .60488283796720099
008 3.2264917382345661
710 -39.670060969538447
170 -4.6993524380497522
620 91.555394951494151
260 26.891664085796119
530 -131.71451320262616
350 -77.673723493402761
440 127.61137616046707
206 72.014231584358484
026 -43.134987007309668
404 .061527573372508713
044 -15.682767197863269
602 -85.687991643409390
062 -7.3712242416679317
116 -43.938720138488073
314 18.679424438038897
134 32.416999349093821
224 -159.92783975498514
512 308.23434313803085
152 64.020268802737633
422 -440.04512030446580
242 -256.28447811007652
332 397.32998367299132
'''
COEFFICIENTS = [(tuple(map(int,index)), float(value))
                for index,value in (line.split() for line in _DATA.strip().splitlines())]


def dimer_ground_hartree(radius):
    r = np.asarray(radius,float)
    if np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError('pair distances must be positive and finite')
    x = 1.09314253*(r-2.04725623)
    morse = .09027661*(np.exp(-2*x)*(1+.00957858*x**3*(1+1.6059377*x))-2*np.exp(-x))
    short = expit(-2*.77802774*(r-6.15941548))
    long = expit(2*2.03590553*(r-6.65881373))
    return morse*short-1.3793/(2*r**4)*long


def trimer_ground_hartree(distances):
    """Full surface, relative to He+ + He + He; permutation-invariant."""
    r = np.asarray(distances,float)
    if r.shape[0] != 3 or np.any(~np.isfinite(r)) or np.any(r <= 0):
        raise ValueError('three finite positive distances are required')
    r1,r2,r3 = np.sort(r,axis=0)
    if np.any(r1+r2 < r3*(1-1e-14)):
        raise ValueError('three distances cannot form a triangle')
    radius = np.sqrt((r2*r2+r3*r3)/2-r1*r1/4)
    cosine = (r3*r3-r2*r2)/(2*r1*radius)
    x,y = r1-2.33974,radius-3.50961
    polynomial = np.zeros_like(radius)
    for (l,m,n),coefficient in COEFFICIENTS:
        polynomial += coefficient*x**l*y**m*cosine**n
    short = expit(2*10000*(r1-1.3))
    radial = expit(-2*1.22*radius)
    polarization = -.25*1.3793*(expit(2*2.2*(r2-2.5))/r2**4
                              +expit(2*2.2*(r3-2.5))/r3**4)
    return dimer_ground_hartree(r1)+polynomial*short*radial+polarization


def molecular_ion_neutral_pair_ev(radius, *, bond_bohr=2.046179, n_angle=32):
    """Spherical potential average at fixed isolated-dimer bond length.

    Average V, not exp(-V/kT), following the explicit spherical-solute
    approximation. Rotor/anisotropic solvent responses are not included.
    """
    distance=np.asarray(radius,float)/BOHR_ANGSTROM
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0):
        raise ValueError('separation must be positive and finite')
    if not np.isfinite(bond_bohr) or bond_bohr <= 1.3 or n_angle < 8:
        raise ValueError('invalid dimer bond or angular resolution')
    # Symmetry permits cos(theta) in [0,1]. Split where sorted pair labels
    # change and across the published extremely narrow r1=1.3 switch. A
    # single global quadrature oscillates with order at these features.
    # The transition boundaries are numerical integration subdivisions,
    # not alterations of the potential or of its printed sharpness.
    denominator=distance*bond_bohr
    center=distance**2+bond_bohr**2/4
    breaks=[np.zeros_like(distance),np.ones_like(distance),
            (center-bond_bohr**2)/denominator,
            (bond_bohr**2-center)/denominator]
    for pair_distance in (1.3-.001,1.3,1.3+.001):
        breaks.append((center-pair_distance**2)/denominator)
    edges=np.sort(np.clip(np.stack(breaks,axis=-1),0,1),axis=-1)
    width=np.diff(edges,axis=-1)
    x,w=leggauss(n_angle)
    x=edges[...,:-1,None]+.5*width[...,None]*(x+1)
    weights=.5*width[...,None]*w
    r=distance[...,None,None]
    a=np.sqrt(r*r+bond_bohr**2/4-r*bond_bohr*x)
    b=np.sqrt(r*r+bond_bohr**2/4+r*bond_bohr*x)
    pairs=np.stack((np.full_like(a,bond_bohr),a,b))
    interaction=trimer_ground_hartree(pairs)-dimer_ground_hartree(bond_bohr)
    return HARTREE_EV*np.sum(weights*interaction,axis=(-2,-1))
