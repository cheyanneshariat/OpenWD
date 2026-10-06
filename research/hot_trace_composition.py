"""Provenance-pinned atomic supplements for a nine-element G191-B2B screen.

This research adapter does not modify the runtime tables or the H/He host.
Fe/Ni remain LTE opacity sources; the five new light elements have explicit SE.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
from hot_daz_data import cache_directory, data_file
from wd_spectra.metals import (ATOMIC_NUMBER, AtomicDatabase,
    VernerPhotoionizationDatabase, read_pg1159_atomic_database,
    read_verner_photoionization_database, read_kurucz_gf100_atomic_database)

# Preval et al. 2013, Table 10. Each entry is an ELEMENTAL number abundance
# inferred from the named ion. Distinct ions' estimates are never summed.
ABUNDANCES = {'C':(1.72e-7,'III'), 'N':(2.16e-7,'V'), 'O':(4.12e-7,'IV'),
    'Al':(1.60e-7,'III'), 'Si':(3.68e-7,'IV'), 'P':(1.64e-8,'V'),
    'S':(1.71e-7,'IV'), 'Fe':(5.00e-6,'V'), 'Ni':(1.01e-6,'V')}
ALTERNATIVES = {'N':(1.58e-7,'IV'), 'P':(8.40e-8,'IV'), 'S':(5.23e-8,'VI'),
    'Fe':(1.83e-6,'IV'), 'Ni':(3.24e-7,'IV')}
LIGHT_COUNTS = {'N':{2:40,3:60,4:40,5:1}, 'O':{2:40,3:60,4:40,5:10,6:1},
    'Al':{2:40,3:20,4:1}, 'P':{2:30,3:50,4:40,5:1},
    'S':{2:30,3:60,4:40,5:30,6:1}}
DATA_FILES = {
 'verner95.dat':('https://www.pa.uky.edu/~verner/dima/photo/table1.dat',
    'fd0a315f100a78f8edb6b3fb12c0022246309cbc834394cbd0e8e5305d00b393'),
 'gfFUV99.dat.gz':('https://tlusty.oca.eu/tlusty/Synspec49/data/gfFUV99.dat.gz',
    '1ae40c9bec4aecc4d48b152263de12364acdadc8fdf156cf28891f9ae48c0cdb')}
# Kurucz measured-level ("positions") line lists for Fe/Ni IV-VII, all
# wavelengths. gfFUV99 is cut to 880-1990 A and so lacks the EUV
# 3d^(n-1)4p -> 3d^n resonance transitions; without them the excited Fe/Ni
# levels in an NLTE atom have no radiative route to the ground term.
KURUCZ_POSITION_FILES = {
 'gf2603.pos':'http://kurucz.harvard.edu/atoms/2603/gf2603.pos',
 'gf2604.pos':'http://kurucz.harvard.edu/atoms/2604/gf2604.pos',
 'gf2605z.pos':'http://kurucz.harvard.edu/atoms/2605/gf2605z.pos',
 'gf2606z.pos':'http://kurucz.harvard.edu/atoms/2606/gf2606z.pos',
 'gf2803.pos':'http://kurucz.harvard.edu/atoms/2803/gf2803.pos',
 'gf2804.pos':'http://kurucz.harvard.edu/atoms/2804/gf2804.pos',
 'gf2805.pos':'http://kurucz.harvard.edu/atoms/2805/gf2805.pos',
 'gf2806z.pos':'http://kurucz.harvard.edu/atoms/2806/gf2806z.pos'}


def checked_atomic_files(directory):
    paths = {}
    for name, (_, expected) in DATA_FILES.items():
        path = Path(directory)/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'atomic-data checksum mismatch: {path}')
        paths[name] = path
    return paths


@dataclass(frozen=True)
class OuterShellFit:
    """Verner & Yakovlev 1995 Eq. 1, lowest-threshold subshell ONLY.

    Inner-shell absorption and Auger channels are omitted. Selecting the outer
    subshell preserves the adjacent-stage SE interpretation. This is not the
    resonance-resolved Ni photoionization of later white-dwarf analyses.
    """
    element: str
    charge: int
    principal_n: int
    orbital_l: int
    threshold_energy_ev: float
    energy_scale_ev: float
    cross_section_scale_megabar: float
    shape_a: float
    shape_p: float
    shape_w: float
    maximum_energy_ev: float = 1e5

    def cross_section(self, photon_energy_ev):
        energy = np.asarray(photon_energy_ev, dtype=float)
        if np.any(~np.isfinite(energy)) or np.any(energy <= 0):
            raise ValueError('photon energy must be finite and positive')
        y = energy/self.energy_scale_ev
        value = (self.cross_section_scale_megabar*1e-18*((y-1)**2+self.shape_w**2)
                 *y**(-5.5-self.orbital_l+.5*self.shape_p)
                 *(1+np.sqrt(y/self.shape_a))**(-self.shape_p))
        return np.where((energy>=self.threshold_energy_ev)&(energy<=self.maximum_energy_ev),value,0.)


def outer_shell_fits(path, elements):
    table = np.loadtxt(path)
    result = {}
    for element in elements:
        z = ATOMIC_NUMBER[element]
        for charge in range(z):
            rows = table[(table[:,0]==z)&(table[:,1]==z-charge)]
            if len(rows)==0:
                raise ValueError(f'missing Verner95 shell: {element} {charge}')
            r = rows[np.argmin(rows[:,4])]
            result[element,charge] = OuterShellFit(element,charge,int(r[2]),int(r[3]),*r[4:10])
    return result


def convert_iron_group_lines(source, destination):
    """Lossless numeric reformat of Fe/Ni gfFUV records for existing GF reader.

    The supplied list uses measured-energy levels. Restrict to 880--1990 A
    (all vacuum), Fe III--VII and Ni III--VI. Do not read empirical observed EWs
    into the opacity. Counts and the input checksum are recorded separately.
    """
    converted = []
    counts = {}
    with gzip.open(source,'rt') as stream:
        for line in stream:
            f = line.split()
            if len(f)<11:
                continue
            code = float(f[1]); z = int(code); q = round(100*(code-z))
            if not ((z==26 and 2<=q<=6) or (z==28 and 2<=q<=5)):
                continue
            w,lg,e1,j1,e2,j2,rad,stark,vdw = map(float,
                (f[0],f[2],f[3],f[4],f[5],f[6],f[7],f[8],f[9]))
            if not 88<=w<=199:
                continue
            if int(f[10])!=0:
                raise ValueError('Fe/Ni supplementary damping records need explicit handling')
            if e1==e2 or not np.isclose(1e7/abs(e2-e1),w,atol=.01,rtol=0):
                raise ValueError('inconsistent UV wavelength / energies')
            converted.append(f'{w:11.4f}{lg:7.3f}{code:6.2f}{e1:12.3f}{j1:5.1f}'
                             f'{"gfFUV-low":10s}{e2:12.3f}{j2:5.1f} {"gfFUV-up":10s}'
                             f'{rad:6.2f}{stark:6.2f}{vdw:6.2f}'+ ' '*63+'\n')
            key=f'{z}.{q:02d}';counts[key]=counts.get(key,0)+1
    Path(destination).write_text(''.join(converted))
    return counts


def load_composition_data(data, directory, *, iron_group=True, kurucz_positions=None):
    paths = checked_atomic_files(directory)
    db = read_pg1159_atomic_database(data.stout,elements=tuple(ABUNDANCES))
    outer = outer_shell_fits(paths['verner95.dat'],('Al','P','Ni'))
    ions = dict(db.ions); supplements = {}
    for key,ion in ions.items():
        if ion.ionization_energy_ev is None and ion.charge<ATOMIC_NUMBER[ion.element]:
            fit=outer.get(key)
            if fit is None:raise ValueError(f'missing ionization threshold {key}')
            ions[key]=replace(ion,ionization_energy_ev=fit.threshold_energy_ev,
                source=ion.source+'; Verner & Yakovlev 1995 outer-shell threshold')
            supplements[f'{key[0]} {key[1]}']=fit.threshold_energy_ev
    db=AtomicDatabase(ions,db.source+'; missing thresholds from Verner & Yakovlev 1995')
    photo=read_verner_photoionization_database(data.verner_photoionization,
                                             elements=tuple(ABUNDANCES))
    fits=dict(photo.fits)
    for key,fit in outer.items():
        if key[0] in ('P','Ni'):fits[key]=fit
    photo=VernerPhotoionizationDatabase(fits,photo.source+'; P/Ni outer shell: Verner & Yakovlev 1995')
    counts={}
    if iron_group:
        converted=cache_directory()/'gfFUV99-fe-ni.gf'
        counts=convert_iron_group_lines(paths['gfFUV99.dat.gz'],converted)
        db=read_kurucz_gf100_atomic_database([converted],db,elements=('Fe','Ni'),
            minimum_wavelength_angstrom=880.,maximum_wavelength_angstrom=1990.)
    kurucz_audit=None
    if kurucz_positions is not None:
        if not iron_group:raise ValueError('Kurucz Fe/Ni positions require the iron-group mode')
        sums=dict(line.split()[::-1] for line in (Path(kurucz_positions)/'SHA256SUMS').read_text().splitlines())
        files=[]
        for name in KURUCZ_POSITION_FILES:
            path=data_file(kurucz_positions,name)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sums[name]:
                raise ValueError(f'Kurucz positions checksum mismatch: {path}')
            files.append(path)
        before={k:len(ion.transitions) for k,ion in db.ions.items() if k[0] in ('Fe','Ni')}
        # Keep the Stout and gfFUV99 lines; append only transitions whose level
        # pair is not already connected (chiefly the missing EUV lines).
        db=read_kurucz_gf100_atomic_database(files,db,elements=('Fe','Ni'),replace_transitions=False,
            supplement_missing_transitions=True)
        kurucz_audit=dict(files={n:dict(url=u,sha256=sums[n]) for n,u in KURUCZ_POSITION_FILES.items()},
            mode='supplement transitions between unconnected level pairs; existing lines unchanged',
            added_transitions={f'{k[0]} {k[1]}':len(db.ions[k].transitions)-n for k,n in before.items()})
    audit=dict(abundance_reference='https://doi.org/10.1093/mnras/stt1604',
        abundances={e:dict(number_ratio=v[0],inferred_from_ion=v[1]) for e,v in ABUNDANCES.items()},
        alternatives={e:dict(number_ratio=v[0],inferred_from_ion=v[1]) for e,v in ALTERNATIVES.items()},
        atomic_files={n:dict(url=u,sha256=h) for n,(u,h) in DATA_FILES.items()},
        ionization_threshold_supplements_ev=supplements,iron_group_input_line_counts=counts,
        kurucz_positions=kurucz_audit,
        iron_group='LTE populations, measured-level UV lines; not complete NLTE blanketing',
        missing_species=['Ge: detected in the paper, but no abundance in its Table 10; no Ge atom here'],
        missing_physics=['H/He structure response','Fe/Ni NLTE populations','inner-shell P/Ni absorption',
            'predicted-energy Fe/Ni lines','complete atoms and accurate collisions for the new light metals'])
    return db,photo,audit
