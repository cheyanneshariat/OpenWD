#!/usr/bin/env python3
"""Build data/stark/kurucz_stark_constants.csv.xz from Kurucz per-ion line lists.

Kurucz gfZZII.all / gfZZII.lines files (http://kurucz.harvard.edu/atoms/ZZII/) give, for each line,
log10(Gamma_e/n_e) [angular FWHM, cm3 s-1] at 10,000 K.  Each bundled Stout ion is matched with
``read_kurucz_gf100_atomic_database(..., replace_matched_atomic_data=True)`` (equal level energies and
statistical weights), and the Kurucz Stark constant of every matched transition is recorded under its
Stout level indices, together with the level energies and weights as a consistency check.  Indices are
needed because Stout contains distinct levels with equal energy and weight (e.g. Fe I 61063.98 cm-1).  Oscillator strengths, van der Waals and
radiative constants are not used.

usage: build_kurucz_stark_table.py KURUCZ_DIR [OUTPUT]
KURUCZ_DIR must contain gfZZII.all or gfZZII.lines files and a PROVENANCE.json recording each
file's source URL, original SHA256, and any trimming applied (the files used for the bundled table
were trimmed to lines between observed levels with a tabulated Stark constant, 90-1100 nm).
"""
import json, lzma, sys
from pathlib import Path
from types import MappingProxyType
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wd_spectra.metals import AtomicDatabase, read_kurucz_gf100_atomic_database, read_stout_atomic_database
from wd_spectra.models import ModelData

ATOMIC_NUMBER = {'C': 6, 'N': 7, 'O': 8, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17,
                 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27,
                 'Ni': 28, 'Cu': 29, 'Zn': 30}


def main():
    kdir = Path(sys.argv[1]); out = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path(__file__).resolve().parents[1] / 'src/wd_spectra/data/stark/kurucz_stark_constants.csv.xz'
    prov = json.loads((kdir / 'PROVENANCE.json').read_text())
    base = read_stout_atomic_database(ModelData.default().stout, maximum_charge=3)
    rows, used = [], {}
    for (el, ch), ion in sorted(base.ions.items()):
        z = ATOMIC_NUMBER.get(el)
        if z is None:
            continue
        files = [p for p in (kdir / f'gf{z:02d}{ch:02d}.all', kdir / f'gf{z:02d}{ch:02d}.lines') if p.exists()]
        if not files:
            continue
        sub = AtomicDatabase(MappingProxyType({(el, ch): ion}), source=base.source)
        matched = read_kurucz_gf100_atomic_database([files[0]], sub, elements=(el,), replace_transitions=False,
                                                     replace_matched_atomic_data=True).ions[el, ch]
        levels = {lv.index: lv for lv in ion.levels}
        n = 0
        for t, m in zip(ion.transitions, matched.transitions):
            assert (t.lower_index, t.upper_index) == (m.lower_index, m.upper_index)
            # Record only Kurucz constants for transitions Stout leaves without a Stark width
            # (the matched transition keeps any width Stout already has).
            c = m.electron_stark_rate_coefficient_cm3_s
            if c is None or t.electron_stark_rate_coefficient_cm3_s is not None:
                continue
            lo, up = levels[t.lower_index], levels[t.upper_index]
            rows.append((el, ch, t.lower_index, t.upper_index, lo.energy_wavenumber, lo.statistical_weight, up.energy_wavenumber,
                         up.statistical_weight, t.wavelength_vacuum_angstrom, np.log10(c)))
            n += 1
        used[files[0].name] = dict(transitions=n, **{k: prov[files[0].name][k] for k in
                                                       ('url', 'original_sha256', 'criterion') if k in prov.get(files[0].name, {})},
                                   wavelength_trim_nm=prov.get(files[0].name, {}).get('wavelength_trim_nm'))
    header = ['# Kurucz electron-impact Stark constants log10(Gamma_e/n_e) [angular FWHM, cm3 s-1] at 10,000 K,',
              '# keyed by Stout v3.00b4 level indices; level energies (cm-1) and weights are checked on use.',
              '# Built by scripts/build_kurucz_stark_table.py from Kurucz per-ion files (source URL, original',
              '# SHA256 and trimming criterion per file in the trailing JSON comment block). Coverage: lines',
              '# 900-11000 A between observed levels with a tabulated Kurucz Stark constant.',
              'element,charge,lower_index,upper_index,lower_energy_cm1,lower_g,upper_energy_cm1,upper_g,vacuum_wavelength_A,log10_gamma_e_per_ne_10000K']
    body = [f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f},{r[5]:g},{r[6]:.4f},{r[7]:g},{r[8]:.4f},{r[9]:.3f}' for r in rows]
    tail = ['# SOURCES ' + json.dumps(used, sort_keys=True)]
    out.write_bytes(lzma.compress(('\n'.join(header + body + tail) + '\n').encode(), preset=9))
    print(f'{len(rows)} transitions from {len(used)} files -> {out} ({out.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
