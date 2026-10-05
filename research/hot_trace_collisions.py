"""Audited CHIANTI fine-level collisions for the exploratory Stout C/Si solver.

No level-index identity is assumed. Each mapping requires configuration,
multiplicity, orbital term, J, and an independently checked level energy.
This is a research adapter; data stay outside the frozen host's table cache.
"""
from hot_daz_data import data_file
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
import numpy as np
from wd_spectra.chianti import read_chianti_scaled_collision_components
from wd_spectra.light_metal_nlte import ChiantiTermCollisionStrength


def _configuration(label):
    return re.sub(r'[.\s]', '', label)


def map_chianti_levels(path, ion):
    """Return a one-to-one mapping and audit; reject ambiguous matches."""
    candidates = {}
    for level in ion.levels:
        match = re.fullmatch(r'(.*?)\.\((\d+)([A-Z])o?<([^>]+)>\)', level.label)
        if match is None:
            continue
        configuration, spin, orbital, _ = match.groups()
        key = (_configuration(configuration), int(spin), orbital, level.statistical_weight)
        candidates.setdefault(key, []).append(level)
    mapping, audit, used = {}, [], set()
    for line in Path(path).read_text(encoding='ascii').splitlines():
        if line.strip() == '-1':
            break
        if not line.strip() or line.lstrip().startswith('%'):
            continue
        index = int(line[:7]); config = line[7:37].strip()
        spin = int(line[42:47]); orbital = line[47:52].strip()
        weight = 2 * float(line[52:57]) + 1
        observed = float(line[57:72]); theory = float(line[72:87])
        energy = observed if observed >= 0 else theory
        row = dict(chianti_index=index, configuration=config, spin=spin, orbital=orbital,
                   statistical_weight=weight, energy_wavenumber=energy,
                   energy_source='observed' if observed >= 0 else 'theoretical')
        tolerance = max(5., 2e-4 * abs(energy))
        matches = [l for l in candidates.get((_configuration(config), spin, orbital, weight), [])
                   if abs(l.energy_wavenumber-energy) <= tolerance]
        if len(matches) > 1:
            raise ValueError(f'ambiguous CHIANTI level {index}: {matches}')
        if matches:
            level = matches[0]
            if level.index in used:
                raise ValueError(f'CHIANTI levels map repeatedly onto Stout level {level.index}')
            used.add(level.index); mapping[index] = level.index
            row.update(stout_index=level.index, stout_label=level.label,
                       energy_difference_wavenumber=level.energy_wavenumber-energy)
        audit.append(row)
    if not mapping:
        raise ValueError(f'no CHIANTI levels match {ion.element} charge {ion.charge}')
    return mapping, audit


class CachedFineCollisionStrength(ChiantiTermCollisionStrength):
    """Exact CHIANTI spline values cached at fixed-host depth temperatures."""
    @lru_cache(maxsize=200000)
    def effective_collision_strength(self, temperature):
        return super().effective_collision_strength(float(temperature))


def carbon_collisions(directory, database, counts, *, charges=(2,3), low_level_limit=None):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    result, audit = {}, {}
    for charge in charges:
        prefix = f'c_{charge+1}'
        for suffix in ('elvlc','scups'):
            name = prefix+'.'+suffix
            actual = hashlib.sha256(data_file(directory,name).read_bytes()).hexdigest()
            if actual != manifest[name]['sha256']:
                raise ValueError(f'CHIANTI input checksum changed: {name}')
        ion = database.ions['C',charge]
        mapping, rows = map_chianti_levels(data_file(directory,prefix+'.elvlc'), ion)
        selected = sorted(ion.levels,key=lambda l:l.energy_wavenumber)[:counts[charge]]
        selected = {l.index for l in selected}
        energies = {l.index:l.energy_wavenumber for l in ion.levels}
        raw = read_chianti_scaled_collision_components(data_file(directory,prefix+'.scups'))
        kept = []
        radiative = {(t.lower_index,t.upper_index) for t in ion.transitions if t.einstein_a>0}
        for (a,b), component in raw.items():
            lower, upper = mapping.get(a), mapping.get(b)
            if lower not in selected or upper not in selected:
                continue
            if low_level_limit is not None and (lower>low_level_limit or upper>low_level_limit):
                continue
            if energies[upper] < energies[lower]:
                lower,upper = upper,lower
            key = ('C',charge,lower,upper)
            if key in result:
                raise ValueError(f'duplicate mapped CHIANTI collision {key}')
            record = CachedFineCollisionStrength((component,),
                f'CHIANTI carbon snapshot {prefix}.scups; '+manifest[prefix+'.scups']['sha256'])
            values = [record.effective_collision_strength(t) for t in (2e4,5e4,1e5,3e5)]
            if not np.all(np.isfinite(values)) or min(values)<0:
                raise ValueError(f'invalid CHIANTI strengths for {key}')
            result[key] = record
            kept.append(dict(chianti_pair=[a,b],stout_pair=[lower,upper],
                             collision_only=(lower,upper) not in radiative,
                             upsilon_at_50000_K=values[1]))
        audit[str(charge)] = dict(source_files={n:manifest[n] for n in (prefix+'.elvlc',prefix+'.scups')},
            total_source_levels=len(rows),matched_levels=len(mapping),
            selected_mapped_levels=sum(v in selected for v in mapping.values()),
            source_pairs=len(raw),selected_pairs=len(kept),
            collision_only_pairs=sum(r['collision_only'] for r in kept),
            level_mapping=rows,collisions=kept)
    if not result:
        raise ValueError('no selected carbon collision pairs')
    return result,audit
