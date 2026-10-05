"""CHIANTI oxygen excitation data, mapped with the audited C adapter's rules."""
import hashlib
import json
from pathlib import Path
import numpy as np
from wd_spectra.chianti import read_chianti_scaled_collision_components
from hot_trace_collisions import map_chianti_levels, CachedFineCollisionStrength


def oxygen_collisions(directory, database, counts, *, charges=(2,3,4,5), low_level_limit=None):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    result, audit = {}, {}
    for charge in charges:
        prefix = f'o_{charge+1}'
        for suffix in ('elvlc','scups'):
            name = prefix+'.'+suffix
            actual = hashlib.sha256((directory/name).read_bytes()).hexdigest()
            if actual != manifest[name]['sha256']:
                raise ValueError(f'CHIANTI input checksum changed: {name}')
        ion = database.ions['O',charge]
        mapping, rows = map_chianti_levels(directory/(prefix+'.elvlc'), ion)
        selected = sorted(ion.levels,key=lambda l:l.energy_wavenumber)[:counts[charge]]
        selected = {l.index for l in selected}
        energies = {l.index:l.energy_wavenumber for l in ion.levels}
        raw = read_chianti_scaled_collision_components(directory/(prefix+'.scups'))
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
            key = ('O',charge,lower,upper)
            if key in result:
                raise ValueError(f'duplicate mapped CHIANTI collision {key}')
            record = CachedFineCollisionStrength((component,),
                f'CHIANTI oxygen snapshot {prefix}.scups; '+manifest[prefix+'.scups']['sha256'])
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
        raise ValueError('no selected oxygen collision pairs')
    return result,audit
