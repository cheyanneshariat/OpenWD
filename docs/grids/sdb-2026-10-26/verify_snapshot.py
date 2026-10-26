"""Check the portable sdB metadata or extracted file bytes; no model solve or NumPy."""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(name):
    p = PurePosixPath(name)
    require(bool(p.parts) and not p.is_absolute() and '..' not in p.parts
            and '\\' not in name, 'Unsafe data path: ' + name)
    return p


def validate_metadata(rows, manifest, masks, qualification, composition):
    by_id = {r['coordinate_id']: r for r in rows}
    require(len(rows) == len(by_id) == 1573, 'Expected 1573 unique coordinates')
    coords = {(int(r['teff_K']), round(float(r['logg_cgs'])*10),
               round(float(r['log10_nHe_nH'])*10)) for r in rows}
    require(coords == set(itertools.product(range(20000, 44001, 2000),
            range(48, 69, 2), range(-50, 1, 5))), 'Requested coordinate membership differs')
    for row in rows:
        require(float(row['log10_nH_nHe']) == -float(row['log10_nHe_nH']),
                'Helium sign mismatch')
        require(float(row['logg_cgs'])*10 == round(float(row['logg_cgs'])*10)
                and float(row['log10_nHe_nH'])*10 == round(float(row['log10_nHe_nH'])*10),
                'Off-grid coordinate')
        require(row['independent_abundance_axes_available'] == 'False'
                and row['physical_accuracy_established'] == 'False',
                'Unexpected scientific qualification')
    hhe = {r['coordinate_id'] for r in rows if r['HHe_status'] == 'Verified'}
    metals = {r['coordinate_id'] for r in rows if r['Metals_status'] == 'QUALIFIED_METALS'}
    failures = {r['coordinate_id'] for r in rows if r['Metals_status'] in {
        'METALS_TIMEOUT', 'METALS_PROCESS_CAP_EXCLUDED',
        'METALS_NATIVE_SIGNAL_ERROR', 'METALS_NATIVE_NOT_CONVERGED'}}
    require((len(hhe), len(metals), len(failures)) == (1542, 1523, 19),
            'Qualified/failure totals differ')
    require(metals.isdisjoint(failures) and metals | failures == hhe,
            'Metals partition differs from qualified HHe')
    require(set(masks['HHe_fit_host_ids']) == hhe
            and set(masks['metals_fit_host_ids']) == metals
            and set(masks['metals_failure_host_ids']) == failures
            and set(masks['HHe_ineligible_host_ids']) == set(by_id)-hhe,
            'Fitting masks disagree with index')
    require(dict(collections.Counter(r['HHe_status'] for r in rows)) == {
        'Verified':1542, 'Timed out':5, 'Independent check failed':6,
        'Native NLTE not converged':17, 'Native LTE check failed':3}, 'HHe outcome classes differ')
    require(dict(collections.Counter(r['Metals_status'] for r in rows)) == {
        'QUALIFIED_METALS':1523, 'METALS_TIMEOUT':14,
        'METALS_PROCESS_CAP_EXCLUDED':2, 'METALS_NATIVE_SIGNAL_ERROR':2,
        'METALS_NATIVE_NOT_CONVERGED':1, 'INELIGIBLE_HHe_FAILED':31}, 'Metals outcome classes differ')
    members = {v['path']: v for v in manifest['members']}
    require(len(members) == len(manifest['members']) == 7672, 'Duplicate or missing members')
    expected = {f'HHe/{cid}/{name}.npz' for cid in hhe
                for name in ['spectrum','populations','atmosphere']}
    expected |= {f'Metals/{cid}/{name}.npz' for cid in metals
                 for name in ['spectrum','populations']}
    require(set(members) == expected, 'Manifest differs from accepted model IDs')
    archives = {f'sdb-2026-10-26-HHe-{i:02d}.tar.gz' for i in range(1,4)}
    archives |= {f'sdb-2026-10-26-Metals-{i:02d}.tar.gz' for i in range(1,6)}
    for path, item in members.items():
        parts = safe_path(path).parts
        require(item['bytes'] > 0 and len(item['sha256']) == 64
                and all(c in '0123456789abcdef' for c in item['sha256']), 'Invalid file identity')
        require(item['coordinate_id'] == parts[1] and item['family'] == parts[0],
                'Member coordinate/family mismatch')
        require(item['archive'] in archives, 'Unexpected archive')
    require({v['archive'] for v in members.values()} == archives, 'Archive membership differs')
    require(sum(v['bytes'] for v in members.values()) == manifest['native_bytes'],
            'Native byte total differs')
    for row in rows:
        cid = row['coordinate_id']
        for family, ids, flag in [('HHe',hhe,'eligible_for_HHe_fit'),
                                  ('Metals',metals,'eligible_for_fixed_metals_fit')]:
            path = row[family+'_spectrum_path']
            require(row[flag] == str(cid in ids), 'Eligibility flag differs from status')
            if cid in ids:
                require(path == f'{family}/{cid}/spectrum.npz'
                        and members[path]['sha256'] == row[family+'_spectrum_sha256'],
                        'Spectrum hash/path mismatch')
            else:
                require(not path and not row[family+'_spectrum_sha256'], 'Excluded model has a spectrum')
    models = qualification['models']
    require(set(models) == hhe, 'Qualification host set differs')
    for cid in hhe:
        for entry in models[cid].values():
            co = entry['coordinate']; row = by_id[cid]
            require(co['coordinate_id'] == cid and co['teff_K'] == int(row['teff_K'])
                    and co['logg_cgs'] == float(row['logg_cgs'])
                    and co['log_nHe_nH'] == float(row['log10_nHe_nH']),
                    'Qualification coordinate differs from index')
            for key in ['terminal_proof_sha256','local_delivery_proof_sha256']:
                require(len(entry[key]) == 64 and all(c in '0123456789abcdef'
                        for c in entry[key]), 'Invalid proof identity')
        require(0 <= models[cid]['HHe']['original_ordinary_rate_independent_defect'] <= 1e-4,
                'HHe defect exceeds original threshold')
        require(('Metals' in models[cid]) == (cid in metals), 'Metals qualification set differs')
    for cid in metals:
        q = models[cid]['Metals']
        require(0 <= q['native_undamped_defect'] < .001 and 1 <= q['native_iterations'] <= 120,
                'Metals defect/iterations exceed threshold')
        nH = (1-composition['original_listed_metals_number_fraction_sum']) / (
            1 + 10**float(by_id[cid]['log10_nHe_nH']))
        require(math.isclose(nH, q['nH_used_for_conversion'], rel_tol=0, abs_tol=1e-12),
                'nH conversion differs')
        for mode in ['NLTE','LTE']:
            require(set(q['composition'][mode]) == set(composition[mode]), 'Element set differs')
            for element in composition[mode]:
                want = composition['log_number_fraction_relative_to_all_particles'][element]-math.log10(nH)
                require(abs(q['composition'][mode][element]-want) < 1e-12, 'X/H conversion differs')
    return dict(coordinates=1573,HHe_qualified=len(hhe),metals_qualified=len(metals),native_files=len(members))


def load_metadata(directory):
    with (directory/'model-index.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    get = lambda name: json.loads((directory/name).read_text())
    return rows,get('manifest.json'),get('fitting-masks.json'),get('qualification.json'),get('metals-composition.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,help='Also verify extracted native file bytes')
    args = parser.parse_args(); metadata = load_metadata(Path(__file__).resolve().parent)
    result = validate_metadata(*metadata)
    if args.data_root:
        root = args.data_root.resolve(strict=True); expected = {v['path'] for v in metadata[1]['members']}
        extras = []; sidecars = []
        for path in root.rglob('*.npz'):
            rel = str(path.relative_to(root))
            if rel in expected: continue
            if path.name.startswith('._') and str(path.with_name(path.name[2:]).relative_to(root)) in expected:
                with path.open('rb') as stream: magic = stream.read(8)
                require(magic == bytes.fromhex('0005160700020000'), 'Extra NPZ is not AppleDouble metadata')
                sidecars.append(rel)
            else: extras.append(rel)
        require(not extras, 'Unexpected native files: '+repr(extras))
        for item in metadata[1]['members']:
            path = root.joinpath(*safe_path(item['path']).parts).resolve(strict=True)
            require(root in path.parents and path.is_file(), 'File leaves data root')
            require(path.stat().st_size == item['bytes'], 'File size differs: '+item['path'])
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
            require(h.hexdigest() == item['sha256'], 'File hash differs: '+item['path'])
        result.update(extracted_file_hashes_checked=len(expected),AppleDouble_metadata=sidecars)
    print(json.dumps(dict(passed=True,**result)))


if __name__ == '__main__': main()
