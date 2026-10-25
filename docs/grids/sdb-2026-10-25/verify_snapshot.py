"""Check the portable sdB snapshot. No model generation or array deserialization."""
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
    path = PurePosixPath(name)
    require(bool(path.parts) and not path.is_absolute() and '..' not in path.parts
            and '\\' not in name, 'Unsafe data path: ' + name)
    return path


def validate_metadata(rows, manifest, masks, qualification, composition):
    """Validate exact memberships and cross-file relationships, not only totals."""
    by_id = {r['coordinate_id']: r for r in rows}
    require(len(rows) == len(by_id) == 1573, 'Expected 1573 unique coordinates')
    coords = {(int(r['teff_K']), round(float(r['logg_cgs'])*10),
               round(float(r['log10_nHe_nH'])*10)) for r in rows}
    requested = set(itertools.product(range(20000, 44001, 2000),
                                     range(48, 69, 2), range(-50, 1, 5)))
    require(coords == requested, 'Requested coordinate membership differs')
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
    ref = {r['coordinate_id'] for r in rows if r['reference_status'] == 'QUALIFIED_REFERENCE'}
    failures = {r['coordinate_id'] for r in rows if r['reference_status'] in {
        'REFERENCE_TIMEOUT', 'REFERENCE_PROCESS_CAP_EXCLUDED', 'REFERENCE_NATIVE_SIGNAL_ERROR'}}
    require(len(hhe) == 1381 and len(ref) == 696 and len(failures) == 685,
            'Qualified/failure totals differ')
    require(ref.isdisjoint(failures) and ref | failures == hhe,
            'Reference partition differs from qualified HHe')
    require(set(masks['HHe_fit_host_ids']) == hhe
            and set(masks['reference_fit_host_ids']) == ref
            and set(masks['reference_failure_host_ids']) == failures
            and set(masks['HHe_ineligible_host_ids']) == set(by_id) - hhe,
            'Fitting masks disagree with index')
    require(dict(collections.Counter(r['HHe_status'] for r in rows)) == {
        'Verified':1381, 'Timed out':182, 'Independent check failed':6,
        'Native NLTE not converged':2, 'Native LTE check failed':2}, 'HHe outcome classes differ')
    require(dict(collections.Counter(r['reference_status'] for r in rows)) == {
        'QUALIFIED_REFERENCE':696, 'REFERENCE_TIMEOUT':682,
        'REFERENCE_PROCESS_CAP_EXCLUDED':2, 'REFERENCE_NATIVE_SIGNAL_ERROR':1,
        'INELIGIBLE_HHe_FAILED':192}, 'Reference outcome classes differ')
    members = {m['path']: m for m in manifest['members']}
    require(len(members) == len(manifest['members']) == 5535, 'Duplicate or missing members')
    expected = set()
    for cid in hhe:
        expected.update(f'HHe/{cid}/{n}.npz' for n in ['spectrum','populations','atmosphere'])
    for cid in ref:
        expected.update(f'reference-metals/{cid}/{n}.npz' for n in ['spectrum','populations'])
    require(set(members) == expected, 'Manifest differs from accepted model IDs')
    for path, item in members.items():
        safe_path(path)
        require(item['bytes'] > 0 and len(item['sha256']) == 64
                and all(c in '0123456789abcdef' for c in item['sha256']), 'Invalid file identity')
        require(item['coordinate_id'] == PurePosixPath(path).parts[1], 'Member coordinate mismatch')
        require(item['archive'] in {f'sdb-2026-10-25-{family}-{n:02d}.tar.gz'
                for family in ['HHe','reference-metals'] for n in [1,2]}, 'Unexpected archive')
    for row in rows:
        cid = row['coordinate_id']
        for key, ids, family in [('HHe',hhe,'HHe'), ('reference',ref,'reference-metals')]:
            path = row[key + '_spectrum_path']
            require(row['eligible_for_HHe_fit' if key=='HHe' else 'eligible_for_reference_composition_fit']
                    == str(cid in ids), 'Eligibility flag differs from status')
            if cid in ids:
                require(path == f'{family}/{cid}/spectrum.npz'
                        and members[path]['sha256'] == row[key + '_spectrum_sha256'],
                        'Spectrum hash/path mismatch')
            else:
                require(not path and not row[key + '_spectrum_sha256'], 'Excluded model has a spectrum')
    models = qualification['models']
    require(set(models) == hhe, 'Qualification host set differs')
    for cid in hhe:
        q = models[cid]['HHe']
        for entry in models[cid].values():
            co = entry['coordinate']
            row = by_id[cid]
            require(co['coordinate_id'] == cid and co['teff_K'] == int(row['teff_K'])
                    and co['logg_cgs'] == float(row['logg_cgs'])
                    and co['log_nHe_nH'] == float(row['log10_nHe_nH']),
                    'Qualification coordinate differs from index')
        require(0 <= q['original_ordinary_rate_independent_defect'] <= 1e-4,
                'HHe defect exceeds original threshold')
        require(('reference-metals' in models[cid]) == (cid in ref), 'Metal qualification set differs')
    for cid in ref:
        q = models[cid]['reference-metals']
        require(0 <= q['native_undamped_defect'] < .001, 'Metal defect exceeds threshold')
        nH = (1-composition['original_listed_metals_number_fraction_sum']) / (
            1 + 10**float(by_id[cid]['log10_nHe_nH']))
        require(math.isclose(nH,q['reference_nH'],rel_tol=0,abs_tol=1e-12), 'nH conversion differs')
        for mode in ['NLTE','LTE']:
            require(set(q['composition'][mode]) == set(composition[mode]), 'Element set differs')
            for element in composition[mode]:
                want = composition['log_number_fraction_relative_to_all_particles'][element]-math.log10(nH)
                require(abs(q['composition'][mode][element]-want)<1e-12, 'X/H conversion differs')
    return dict(coordinates=1573,HHe_qualified=len(hhe),reference_qualified=len(ref),native_files=len(members))


def load_metadata(directory):
    with (directory/'model-index.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    get = lambda name: json.loads((directory/name).read_text())
    return rows,get('manifest.json'),get('fitting-masks.json'),get('qualification.json'),get('reference-composition.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, help='Also check extracted native file bytes')
    args = parser.parse_args()
    metadata = load_metadata(Path(__file__).resolve().parent)
    result = validate_metadata(*metadata)
    if args.data_root:
        root = args.data_root.resolve(strict=True)
        for item in metadata[1]['members']:
            path = root.joinpath(*safe_path(item['path']).parts).resolve(strict=True)
            require(root in path.parents and path.is_file(), 'File leaves data root')
            require(path.stat().st_size == item['bytes'], 'File size differs: '+item['path'])
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(block)
            require(digest.hexdigest()==item['sha256'], 'File hash differs: '+item['path'])
        result['extracted_file_hashes_checked'] = len(metadata[1]['members'])
    print(json.dumps(dict(passed=True,**result)))


if __name__ == '__main__':
    main()
