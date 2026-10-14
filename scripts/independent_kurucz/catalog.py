"""Build a compact, reproducible missing-transition catalogue; no model run."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import numpy as np
from wd_spectra.metals import (read_stout_atomic_database,
    ATOMIC_NUMBER,
    augment_chromium_ii_kurucz_transitions, augment_iron_ii_kurucz_uv_transitions,
    augment_silicon_ii_kurucz_transitions)
from wd_spectra.models import ModelData
from wd_spectra.constants import PI, ELEMENTARY_CHARGE_ESU, ELECTRON_MASS, LIGHT_SPEED
LEDGER = REPO / "src/wd_spectra/data/atomic/independent-kurucz/provenance.json"
CACHE = None

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""): h.update(block)
    return h.hexdigest()

def parse(row):
    if len(row) < 101: return None
    try:
        wavelength_nm = float(row[:11]); loggf = float(row[11:18]); code = float(row[18:24])
        e1 = float(row[24:36]); g1 = 2*float(row[36:41])+1
    except ValueError: return None
    for start, stop, js, je, damping in [(51,63,63,68,79), (51,64,64,69,80),
                                        (52,64,64,69,80), (52,65,65,70,81)]:
        try:
            e2 = float(row[start:stop]); g2 = 2*float(row[js:je])+1
            widths = [float(row[damping+i:damping+i+6]) for i in (0,6,12)]
        except ValueError: continue
        break
    else: return None
    if e1 > e2 and e1 >= 0 and e2 >= 0: e1,e2,g1,g2 = e2,e1,g2,g1
    return wavelength_nm, loggf, code, e1,g1,e2,g2,*widths

def main():
    global CACHE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path, help="Local pinned Kurucz gfZZCC files")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    CACHE = args.output
    CACHE.mkdir(parents=True, exist_ok=False)
    provenance = json.loads(LEDGER.read_text())["kurucz"]
    elements = tuple(dict.fromkeys(k.split("_")[0] for k in provenance))
    raw = read_stout_atomic_database(ModelData.default().stout,
        elements=elements, maximum_charge=3, include_default_supplements=False)
    # Freeze the pre-qualification level set, including its two existing extensions.
    atom = augment_silicon_ii_kurucz_transitions(
        augment_iron_ii_kurucz_uv_transitions(augment_chromium_ii_kurucz_transitions(raw)))
    reports = {}
    for element in elements:
        for charge in [0,1]:
            info = provenance[f"{element}_{charge}"]
            path = args.raw / info["filename"]
            if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
                raise ValueError(f"Kurucz source checksum/size mismatch: {path.name}")
            ion = atom.ions[element, charge]
            bins = defaultdict(list)
            for level in ion.levels:
                bins[round(level.energy_wavenumber*10), round(level.statistical_weight)].append(level)
            def level_match(energy, weight):
                matches = [level for key in [round(energy*10)-1,round(energy*10),round(energy*10)+1]
                           for level in bins[key,round(weight)]
                           if abs(level.energy_wavenumber-energy) <= .1 and level.statistical_weight == weight]
                return matches[0] if len(matches) == 1 else None
            existing = {(t.lower_index,t.upper_index) for t in raw.ions[element, charge].transitions}
            records = {}; rejected = set(); counts = Counter()
            for number, row in enumerate(path.open(encoding="latin1"), 1):
                counts["source_rows"] += 1
                values = parse(row)
                if values is None: counts["unparsed"] += 1; continue
                _,loggf,code,e1,g1,e2,g2,rad,stark,vdw = values
                assert int(code+.000001) == ATOMIC_NUMBER[element]
                assert round(100*(code-int(code+.000001))) == charge
                if e1 < 0 or e2 < 0: counts["predicted_energy"] += 1; continue
                if e2 <= e1: counts["zero_energy_gap"] += 1; continue
                lower, upper = level_match(e1,g1), level_match(e2,g2)
                if lower is None or upper is None: counts["unmatched_or_ambiguous_levels"] += 1; continue
                pair = lower.index,upper.index
                if pair in existing: counts["existing_transition"] += 1; continue
                if pair in rejected: counts["ambiguous_duplicate_rows"] += 1; continue
                wavelength = 1e8/(e2-e1); f = 10**loggf/g1
                a = 8*PI**2*ELEMENTARY_CHARGE_ESU**2/(ELECTRON_MASS*LIGHT_SPEED*(wavelength*1e-8)**2)*g1/g2*f
                record = [lower.index,upper.index,wavelength,f,a,
                          10**rad if rad > 0 else np.nan,
                          10**stark if stark < 0 else np.nan,
                          10**vdw if vdw < 0 else np.nan, number]
                distance = abs(e1-lower.energy_wavenumber)+abs(e2-upper.energy_wavenumber)
                if pair in records:
                    old, old_distance = records[pair]
                    # Repeated entries sharing matched levels are not added twice.
                    # A materially different f is ambiguous, not averaged or fitted.
                    if abs(math.log(f/old[3])) > .01:
                        del records[pair]; rejected.add(pair); counts["ambiguous_duplicate_pairs"] += 1; continue
                    counts["duplicate_rows"] += 1
                    if distance >= old_distance: continue
                records[pair] = record,distance
            data = np.asarray([r[0] for r in records.values()], dtype=np.float64).reshape(-1,9)
            if len(data): data = data[np.argsort(data[:,2], kind="stable")]
            fingerprint = digest(path)
            levels = np.asarray([[l.index,l.energy_wavenumber,l.statistical_weight] for l in ion.levels])
            np.savez(CACHE / f"{element}_{charge}.npz", records=data, levels=levels)
            report = dict(status="prepared", element=element, charge=charge, source_url=info["url"],
                          source_filename=path.name, source_sha256=fingerprint,
                          counts=dict(counts), additional_transitions=len(data),
                          above_f_1e4=int(np.sum(data[:,3]>=1e-4)),
                          COS_above_f_1e4=int(np.sum((data[:,2]>=1132)&(data[:,2]<=1444)&(data[:,3]>=1e-4))),
                          levels_added=0)
            reports[f"{element}_{charge}"] = report
            (CACHE / "manifest.json").write_text(json.dumps(reports,indent=2)+"\n")
            print("CATALOG", element, charge, len(data), report["above_f_1e4"], flush=True)
    (CACHE / "manifest.json").write_text(json.dumps(reports,indent=2)+"\n")
    print("DONE", sum(v.get("additional_transitions",0) for v in reports.values()),flush=True)

if __name__ == "__main__": main()
