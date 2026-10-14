"""Prospective independent-data eligibility for all Kurucz supplements.

No WD observation is read. Genuine Stout records are the protected base.
All additions, including the previous default Cr/Fe/Si imports, are reviewed.
"""
import ast
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shlex

import numpy as np
from bs4 import BeautifulSoup

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src/wd_spectra"
LEDGER = SRC / "data/atomic/independent-kurucz/provenance.json"
HERE = None
QUALIFIED = None
REFERENCES = None
RATED = {"AAA", "AA", "A+", "A", "B+", "B", "C+", "C"}
# Original strength provenance reviewed in the preceding read-only audit.
NIST_INDEPENDENT = {"T9787", "T6066", "T10762", "T7952", "T3499", "T4922"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_label(text):
    return re.sub(r"[\s.$|^{}_()<>]", "", text).replace("o", "")


def main():
    global HERE, QUALIFIED, REFERENCES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    HERE, REFERENCES, QUALIFIED = args.catalog.parent, args.references, args.output
    QUALIFIED.mkdir(parents=True, exist_ok=False)
    ledger = json.loads(LEDGER.read_text())
    for name, info in ledger["independent_sources"].items():
        if digest(REFERENCES / name) != info["sha256"]:
            raise ValueError(f"Independent source checksum mismatch: {name}")
    catalog = json.loads((args.catalog / "manifest.json").read_text())
    tree = ast.parse((SRC / "metals.py").read_text())
    ionization = next(ast.literal_eval(n.value.args[0]) for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
        and t.id == "IONIZATION_ENERGY_EV" for t in n.targets))
    constants = {}
    for n in ast.parse((SRC / "constants.py").read_text()).body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in ("PLANCK", "LIGHT_SPEED"):
                    constants[t.id] = ast.literal_eval(n.value)
    ev_to_cm = 1.602176634e-12 / (constants["PLANCK"] * constants["LIGHT_SPEED"])
    reports, qualified_manifest = {}, {}
    source_files = set()
    for key, info in catalog.items():
        with np.load(args.catalog / (key + ".npz")) as data:
            rows, levels = data["records"], data["levels"]
        index_energy = {int(i): float(e) for i, e, g in levels}
        index_weight = {int(i): float(g) for i, e, g in levels}
        bins = defaultdict(list)
        for i, e, g in levels:
            bins[round(e), round(g)].append((int(i), e, g))

        def match(e, g):
            found = [v for bucket in (round(e)-1, round(e), round(e)+1)
                     for v in bins[bucket, round(g)] if abs(v[1]-e) <= 1 and v[2] == g]
            return found[0][0] if len(found) == 1 else None

        # Each source has its own duplicate-conflict rejection before fusion.
        evidence = defaultdict(list)
        counts = Counter()

        def add_records(records, reference, priority):
            seen, rejected = {}, set()
            for lo, up, f, row_no, flags in records:
                if lo is None or up is None or f <= 0 or not np.isfinite(f):
                    counts[reference + ":unmatched_or_invalid"] += 1
                    continue
                if index_energy[lo] > index_energy[up]: lo, up = up, lo
                pair = lo, up
                if pair in rejected: continue
                if pair in seen and abs(math.log(f / seen[pair][0])) > .01:
                    rejected.add(pair); del seen[pair]
                    counts[reference + ":conflicting_duplicate"] += 1
                    continue
                seen[pair] = f, row_no, flags
            for pair, (f, row_no, flags) in seen.items():
                evidence[pair].append(dict(f=f, source=reference, source_row=row_no,
                    priority=priority, flags=flags))
            counts[reference + ":matched_pairs"] = len(seen)

        if key in ("Cr_1", "Fe_1"):
            path = (REFERENCES / "CrII_2023.E1.loggf" if key == "Cr_1"
                    else REFERENCES / "FeII_2021.E1.loggf")
            source_files.add(path)
            pattern = re.compile(r"^\s*[\d.]+\s+([-+\d.]+)\s+([\d.]+)\s+([\d.]+\*?)\s+.*?\s+([\d.]+)\s+([\d.]+\*?)\s+")
            parsed = []
            for number, text in enumerate(path.open(), 1):
                m = pattern.match(text)
                if not m: continue
                loggf, j1, e1, j2, e2 = m.groups()
                if "*" not in e1 or "*" not in e2:
                    counts["SRON:unknown_energy"] += 1; continue
                e1, e2, j1, j2 = float(e1[:-1]), float(e2[:-1]), float(j1), float(j2)
                lo, up = match(e1, 2*j1+1), match(e2, 2*j2+1)
                lower_g = 2 * (j1 if e1 < e2 else j2) + 1
                parsed.append((lo, up, 10**float(loggf)/lower_g, number,
                    "independent calculation; both energies marked known"))
            add_records(parsed, "SRON-" + key, 2)

        if key == "Cr_1":
            path = REFERENCES / "ward2023/table5_mrt.txt"
            source_files.add(path)
            parsed = []
            for number, text in enumerate(path.read_text().splitlines(), 1):
                try:
                    e1, e2 = float(text[70:80]), float(text[81:91])
                    j1 = Fraction(text[22:44].split()[-1])
                    j2 = Fraction(text[45:69].split()[-1])
                    rate, error = float(text[105:111])*1e6, float(text[112:116])*1e6
                except (ValueError, IndexError, ZeroDivisionError): continue
                if rate <= 0 or error/rate > .25: continue
                lo, up = match(e1, 2*float(j1)+1), match(e2, 2*float(j2)+1)
                wavelength = 1e8/(e2-e1)
                f = 1.499193783498723e-16 * wavelength**2 * (2*float(j2)+1)/(2*float(j1)+1) * rate
                parsed.append((lo, up, f, number,
                    "measured branching fraction; experimental or independent theoretical lifetime; relative uncertainty <=0.25"))
            add_records(parsed, "Ward2023-Cr_1", 0)

        nist = REFERENCES / "nist" / (key + ".tsv")
        if nist.exists():
            source_files.add(nist)
            parsed = []
            for number, row in enumerate(csv.DictReader(nist.open(), delimiter="\t"), 2):
                try:
                    if row["Acc"].strip() not in RATED: continue
                    refs = set(re.findall(r"T\d+", row["tp_ref"]))
                    if not refs or not refs <= NIST_INDEPENDENT:
                        counts["NIST:unreviewed_strength_provenance"] += 1; continue
                    if row["Type"].strip() not in ("", "E1"): continue
                    e1 = float(row["Ei(cm-1)"].strip("[]"))
                    e2 = float(row["Ek(cm-1)"].strip("[]"))
                    lo = match(e1, 2*float(Fraction(row["J_i"]))+1)
                    up = match(e2, 2*float(Fraction(row["J_k"]))+1)
                    flags = "rated " + row["Acc"] + "; " + row["tp_ref"]
                    if "LS" in row["tp_ref"]: flags += "; component derived by LS splitting"
                    parsed.append((lo, up, float(row["fik"]), number, flags))
                except (ValueError, KeyError, ZeroDivisionError): continue
            add_records(parsed, "NIST-" + key, 0)

        if key in ("Si_0", "Si_1"):
            folder = REFERENCES / "aa682a184"
            level_path = folder / ("tableb1.dat" if key == "Si_0" else "tableb5.dat")
            rates_path = folder / ("tableb4.dat" if key == "Si_0" else "tableb8.dat")
            source_files.update([level_path, rates_path, folder/"ReadMe"])
            mapped = {}
            for text in level_path.read_text().splitlines():
                name = text[:36] if key == "Si_0" else text[:23]
                energy = float(text[64:73] if key == "Si_0" else text[44:53])
                js = re.findall(r"_(\d+(?:/\d+)?)_", name)
                if not js: continue
                mapped[normalize_label(name)] = match(energy, 2*float(Fraction(js[-1]))+1)
            parsed = []
            for number, text in enumerate(rates_path.read_text().splitlines(), 1):
                if key == "Si_0":
                    upper, lower, gf, uncertainty = text[:46], text[47:93], text[148:157], text[158:163]
                else:
                    upper, lower, gf, uncertainty = text[:22], text[23:45], text[100:109], text[110:115]
                if float(uncertainty) > .25: continue
                lo, up = mapped.get(normalize_label(lower)), mapped.get(normalize_label(upper))
                if lo is None: continue
                parsed.append((lo, up, float(gf)/index_weight[lo], number,
                    "MCDHF; reported gauge discrepancy <=0.25; not a precision guarantee"))
            add_records(parsed, "Pehlivan2024-" + key, 1)
            if key == "Si_0":
                path = folder / "tableb3.dat"; source_files.add(path)
                measured = []
                for number, text in enumerate(path.read_text().splitlines(), 1):
                    lo = mapped.get(normalize_label(text[18:38]))
                    up = mapped.get(normalize_label(text[:17]))
                    if lo is None or up is None: continue
                    loggf, error = float(text[76:82]), float(text[83:87])
                    if 10**error-1 > .25: continue
                    measured.append((lo, up, 10**loggf/index_weight[lo], number,
                        "measured branching fraction and calculated lifetime; reported uncertainty <=0.25"))
                add_records(measured, "Pehlivan2024-BF-Si_0", 0)

        if key == "Si_1":
            # Theory energies are not forced to match experiment. Map all 56
            # configurations/terms/J uniquely by spectroscopic designation.
            path = SRC / "data/runtime/cache/metal-opacity/atomic-line-list/stout/si/si_2/si_2.nrg"
            labels = defaultdict(list)
            for text in path.read_text().splitlines():
                p = shlex.split(text)
                if len(p) == 4 and p[0].isdigit(): labels[normalize_label(p[3])].append(int(p[0]))
            html_path = REFERENCES / "aggarwal2014.html"
            pdf_text = REFERENCES / "aggarwal2014_tab2.txt"
            source_files.update([html_path, pdf_text, HERE/"references/aggarwal2014_tab2.pdf"])
            paper_map = {}
            soup = BeautifulSoup(html_path.read_text(), "html.parser")
            for table in soup.find_all("table"):
                mapped_here = {}
                for tr in table.find_all("tr"):
                    cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                    if len(cells) != 10 or not cells[0].isdigit(): continue
                    found = labels[normalize_label(cells[1]+cells[2])]
                    if len(found) == 1: mapped_here[int(cells[0])] = found[0]
                if len(mapped_here) > len(paper_map): paper_map = mapped_here
            assert len(paper_map) >= 40, ("Si II designation mapping incomplete", len(paper_map))
            def scientific(text):
                text = text.replace("−", "-")
                return float(re.sub(r"(?<=\d)([+-]\d+)$", r"e\1", text))
            parsed, all_pairs, e1_count = [], set(), 0
            for number, text in enumerate(pdf_text.read_text().splitlines(), 1):
                p = text.split()
                if len(p) != 9 or not p[0].isdigit() or not p[1].isdigit(): continue
                i, j = int(p[0]), int(p[1]); all_pairs.add((i,j))
                f = scientific(p[4])
                if f <= 0: continue
                e1_count += 1
                parsed.append((paper_map.get(i), paper_map.get(j), f, number,
                    "GRASP independent calculation; identity mapped by configuration/term/J"))
            # All 1385 printed rows are read. The prose says 460 E1 entries;
            # the complete supporting PDF actually prints 458 nonzero f's.
            # No values are invented for zero or omitted entries.
            assert len(all_pairs) == 1385 and e1_count == 458, "Supporting-table extraction changed"
            counts["Aggarwal2014:printed_nonzero_E1"] = e1_count
            counts["Aggarwal2014:mapped_levels"] = len(paper_map)
            add_records(parsed, "AggarwalKeenan2014-Si_1", 2)

        limit = ionization[info["element"]][info["charge"]] * ev_to_cm
        keep = np.zeros(len(rows), dtype=bool)
        selected_evidence = []
        why = Counter()
        for i, row in enumerate(rows):
            pair = int(row[0]), int(row[1])
            if max(index_energy[v] for v in pair) >= limit:
                why["above_ionization_limit"] += 1; continue
            sources = evidence.get(pair, [])
            if not sources:
                why["no_qualified_independent_strength"] += 1; continue
            sources = sorted(sources, key=lambda e: (e["priority"], e["source"], e["source_row"]))
            best = sources[0]
            ratio = float(row[3]) / best["f"]
            if not .1 <= ratio <= 10:
                why["gross_strength_disagreement"] += 1; continue
            keep[i] = True
            selected_evidence.append(dict(pair=list(pair), wavelength_A=float(row[2]),
                Kurucz_f=float(row[3]), independent_f=best["f"], ratio=ratio,
                primary_evidence=best, corroborating_sources=[e["source"] for e in sources]))
        target = QUALIFIED / (key + ".npz")
        np.savez_compressed(target, records=rows[keep], levels=levels)
        (QUALIFIED / (key + ".evidence.json")).write_text(json.dumps(selected_evidence, indent=2)+"\n")
        reports[key] = dict(considered=len(rows), qualified=int(keep.sum()),
            qualified_above_floor=int(np.sum(keep & (rows[:,3] >= 1e-4))),
            rejection_counts=dict(why), source_matching_counts=dict(counts))
        qualified_manifest[key] = dict(info, additional_transitions=int(keep.sum()),
            above_f_1e4=reports[key]["qualified_above_floor"], qualified_sha256=digest(target))
        print(key, "qualified", reports[key]["qualified_above_floor"], flush=True)
    (QUALIFIED / "manifest.json").write_text(json.dumps(qualified_manifest, indent=2)+"\n")
    (QUALIFIED / "atomic_policy.json").write_text(json.dumps(dict(
        rule="All Kurucz additions relative to original Stout: uniquely matched observed existing levels, both bound, and qualified independent strength agreeing within factor ten",
        strength_values_used="Kurucz; no fitted oscillator strength or width adjustment",
        independent_source_priority=["reviewed NIST C-or-better", "2024 MCDHF with reported gauge discrepancy <=0.25", "independent observed-level calculation"],
        elements=list(dict.fromkeys(i["element"] for i in catalog.values())),
        chromium_exception=False, previous_Kurucz_supplements_exempt=False,
        original_Stout_records_preserved=True, existing_level_set_preserved=True,
        WD_observations_used=False, f_floor=1e-4, opacity_threshold=1e-3,
        report_by_ion=reports,
        source_sha256={str(p.relative_to(REFERENCES)):digest(p) for p in sorted(source_files)},
        preparation_script_sha256=digest(Path(__file__))), indent=2)+"\n")


if __name__ == "__main__":
    main()
