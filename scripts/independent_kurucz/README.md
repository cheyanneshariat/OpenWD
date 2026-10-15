# Reproducing the default atomic-data selection

These offline tools apply the same prospective rule to all 30 neutral/singly
ionized Kurucz inputs. They never read white-dwarf observations or run an
atmosphere. Install OpenWD plus `beautifulsoup4` for the Si II designation
table parser; `pdftotext` is needed only to reproduce the supporting-table
text extraction.

Obtain the pinned public raw files into `raw/`, using each original filename,
and independent reference files into `references/`, using the relative names
in [provenance.json](../../src/wd_spectra/data/atomic/independent-kurucz/provenance.json).
Keep the pinned NIST tab-separated export columns including f, energy, J,
accuracy, type and original strength reference. NIST and publisher pages
may change their export formatting; every input is checksum-checked rather
than silently accepting a changed table. The Aggarwal Table 2 PDF is in the
article's supplementary ZIP; extract its text with `pdftotext -layout`.

```sh
python scripts/independent_kurucz/catalog.py raw/ candidate-catalog/
python scripts/independent_kurucz/qualify.py candidate-catalog/ references/ qualified/
python scripts/independent_kurucz/verify.py qualified/
```

The first two commands refuse existing output directories. The extractor
uses the original Stout transitions and the two level extensions from the
previous default reader. It rejects predicted energies, ambiguous level
matches and conflicting duplicate strengths. The selector checks the ground
ionization limit, independent source qualifications and the factor-ten
strength rule, retaining Kurucz atomic values unchanged. Weak qualified
records are kept as well as absorption records above f=10⁻⁴.

`verify.py` compares the seven bundled arrays, every per-transition evidence
row and the selection/rejection reports for all 30 ions. NPZ compression
metadata is not part of this content comparison. The production loader still
requires the exact bundled byte checksums. These scripts do not automatically
replace production data or update checksum pins; a future catalogue revision
must be reviewed and validated explicitly.
