# Niobium atomic data

Stout stops at Zn, so `read_niobium_atomic_ion` in `wd_spectra/metals.py`
builds Nb I--VI from these checksum-pinned files (query URLs and SHA-256 in
`NIOBIUM_ATOMIC_DATA_FILES`):

- `nist-asd-nb{1..6}-levels.tsv`: NIST ASD 5.12 energy levels, retrieved
  2026-10-07. They give the partition functions and Ritz wavelengths.
- `nist-asd-nb4-lines.tsv`: all 819 Nb IV E1 lines with NIST transition
  probabilities (Tauheed & Reader 2005 for 800 of them). In this export the
  column labelled `fik` holds the line strength S in atomic units. NIST
  prints log gf = 1.76 for 1955.93 A, while its A and S give 0.00; OpenWD
  derives f from A for every line.
- `nilsson2010-nb3-table7.csv`: hand transcription of the 76 Nb III HFR+CPOL
  gA values with log gf > -0.5 in Nilsson et al. (2010), Table 7. Every row's
  log gf agrees with its gA within 0.021 dex. Rows are matched to NIST levels
  by both J values, parity and wavelength, because NIST and Iglesias (1955)
  name some strongly mixed levels differently.
- `nist-asd-nb-ionization.csv`: NIST ionization energies.

NIST has no transition probabilities for Nb I, II or V, so those stages carry
partition functions and charge but no lines. No Nb photoionization data are
bundled; Verner's fits stop at Zn.
