# Rauch et al. Zn IV-V and Cu IV-VI line data

The bundled Stout Zn IV-VI and Cu IV-VI files contain only forbidden (M1/E2)
transitions. `atomic_database_with_rauch_zn_cu_transitions` in
`wd_spectra/metals.py` adds the E1 lines of these checksum-pinned files to the
Stout ions (source URLs and SHA-256 in `RAUCH_ZN_CU_ATOMIC_DATA_FILES`). It is
opt-in, through `DABConfig(metal_line_supplements=("rauch-zn-cu",))`.

- `rauch2014-zn4-table1.dat`, `rauch2014-zn5-table2.dat`, `rauch2014-ReadMe`:
  unmodified CDS copies of J/A+A/564/A41, the HFR log gf and gA values of
  Rauch et al. (2014, A&A 564, A41) for Zn IV (400 lines) and Zn V (1879).
  The wavelengths are in air above 2000 A. Each level, given by its energy,
  parity and J, is matched to one Stout level within 1 cm^-1. All 63 Zn IV
  levels match; 10 of 157 Zn V levels are theoretical and absent from Stout,
  so 228 Zn V lines are omitted.
- `toss-cu{4,5,6}.csv`: the Cu IV, V and VI rows of the GAVO TOSS table
  `toss.data` (Rauch et al. 2020, A&A 637, A4), retrieved on 2026-10-07 with
  the TAP query in `RAUCH_ZN_CU_ATOMIC_DATA_FILES`. The column `einsteina`
  holds gA, not A, and `vacuum_wavelength` is in metres. The service's level
  energy columns are not included: for Zn IV they disagree with the CDS
  table in 399 of 400 rows, while its wavelengths, parities, J values, log gf
  and gA agree in every row. Each Cu line is therefore attached to the unique
  Stout level pair with its parities and J values whose Ritz wavenumber lies
  within 0.6 cm^-1 of the TOSS wavenumber. Lines with no or several such
  pairs, or two lines on one pair, are omitted (attached: Cu IV 7576/8785,
  Cu V 4218/5456, Cu VI 2717/3797). Applied to the Zn tables, this
  wavelength-only matching reproduces the energy matching for every line it
  attaches.

Cu VII (2253 TOSS lines) is not bundled: Stout has only its four
ground-configuration levels. Wavelengths are Ritz values from the Stout level
energies and f follows from gA, exactly as for Stout and Nb. No levels are
added, so partition functions and the label-based broadening estimates are
unchanged.

Terms of use: CDS states that VizieR data are free for scientific use and
that the original authors and publication must be cited. TOSS asks users to
acknowledge: "The TOSS service (http://dc.g-vo.org/TOSS) used for this paper
was constructed as part of the activities of the German Astrophysical
Virtual Observatory." Neither source states an explicit open licence. Cite
Rauch et al. (2014, 2020) and acknowledge TOSS when using these data.
