# Independently corroborated Kurucz catalogue, version 1

These are the default missing-transition supplements to Stout. They replace
the former full Cr II and selected Fe II/Si II imports. No observations enter
the common selection. Original Stout transitions and the two previously
imported Cr II/Fe II levels are preserved; no further levels are introduced.

The [validation report](../../../../../docs/validation/independent-kurucz.md)
describes eligibility, counts, performance and accepted shortcomings.

## File format and provenance

`manifest.json` pins every numerical catalogue and compressed evidence file.
The loader additionally pins the manifest's own SHA-256 in
`wd_spectra._kurucz_supplements`. Each NPZ contains float64 arrays:

- `levels`: existing level index, energy (cm⁻¹), statistical weight.
- `records`: lower index, upper index, vacuum wavelength (Å), absorption f,
  Einstein A (s⁻¹), total radiative rate (s⁻¹), electron Stark coefficient
  (cm³ s⁻¹), neutral-H vdW coefficient (cm³ s⁻¹), original Kurucz row number.
  NaN denotes absent damping data, not a fitted value.

The `.evidence.json.xz` files have one row per qualified record, including
weaker decay channels. They record the matched pair, Kurucz f, independent f,
their ratio, preferred source, source row, evidence flags and corroborating
source names. `provenance.json` provides public URLs, checksums and sizes for
all 30 raw Kurucz inputs and every independent reference used by the selector.
The raw reference files are not required at runtime.

Independent sources:

- [SRON Fe II orthogonal-operator calculation](https://ftp.sron.nl/pub/tonr/orth/iron/FeII.E1.loggf),
  pinned 2021-08-07 file; both level energies must be marked known.
- [SRON Cr II orthogonal-operator calculation](https://ftp.sron.nl/pub/tonr/orth/chromium/CrII.E1.loggf),
  pinned 2023-05-16 file; both level energies must be marked known.
- [Ward et al. (2023)](https://doi.org/10.3847/1538-4357/acfafe), Cr II Table 5:
  measured branching fractions with experimental or independent theoretical
  lifetimes, reported fractional A uncertainty ≤0.25.
- [Pehlivan Rhodin et al. (2024)](https://doi.org/10.1051/0004-6361/202245686),
  [CDS tables](https://cdsarc.cds.unistra.fr/ftp/J/A+A/682/A184/): Si I/II MCDHF
  rates with length/velocity gauge discrepancy ≤0.25. Measured Si I branching
  fractions with calculated lifetimes are identified separately.
- [Aggarwal & Keenan (2014)](https://doi.org/10.1093/mnras/stu883), full supporting
  Si II Table 2: GRASP strengths matched by configuration/term/J.
- [NIST ASD](https://physics.nist.gov/PhysRefData/ASD/lines_form.html), C-or-better
  E1 strengths with reviewed original provenance. Only references T9787,
  T6066, T10762, T7952, T3499 and T4922 qualify. A NIST listing that merely
  copies Kurucz does not count as independent evidence. LS-split components
  are identified in the evidence flags.

Most strengths are corroborated by independent calculations rather than
laboratory measurements. Kurucz f and damping constants are retained
unchanged; no precision uncertainty is inferred from the factor-ten screen.
The source authors retain their data rights; see
[third-party notices](../../../../../THIRD_PARTY_NOTICES.md).

## Reproduction

The [offline preparation scripts](../../../../../scripts/independent_kurucz/README.md)
rebuild candidate catalogues and the common selection using pinned public
inputs. Verification compares numerical array contents and all evidence
rows, rather than ZIP-container timestamps. Runtime needs only this bundled
directory and performs no network requests.
