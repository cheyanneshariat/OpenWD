# Bundled atomic supplements

## Default: independently corroborated Kurucz lines

`read_stout_atomic_database` applies the checksum-pinned
[common selection](independent-kurucz/README.md) to all loaded eligible ions.
It admits 19,966 absorption additions in C I, Si I/II, S I, Ti I, Cr II and
Fe II, with independent source evidence and the same rule for every element.
Original Stout records and the two previously added levels are retained.
Qualified weaker branches contribute to natural widths. Runtime ranking,
line budgets and a conservative addition-only peak-opacity screen control cost.

This replaces the former default full Cr II and selected Fe II/Si II imports
described below. Those files and explicit augmentation functions remain for
historical controls and bare/custom-atom experiments. Calling them on an
already qualified default atom leaves that atom unchanged. Set
`include_default_supplements=False` to read original Stout alone; the legacy
Si II configuration flag does not remove qualified lines already in the reader.
See the [observational and timing validation](../../../../docs/validation/independent-kurucz.md).

## Historical full Kurucz Cr II import

`gf2401.all.xz` is an xz-compressed, otherwise unmodified copy of Robert L.
Kurucz's full Cr II list:

- Source: <http://kurucz.harvard.edu/atoms/2401/gf2401.all>
- SHA-256 of the decompressed file:
  `6f0c4d0e01421fb0549ddbcf5649af391ca9b930e4e83e693e662d5874505205`
- Uncompressed size: 14,588,693 bytes.

`augment_chromium_ii_kurucz_transitions` applies it to a bare/custom atom,
using `read_kurucz_gf100_atomic_database` in missing-transition mode with no
wavelength restriction. Existing Stout levels and transitions are retained
unchanged. Levels match within 0.1 cm^-1 and equal statistical weight;
transitions match by level pair and vacuum wavelength within 0.08 Å.
Kurucz vacuum wavelengths are calculated from the source level energies.
Added lines retain the source oscillator strengths and radiative, electron
Stark and neutral-H damping constants supported by the reader.

For the bundled Stout Cr II atom this changes 913 levels / 138 transitions
to 914 levels / 90,559 transitions. The new level's contribution to the ideal
partition function is below 1.5e-5 at 15,000 K. Other ions are unchanged.
The source includes weak and predicted transitions; normal strength,
population and line-budget cuts still select the lines actually used.

The package verifies the decompressed checksum before reading it and requires
no download. `include_default_supplements=False` reads the original Stout
database for diagnostic comparisons. Reapplying
`augment_chromium_ii_kurucz_transitions` is idempotent.

These are independently authored atomic data; credit Robert L. Kurucz and
the original source in scientific uses. See [third-party notices](../../../../THIRD_PARTY_NOTICES.md).

## Historical Kurucz Si II import

`gf1401_observed.all.xz` is an xz-compressed subset of Kurucz's Si II list:

- Source: <http://kurucz.harvard.edu/atoms/1401/gf1401.all> (original
  SHA-256 `d18ba25a8496d777d4017039c1e0567533b250e4d14c2e2eb0c24e5295c12c19`,
  3008 lines).
- Kept: 2307 lines with both level energies non-negative (observed levels;
  Kurucz marks predicted levels with negative energies), a non-zero electron
  Stark constant, and 90--1100 nm. Lines are otherwise unmodified.
- SHA-256 of the decompressed subset:
  `8dc64ef88de44a91d59ec431b631149a87afca70f7a805a5b0537db4af1e0b39`.

`augment_silicon_ii_kurucz_transitions` (DZ/DAZ option
`si_ii_kurucz_transitions`) reads it in missing-transition mode and keeps only
transitions whose two levels both match existing Stout Si II levels (0.1 cm^-1
and equal weight). Stout contains the 3s3p(3Po)3d 2Fo levels at 131677.1 and
131918.8 cm^-1 but no transitions to them, so the 3s3p2 2D - 2Fo lines at
1305.59, 1309.45 and 1309.73 A (log gf +0.47, +0.32, -0.86) are missing.
The supplement adds 1471 such transitions and no levels, so partition
functions are unchanged. Kurucz-only levels are not imported; most lie far
above the first ionization limit, and lines to them (for example Si II
1399.94/1400.27 A) are not observed in hot DAZ COS spectra.

## Historical Kurucz Fe II ultraviolet missing transitions

`gf2601_uv_missing.all.xz` contains 145 otherwise unmodified rows from
Kurucz's Fe II `gf2601.all` (K13):

- Source: <http://kurucz.harvard.edu/atoms/2601/gf2601.all>.
- Full source SHA-256:
  `5a7d9d26eef58a2a32da523f8221f50cb08d3316cff20b45e22c8913e5390892`.
- Decompressed subset SHA-256:
  `0574e40ca8cae74f68f1b49488de6453841299dd0bdadc8cfc9a68a1a1e75242`
  (23,345 bytes).
- Selection: 142 missing transitions in 1189--1198 Å connecting existing
  Stout levels, plus 1357.761, 1357.796 and 1361.374 Å. All level-energy
  signs are positive; this indicates observed levels in Kurucz's convention,
  while the oscillator strengths remain calculated atomic data.

`augment_iron_ii_kurucz_uv_transitions` applies the subset to a bare/custom
atom using missing-transition
mode and leaves every existing Stout line, oscillator strength and damping
constant unchanged, including Fe II 1360.858 Å. Other ions are unchanged;
line-selection and width caches are rebuilt. Reapplication is idempotent.
`include_default_supplements=False` retains the original Stout database.

For the bundled Fe II atom, 1008 levels / 6724 transitions become 1009
levels / 6869 transitions. The sole added upper level, at 106524.400 cm⁻¹
with statistical weight 16, belongs to 1357.796 Å. It increases the ideal
Fe II partition function by 6.93e-6 fractionally at 15300 K. The level has
Kurucz observed-energy provenance; it is absent from the bundled Stout/NIST
laboratory-confirmed level set. The normal oscillator-strength and population
cuts still apply: 65 of the 142 added 1190-Å transitions exceed `f=1e-4`.

The missing 1361.374-Å component has the 1.671-eV lower excitation listed
for Fe II 1361.37 by [Jura et al. (2012), Table 1](https://arxiv.org/abs/1203.2885).
[Williams et al. (2025), Table A1](https://arxiv.org/abs/2506.19931) lists both
1361.363 and 1361.374 Å. Independent UV identifications include 1357.796 Å
in [Eaton (2008), Table 1](https://journalofastronomicaldata.be/JAD14/jad14_3/jad14_3.pdf)
and 1190.883/1193.575 Å in [Smith (2010), Fig. 2](https://arxiv.org/abs/0911.4747).
These identifications support filling missing opacity; no abundance, damping
or oscillator-strength factor is fitted to the white dwarfs. The selection
and provenance are also recorded in `gf2601_uv_missing.provenance.json`.

## Fe II oscillator strengths

`feii_melendez_barbuy_2009.dat` contains the separately selected
Melendez--Barbuy (2009) optical Fe II oscillator strengths. It is not part of
the default missing-transition supplementation policy.
