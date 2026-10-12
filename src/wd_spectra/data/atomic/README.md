# Bundled atomic supplements

## Kurucz Cr II

`gf2401.all.xz` is an xz-compressed, otherwise unmodified copy of Robert L.
Kurucz's full Cr II list:

- Source: <http://kurucz.harvard.edu/atoms/2401/gf2401.all>
- SHA-256 of the decompressed file:
  `6f0c4d0e01421fb0549ddbcf5649af391ca9b930e4e83e693e662d5874505205`
- Uncompressed size: 14,588,693 bytes.

`read_stout_atomic_database` applies it by default whenever Cr II is present,
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

## Kurucz Si II

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

## Fe II oscillator strengths

`feii_melendez_barbuy_2009.dat` contains the separately selected
Melendez--Barbuy (2009) optical Fe II oscillator strengths. It is not part of
the default Cr II supplementation policy.
