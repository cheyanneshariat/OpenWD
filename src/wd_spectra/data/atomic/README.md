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

## Fe II oscillator strengths

`feii_melendez_barbuy_2009.dat` contains the separately selected
Melendez--Barbuy (2009) optical Fe II oscillator strengths. It is not part of
the default Cr II supplementation policy.
