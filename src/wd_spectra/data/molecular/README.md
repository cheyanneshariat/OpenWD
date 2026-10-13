# Bundled molecular data

## H2 Lyman and Werner lines

`h2_lyman_werner_kurucz.asc.xz` is the xz-compressed concatenation of Robert
L. Kurucz's `h2bx.asc` (B-X, Lyman) and `h2cx.asc` (C-X, Werner) lists from
<http://kurucz.harvard.edu/linelists/linesmol/>, otherwise unmodified.

- SHA-256 of the original files: `h2bx.asc`
  `dd6b957650ed7fd9272b6ebdfe14d8c3ea196e31623701487d4269b3099fcb07`,
  `h2cx.asc`
  `29213618a2b127fdfb1113e88708b6dce58a89771aa6ad37d7bc9eec9e37244d`.
- SHA-256 of the decompressed bundle:
  `7c94475dd574c006005607b795f4d78703c160ac3c456a0717846ab2b4c515de`
  (2,136,450 bytes, 28,486 lines).

`wd_spectra.h2_lines.read_h2_lyman_werner_data` verifies the checksum before
reading. Kurucz marks calculated (unmeasured) level energies with a negative
sign. Only the 10,259 transitions whose two levels are both measured contribute
opacity, because predicted wavelengths are not accurate at COS resolution. All
28,486 transitions enter the radiative decay rates and fluorescent branch sums
of their upper levels.

Kurucz's gf includes the nuclear-spin weight divided by four:
gf = g_ns (2J''+1) f / 4. With this convention the radiative lifetimes of each
upper vibrational level are independent of J'. For example, B v'=0 gives
0.54 ns, compared with 0.53 ns from Abgrall, Roueff & Drira (2000).

## H2 radiative dissociation probabilities

`h2_abgrall2000_table1.dat`, `table2.dat` and `table3.dat` are tables 1-3 (B,
C+ and C- states) of Abgrall, Roueff & Drira (2000, A&AS 141, 297), from CDS
catalogue J/A+AS/141/297, unmodified. SHA-256:

- table 1: `f4afc90fcfc408a15f9ba1a1a4e6d05dae359cb0021f2cd261b0c00144b98fff`
- table 2: `04054067c6d7748d0347dd2f0e5772bf367ac4ef141286c4dc8629e82829a575`
- table 3: `c4d876db564a0ba7718bc2b1c2dcf07ff681e4d43e494a86aedcf047074d1201`

They provide the spontaneous dissociation rate into the X-state continuum of
each upper level. This rate enters the fluorescent source function and the line
damping.

## Other files

`barklem_collet_h2_partition.csv`, `h2_photoabsorption_1nm.csv` and
`h2plus_photodissociation.csv` are used by the molecular EOS and continuum
opacities; see `THIRD_PARTY_NOTICES.md`.
