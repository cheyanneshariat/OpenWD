# Hot DA/DAO trace-metal data

Inputs of the experimental NLTE trace-metal module (`wd_spectra.hot_trace_metals`)
and of the G191-B2B benchmark drivers in `research/`. Large text tables are
xz-compressed; `research/hot_daz_data.py` decompresses them on demand. Every
checksum below refers to the uncompressed file.

| Directory | Contents | Source | Checksums |
| --- | --- | --- | --- |
| `atomic/` | `verner95.dat` (Verner & Yakovlev 1995 partial cross sections); `gfFUV99.dat.gz` (Kurucz measured-level FUV lines distributed with Synspec) | https://www.pa.uky.edu/~verner/dima/photo/table1.dat ; https://tlusty.oca.eu/tlusty/Synspec49/data/gfFUV99.dat.gz | `research/hot_trace_composition.py:DATA_FILES` |
| `kurucz/` | `gf2603.pos` ... `gf2806z.pos`: Fe/Ni IV–VII measured-level line lists, used to supplement missing EUV transitions | http://kurucz.harvard.edu/atoms/26xx and /28xx | `kurucz/SHA256SUMS` |
| `chianti/recombination/` | `fe_5`–`fe_8`, `ni_5`–`ni_8` `.rrparams`/`.drparams` (Shull & van Steenberg 1982, Badnell 2006; Mazzotta et al. 1998) | CHIANTI database (see `PROVENANCE.txt`; `fe_8`/`ni_8` from the CHIANTI 11.0.2 tarball) | `chianti/recombination/SHA256SUMS` |
| `chianti/carbon/` | `c_2`–`c_4` `.elvlc`/`.scups` (`c_2` added for sdB C II) | CHIANTI (sohoftp mirror) | `chianti/carbon/manifest.json` |
| `chianti/oxygen/` | `o_3`–`o_6` `.elvlc`/`.scups` | CHIANTI (sohoftp mirror) | `chianti/oxygen/manifest.json` |
| `observations/g191-b2b/` | MAST HLSP `wd-linelist` coadds (FUSE, STIS E140H, E230H) and line lists of Preval et al. (2013) | https://archive.stsci.edu/prepds/wd-linelist/ | `observations/g191-b2b/SHA256SUMS` |

The CHIANTI GitHub repository from which the original recombination files were
taken no longer exists; the files are identical to those in the CHIANTI 11.0.2
database release.
