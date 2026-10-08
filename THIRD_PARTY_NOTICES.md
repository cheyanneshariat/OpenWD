# Third-party notices

The BSD-3-Clause license in `LICENSE` covers the OpenWD source code. The
scientific data distributed with the package remain independently authored
works and retain their original terms and attribution.

## Observed UVES spectrum in the DZ example

`examples/data/pg1225-uves.npz` is a coadd derived from public ESO UVES data.
The data remain copyright ESO under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
`examples/data/pg1225-uves.json` preserves the original FITS headers,
source-product identities and processing information. See the
[data notes](examples/data/README.md) and the
[ESO data policy](https://archive.eso.org/cms/eso-data-access-policy.html).

The checkout-only cool-model research workflows additionally use external
HITRAN H2-He CIA and ExoMol RACPPK H2 state data. Those files are not
redistributed in this checkpoint. Follow the source attribution and data
terms linked in [the research data instructions](research/cool_models/README.md).
The small bundled HNC cache and DAB checkpoints there are OpenWD-generated
research outputs, not published reference spectra; their source models,
provenance and limitations are documented alongside them.

## Allard neutral-hydrogen/proton Lyman profiles

The temperature-dependent Ly-alpha, Ly-beta, and Ly-gamma tables in
`src/wd_spectra/data/runtime/allard_data` were supplied by Nicole Allard and
are redistributed in OpenWD with the author's permission. Scientific uses
should cite the applicable Allard line-profile calculations. The historical
`laquasi.dat`, `lbquasi.dat`, and `lgquasi.dat` files are the versions
distributed with TLUSTY 205.

## Tremblay--Bergeron hydrogen Stark profiles

The Doppler-convolved Lyman, Balmer, Paschen, and Brackett profile tables in
`src/wd_spectra/data/stark` are licensed under CC BY 4.0. Cite Tremblay &
Bergeron (2009), ApJ, 696, 1755.

## Stout atomic line data

The Stout Atomic Line List files are copyright Peter van Hoof, Royal
Observatory of Belgium, and licensed under CC BY 4.0. The original license is
retained beside the tables in
`src/wd_spectra/data/runtime/cache/metal-opacity/atomic-line-list/stout`.

## Hot-star NLTE atomic inputs

OpenWD has permission to redistribute the release copies of the hot-star
atomic inputs installed under `src/wd_spectra/data/runtime/cache/`. These
files remain scientific data from their named projects and are not covered by
OpenWD's BSD source-code license:

- Curtin CCC database shell-resolved electron--hydrogen excitation and
  ionization cross sections in `ccc/e-H_XSEC_LS.zip`.
- TLUSTY 200 source coefficients, its 14-level He I atom, and the C III,
  C IV, O IV, O V, and O VI model atoms under `tlusty-source/` and
  `tlusty-atoms/`. Credit Hubeny and Lanz and the Opacity Project as
  appropriate for the selected records.
- TLUSTY's 24-level He I atom (`tlusty-atoms/he1.dat`, the OSTAR2002/BSTAR2006
  atom; every LS term through n = 4), obtained unmodified from the TLUSTY
  website (tlusty.oca.eu, Tlusty2002/database/atom/he1.dat; SHA-256
  86f265a6bad0dcfa45202153e68b5090f8e284f5caac061dd7aea0d3c0ab7384). Credit
  Hubeny and Lanz (and Lanz & Hubeny 2003, 2007 for the OSTAR2002/BSTAR2006
  grids that use it).
- Tübingen Model-Atom Database (TMAD) C III--V and O III--VII structure and
  formal-synthesis atoms under `tmad-atoms/`. Credit T. Rauch and the TMAD
  contributors; the files retain their embedded authorship headers.
- SIROCCO-distributed TOPbase O VI level and photoionization data under
  `sirocco-atomic/`.
- CHIANTI O VI effective collision strengths under `chianti/o_6/`. Credit
  the CHIANTI collaboration and the sources identified by that database.

Scientific publications should cite the original databases and atomic-data
papers used by the applicable model. Redistribution permission does not alter
their attribution or ownership.

## D6 level-resolved photoionization inputs

OpenWD has permission to redistribute the release copies of the D6
Opacity Project/TOPbase inputs installed under
`src/wd_spectra/data/runtime/cache/`, on the same terms as the hot-star
inputs above. Their source URLs and SHA-256 checksums are pinned in
`wd_spectra.d6` (`D6_TOPBASE_FILES`, `D6_TLUSTY_TOPBASE_FILES`,
`D6_TLUSTY_RAP_FILES`). They remain scientific data from their named projects
and are not covered by OpenWD's BSD source-code license:

- SIROCCO-distributed TOPbase C II and O II level and photoionization files
  (`sirocco-atomic/c_2_*.dat`, `o_2_*.dat`), from the SIROCCO repository at
  commit `e3a8c4db`.
- Public TLUSTY model atoms (Hubeny and Lanz) with Opacity Project cross
  sections for C I, O I, Ne I, Mg I–II, Al II, Si I–II and S II, and the
  Fe II `.rap` companion file (`tlusty-atoms/`).

Credit the Opacity Project (Cunto et al. 1993) and the TLUSTY and SIROCCO
distributions.

## Hollands et al. (2025) J1637 digitization

`src/wd_spectra/data/validation/hollands2025_j1637_figure1.npz` records the
vector paths of Figure 1 of Hollands et al. (2025, MNRAS 541, 2231; DOI
[10.1093/mnras/staf950](https://doi.org/10.1093/mnras/staf950)). It is a
regression target derived from the published figure, not the authors'
numerical data; cite the paper when using it.

## H2db magnetic hydrogen data (DAH)

`src/wd_spectra/data/runtime/cache/h2db/h2db_balmer_subset.npz` is a subset of
the Hydrogen Database of C. Schimeczek and G. Wunner (DaRUS,
doi:[10.18419/DARUS-2118](https://doi.org/10.18419/DARUS-2118); method in
Comput. Phys. Commun. 185, 614, 2014), distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It contains the
Balmer transitions and all stationary-state energies for beta <= 3, copied
without modification (transition energies and dipole strengths stored as
single precision). `tools/build_h2db_subset.py` rebuilds it from the public
archive. Cite Schimeczek & Wunner when using DAH models.

## DAH Kurucz/Griem profile prescription

`src/wd_spectra/kurucz_griem.py` implements the equations and historical
constants of Kurucz (1970), SAO Special Report 309, section 5.14 and the
`STARK` listing on printed page 243, based on Griem's Stark theory.
The original approximation is normalized numerically in frequency and
combined with OpenWD's full oscillator strengths and occupation factors.
It is not the later HPROF4 routine or the unpublished Jordan/Moss code.
Cite [Kurucz (1970)](https://articles.adsabs.harvard.edu/pdf/1970SAOSR.309.....K)
and [Moss et al. (2024)](https://doi.org/10.1093/mnras/stad3825) when using
this prescription. The bundled paper controls in `tests/data/dah_paper`
are OpenWD predictions, with provenance recorded in their manifest.

## DAH observed validation spectra

`src/wd_spectra/data/validation/dah/` holds resampled copies of public
spectra used only for validation: SDSS/BOSS spectra (SDSS Collaboration; see
the SDSS data policy), the Gianninas GH Leo spectrum from the Montreal White
Dwarf Database, and model curves of Hardy, Dufour & Jordan (2023, MNRAS 520,
6111) digitized from their CDS figures (regression targets derived from the
published figures, not the authors' numerical models). Survey spectra of the
Hardy targets are de-reddened with MWDD E(B-V). Cite the original surveys and
papers when using them. The weak-field GD 9 and G 76−48 controls come from
public ESO SPY/UVES spectra. The documentation's eight-panel figures also
include SDSS/BOSS J1007+1237 (plate 5328, MJD 55982, fiber 66). The unscaled
and smoothly rescaled figure variants are labelled separately in the guide;
the smooth correction is a model-derived comparison operation.

## Other scientific tables

The DQ constitutive data in `src/wd_spectra/data/dq` include derived ExoMol
12C2 8states cross sections and partition functions, Hornkohl/Parigger Swan
line records, and an OpenWD-computed dense-helium correction grid. They are
scientific input tables, not model atmospheres or observed spectra.
Original data authors retain their rights; these data are not relicensed as
OpenWD source code. Checksums and processing provenance are in `manifest.json`,
`report.json`, and the NPZ metadata. Scientific use should credit:

- ExoMol 8states C₂ data (Yurchenko et al.), including the updated state energies
  identified in the opacity-table provenance.
- Parigger et al. (2015), DOI [10.1016/j.sab.2015.02.018](https://doi.org/10.1016/j.sab.2015.02.018).
  The supplied dimensional strength convention was not recovered: OpenWD uses
  one rotationless-band calibration to Brooke's `A(0,0)=7.626e6 s^-1`, preserves
  relative strengths, and does not fit stellar spectra to set this scale.
- Brooke et al. (2013), [arXiv:1212.2102](https://arxiv.org/abs/1212.2102).
- Cooper (1979), [NASA TM-78574](https://ntrs.nasa.gov/citations/19790013711),
  and Nicholls (1965), [Franck–Condon factors, Table 6](https://pmc.ncbi.nlm.nih.gov/articles/PMC6716003/),
  for the historical C₂ C–A estimate in `c2-ca-historical.npz`. OpenWD uses
  the measured squared electronic moment 0.93 atomic units (reported
  uncertainty 0.18), assumes it constant across the included bands, and
  combines the published band constants with ExoMol lower-state populations.
  Its finite-bin rigid-rotor envelope is an approximation, not a reproduction
  of a modern rotational line list or a validated dense-helium pressure profile.
- Iglesias et al. (2002), DOI [10.1086/340689](https://doi.org/10.1086/340689),
  and Blouin et al. (2018), DOI [10.3847/1538-4357/aad4a9](https://doi.org/10.3847/1538-4357/aad4a9),
  for dense-continuum/refractivity prescriptions. Numerical interpolation and
  extrapolation choices remain OpenWD approximations documented in the data.

The derived ExoMol file `c2-8states-r15000.npz` is distributed under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/),
following the [ExoMol data licence](https://www.exomol.com/data/licence/).
OpenWD's changes comprise cross-section binning with updated state-energy
differences, explicit finite-state partition sums, separation of the Swan
component, and rotational-overlap redistribution; these contributions to
that data file use the same CC BY-SA 4.0 licence. Credit Yurchenko et al.
(2018, MNRAS 480, 3397) and McKemmish et al. (2020, MNRAS 497, 1081).
This data licence does not replace the separate BSD licence for OpenWD code.

The additional `c2-ca-historical.npz` table also uses CC BY-SA 4.0: it derives
its populations and partition function from those ExoMol data. Credit both
ExoMol papers above and Cooper/Nicholls. OpenWD's additional processing is
the historical-band envelope integration; `tools/build_dq_ca_historical.py`
and `tools/package_dq_ca.py` reproduce the physical arrays without stellar
observations. The NPZ provenance retains the original research description
and input hashes; numerical release qualification does not remove its stated
spectroscopic approximations.

The new default `c2-ca-2024.npz` likewise retains CC BY-SA 4.0 for its
ExoMol-derived populations and OpenWD envelope processing. Its band strengths
are numerical Einstein coefficients from [Lino da Silva (2024), slide 20,
lower table](https://indico.esa.int/event/466/contributions/9848/), using the
transition moments of [Babb, Smyth & McLaughlin (2019)](https://arxiv.org/abs/1904.07831).
Credit these sources as well as the ExoMol papers and historical band-constant
sources above. OpenWD uses 63 matched bands, converts A to oscillator strength,
and retains the previous approximate rotational envelopes. The source PDF hash,
coefficient table, conversion, and input identities are embedded in the NPZ;
`tools/build_dq_ca_2024.py` reproduces its physical arrays without observations.

`swan-completed.npz` retains all original Hornkohl/Parigger calibrated lines
and appends ExoMol transitions outside their per-band v/J coverage. The ExoMol
additions and OpenWD's processing of those data retain CC BY-SA 4.0; this does
not relicense the independently supplied Hornkohl records. Credit both sets of
authors and Brooke's absolute band-rate anchor. Input/output checksums and
selection details are in `manifest.json`; `tools/build_dq_swan_completed.py`
reproduces the selection from the audited ExoMol branch cache. No stellar
parameters or opacity multipliers are fitted to form either default table.

The runtime data directory also contains evaluated NIST ASD strong-line data,
Verner et al. photoionization fits, CHIANTI Ca II collision strengths,
Beauchamp He I and Schoening/SYNSPEC He II profiles, Becker et al. He-REOS.3,
and published atomic/molecular continuum tables. Source identifiers and
checksums are recorded in the corresponding OpenWD readers. Redistribution of
these numerical data does not place them under the OpenWD source-code license.

## Niobium atomic data

`src/wd_spectra/data/runtime/cache/metal-opacity/niobium/` contains unmodified
NIST ASD 5.12 exports (Kramida, Ralchenko, Reader and the NIST ASD Team,
doi:10.18434/T4W30F), retrieved on 2026-10-07: the Nb I--VI energy levels,
the Nb IV transition probabilities, which are almost all from Tauheed & Reader
(2005, Phys. Scr. 72, 158), and the Nb ionization energies. It also contains
a hand transcription of the Nb III transition probabilities in Table 7 of
Nilsson et al. (2010, A&A 511, A16, doi:10.1051/0004-6361/200913574). The
article itself is not redistributed. Query URLs and checksums are in
`NIOBIUM_ATOMIC_DATA_FILES` in `metals.py`. Cite NIST ASD and the original
sources when using these data.

## Rauch et al. Zn and Cu oscillator strengths

`src/wd_spectra/data/runtime/cache/metal-opacity/rauch-zn-cu/` contains
unmodified CDS copies of J/A+A/564/A41, the Zn IV and Zn V HFR oscillator
strengths of Rauch et al. (2014, A&A 564, A41,
doi:10.1051/0004-6361/201423491), and Cu IV--VI rows of the Tuebingen
Oscillator Strengths Service (TOSS) table `toss.data` (Rauch et al. 2020,
A&A 637, A4, doi:10.1051/0004-6361/201936620; TOSS doi:10.21938/3i01isnuCODnh1zjbCvuwA),
retrieved on 2026-10-07. CDS permits scientific use with citation of the
original authors and publication; TOSS requests the acknowledgement quoted in
that directory's README. Neither states an explicit open licence, and these
data are not relicensed under OpenWD's source-code license. Query URLs and
checksums are in `RAUCH_ZN_CU_ATOMIC_DATA_FILES` in `metals.py`.

## Korg.jl Stancil (1994) table transcription

The numerical Stancil (1994) H2+/He2+ opacity-table transcription used as
the source for `src/wd_spectra/helium_molecular.py` is adapted from Korg.jl.

Copyright (c) 2021, Adam Wheeler. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Original scientific data: P. C. Stancil, *Continuous absorption by He2+ and
H2+ in cool white dwarfs*, ApJ 430, 360 (1994), DOI 10.1086/174411.
