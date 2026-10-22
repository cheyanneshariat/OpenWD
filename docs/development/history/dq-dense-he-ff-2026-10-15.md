# Dense-He He⁻ free-free correction δ_ff (2026-10-15)

Question: is OpenWD's dense-helium He⁻ ff correction δ_ff too strong at
ρ ≳ 0.3 g cm⁻³? It was. There were two causes. The Born kernel was not one of them.

## 1. The Born kernel is correct; Iglesias et al. (2002) Figs. 2–3 are not reproducible

Independent derivation (Fermi golden rule, acceleration-form dipole matrix
element (ε̂·q)V(q), Maxwellian electrons): the rate at momentum transfer q is
∝ q³|Ṽ(q)|² exp[−(q/2 − E/q)²/2kT] dq (atomic units), with the 1D Maxwellian
of the electron velocity component along q enforcing E_f − E_i = ħω. This is
Iglesias, Rogers & Saumon (2002, ApJ 569, L111) Eq. 3.4 as printed, *with*
its 1/k prefactor. Blouin et al. (2018) Eq. 5 drops the 1/k in transcription.
`dense_helium_continuum._weighted_structure` integrates k⁴|Ṽ|²exp d(ln k),
which is identical, and `tests/test_dense_helium_continuum.py` checks it
against an independent quadrature.

Fig. 3 of Iglesias, digitized (3240 K, 1.23 g cm⁻³, q = k aᵢ, aᵢ = 2.057 a₀):

| | their Fig. 3 | ours |
|---|---|---|
| S(q) first peak | q = 4.69, S = 1.46 | 4.42, 1.62 (PY) |
| I₀ peak, u = 4 | q = 1.41 | 0.80 |
| I₀ peak, u = 16 | q = 2.38 | 1.27 |

Their S(q) axis is physical: the hard-sphere estimate for He at these
conditions (σ_eff ≈ 1.65 Å, η ≈ 0.48, kσ ≈ 7.2) gives q ≈ 4.6. Their I₀
peaks are not. An electron that absorbs ħω = 16 kT from a 3240 K Maxwellian
leaves with q ≈ √(2mħω) ≈ 1.18/aᵢ. q = 2.38/aᵢ would need ≈ 4ħω of final
kinetic energy, which is suppressed by e⁻⁹. Their Fig. 2 also contradicts their
Fig. 3. With their own digitized S(q):

| u | 0 | 1 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|
| Fig. 2 δ_ff | 0.065 | 0.066 | 0.077 | 0.105 | 0.143 | 0.190 |
| Eq. 3.4, their S | 0.071 | 0.071 | 0.075 | 0.080 | 0.087 | 0.095 |
| I₀ shifted to Fig. 3 peaks | 0.103 | 0.105 | 0.127 | 0.174 | 0.242 | 0.334 |

No single distortion (photon energy ×2 or ×4, T ×2, Å/a₀ mix-up) reproduces both
figures. δ_ff(u→0) = S(0) = δ_sc in Fig. 2 matches the physical kernel and
excludes the Fig. 3 placement. The steep rise in Fig. 2 is therefore not a
target. Kowalski, Mazevet & Saumon (2006, cond-mat/0511400) give no absolute
δ_ff values: they report QMD absorption curves only.

## 2. Cause A: the Percus–Yevick structure factor at high density

δ_ff ≈ S(k aᵢ ≲ 1.5), so the fluid structure sets the answer. The v1 table used
OZ/Percus–Yevick with the Ross & Young (1986) exp-6 potential. PY is not
thermodynamically consistent: at 3000 K and 1.78 g cm⁻³ its virial-route
compressibility differs from its OZ route by 2.6×.

S(0), 3000 K:

| ρ (g cm⁻³) | He-REOS.3 (kT/m)(∂ρ/∂P)_T | PY | Rogers–Young | HNC | MC (Aziz; Rohrmann 2018 fit) |
|---|---|---|---|---|---|
| 0.316 | 0.354 | 0.377 | 0.378 | 0.389 | |
| 1.000 | 0.088 | 0.066 | 0.085 | 0.098 | 0.071 |
| 1.778 | 0.040 | 0.013 | 0.031 | 0.039 | 0.020 |

Exact canonical Monte Carlo for the *same* Ross–Young potential (N = 1372,
3×10⁴ sweeps, S(k) on box wavevectors, block errors; scratch code
`mc_exp6.c`) at 3000 K and 1.0 g cm⁻³, over the q range that δ_ff samples
(0.35 ≤ q ≤ 3): RY agrees to 1.2% (χ²/N = 7). PY is 24% low (χ²/N = 1765) and
HNC 13% high (χ²/N = 1142). At 1.778 g cm⁻³ (MC S(q→0) ≈ 0.032), RY agrees to 0.4% (χ²/N = 2.3), PY is 2.3× low (χ²/N = 14197) and HNC 20% high. The residual gap between RY and REOS (0.031 vs 0.040) is the pair potential, not the closure.

Fix: the Rogers & Young (1984, PRA 30, 999) closure,
`_dq/dq_rogers_young_fluid.py`. Its mixing parameter α is fixed at every
state point by equating the virial and fluctuation compressibilities. This is
an internal consistency condition and nothing is fitted to data. Above about
10⁴ K the virial includes the hard-wall contact term at the exp-6 core (the
inner stationary point, φ = 7.6 eV, σ = 0.718 Å). The solution reaches
3.25 g cm⁻³ at every table isotherm and stops near 3.37 g cm⁻³, where the
exp-6 potential is used beyond its turnover. The table now ends at the next
v1 knot, 2.913 g cm⁻³ (it was 2.513). J1803 had requested 2.55.

## 3. Cause B: Blouin's 1/|ε(ω)|² double-counts dielectric dressing

Blouin et al. (2018) Eq. 4 divides δ_ff by |ε(ω)|² = N⁴, where ε is the k = 0
optical dielectric constant. The DQ ray kernel also divides the extinction by N,
so the net factor was 1/N⁵. That is 2.2 at 1.23 g cm⁻³ (ε = 1.48) and ≈ 4 at
2.5 g cm⁻³.

* Iglesias' own Fig. 2 has δ_ff(0) = S(0) = 0.065. With 1/|ε|² it would be
  0.028, so their curve excludes this factor. Their Rosseland mean (Eq. 4.6)
  applies only δ_ff/N.
* Free-carrier absorption by neutral scatterers in a transparent dielectric:
  the photon field normalization (1/ε) times the group-velocity conversion (N)
  gives 1/N. The carrier is driven by the macroscopic field, with no Lorentz
  local-field enhancement for a delocalized electron. A short-range e–He
  potential at q ≈ 0.3–1.3 a₀⁻¹ is not screened by the k = 0 optical ε.

Decision (user, 2026-10-15): drop 1/|ε|². This is a deliberate departure from
Blouin et al. (2018) Eq. 4. The shipped table therefore uses `--dielectric propagation`: no 1/|ε|² in
δ_ff, with the transport kernel's 1/N as the only dielectric factor. Both the
bare ratio (`correlation_ratio`) and ε (`dielectric`) are stored, so
Blouin's prescription is `correlation_ratio/dielectric**2`
(`tools/build_dq_dense_continuum.py --dielectric blouin`).

## Net change in δ_ff (new / v1)

| T (K) | 0.3 | 0.6 | 1.0 | 1.23 | 1.8 | 2.5 g cm⁻³ |
|---|---|---|---|---|---|---|
| 3240 | 1.26 | 1.62 | 2.42 | 3.15 | 6.6 | 17.9 |
| 4500 | 1.26 | 1.59 | 2.26 | 2.84 | 5.2 | 11.7 |
| 6000 | 1.26 | 1.56 | 2.15 | 2.62 | 4.4 | 8.5 |

The ratio is almost independent of wavelength. At 3240 K and 1.23 g cm⁻³, the
new δ_ff runs from 0.073 (12000 Å) to 0.086 (3000 Å).

## Atmospheres (cold, fixed published parameters, C kernel, one at a time)

Frozen tree `release/OpenWD-dq-heff` (7f4423d plus these changes, table
sha256 d6337a48…), `run_dq3.py`, 40→43 depths, standard quality,
OMP/OPENWD threads = 1, run one after another. SDSS comparison uses
`scoring.py` (10 Å FWHM, 3850–6720 Å, absolute flux from published mass,
log g and parallax, nothing scaled). J1311 is compared with the digitized
BD19 figure. "v4" is the 7f4423d run with the v1 table.

| Object | Cold result | Time | Flux ratio | frac_rms v4 → new | 3900–4300 | 4300–4700 | 4700–5170 | 6200–6700 |
|---|---|---|---|---|---|---|---|---|
| J1803 (4400 K) | converged, certificate verified (first ever) | 17.5 min | 1.0013 | failed → 0.286 | 0.96 | 0.37 | 0.45 | 1.06 |
| J0804 (5364 K) | converged, verified | 29.4 min | 1.0002 | 0.170 → 0.146 | 1.11 → 0.99 | 0.81 → 0.80 | 0.77 → 0.80 | 0.95 → 1.06 |
| J1311 (5529 K) | converged, verified | 44.7 min | 1.0013 | 0.299 → 0.332 | 1.34 → 1.37 | 1.04 → 1.09 | 1.07 → 1.12 | 1.10 → 1.14 |

Band columns give model/observed mean flux.

* J1803 now converges, and its deepest layer is 2.55 g cm⁻³, inside the table.
  Its photosphere is still dense: ρ = 1.06, 1.43 and 1.73 g cm⁻³ at τ_R = 0.09,
  0.4 and 5. With the bounded carbon wings, the optical continuum there is He⁻ ff
  plus scattering. C lines are below 10⁻⁸ cm² g⁻¹ against He⁻ ff ≈ 1×10⁻⁴.
  The new δ raised κ_ff by ≈ 3.7× at τ = 0.4, but the REOS isotherm is so stiff
  there (P rises only 3× from τ = 0.09 to 5) that the density barely moves. The
  denser photosphere implied by Blouin & Dufour (~0.7 g cm⁻³) is therefore
  not a δ_ff question. The parent session's diagnostic `breakdown.py` calls the
  generic metal-line routine without `red_wing_boltzmann`, so its "C lines
  <2000" column (≈10⁻³) overstates the production opacity by ~10⁵. Fixed in
  this session's copy.
* J0804: the photosphere is unchanged (ρ ≈ 0.58 at τ = 0.1, 0.69 at τ = 0.5).
  The continuum slope improved (blue 1.11 → 0.99) and frac_rms fell 14%.
* Remaining discrepancy in both cool stars: the Swan bands are too deep (J1803
  0.37–0.45, J0804 0.80). This is the open Swan-band issue, independent of
  δ_ff.
* J1311 (C-rich, lower photospheric density): the shape improved
  (shape_rms 0.162 → 0.152), but the absolute level is ~15% high against the
  digitized figure.

## Files

* `src/wd_spectra/_dq/dq_rogers_young_fluid.py`: RY closure and consistency solver.
* `tools/build_dq_dense_continuum.py`: sequential, checkpointed table builder
  (274 s).
* `src/wd_spectra/data/dq/correction.npz`: new table, sha256 d6337a48….
* Tests: `tests/test_dq_rogers_young_fluid.py`, `tests/test_dq_dense_continuum_table.py`.
* `tests/test_dq_spectral_regression.py` (J1235, J1311) already failed on
  7f4423d and needs user-approved recapture.
