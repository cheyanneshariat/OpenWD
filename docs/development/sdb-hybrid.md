# Experimental hot-subdwarf (sdB) H/He models

Status: research workflow on branch `codex/sdb`, not a public preset. Metals
are not yet included in the structure. Hot sdOs (above about 40 kK) are out of
scope for now.

## Method

The sdB models follow the hybrid LTE/NLTE recipe used for B-type stars and
sdBs (Przybilla et al. 2006; Irrgang et al. 2018): an LTE line-blanketed
structure, then NLTE populations and line formation on that fixed structure.

1. **LTE H/He structure.** `research/sdb_lte_hhe.py` runs the public DAB preset
   (`compute_dab`) at fixed Teff, log g and He/H. The preset has no gravity
   limits. Its line physics (Tremblay & Bergeron 2009 Balmer profiles, the
   Tremblay 2026 He I tables, and the He II tables) is the same as in the NLTE step.
2. **NLTE H/He populations on the fixed structure.**
   `research/sdb_hybrid_nlte.py` rebuilds the LTE (T, P) structure in the
   restricted hot H/He atom (`HotNLTEModel`). It solves the fixed-temperature
   statistical equilibrium with the formal-solution radiation field and
   synthesizes the spectrum. It also writes the same synthesis with J = B
   populations (`spectrum-lte.npz`), which isolates the NLTE effect.

The self-consistent DO/DAO temperature + NLTE solver
(`solve_hot_nlte_atmosphere`) is not used. At sdB densities its full-NLTE
stage is nearly singular in the outer layers (trace He III and weakly
determined optically thin temperatures) and does not converge.

## Opt-in solver options added for sdBs (defaults unchanged)

- `HotNLTEModel.solve_populations(..., accelerated_lambda=True)`: multilevel
  accelerated lambda iteration (`_hot_mali.py`). Line rates are preconditioned
  with the profile-weighted diagonal Lambda* times the line's share of the
  extinction, in the Rybicki & Hummer (1991) form. The fixed point is unchanged.
  Without it, the fixed-point iteration falsely converges through the
  optically thick H/He transitions: in a 30 kK test atom it was 56% from the
  solution after 3000 iterations, while its per-iteration change was still
  small. With MALI the grid below converges in 45-308 iterations.
- `HotNLTEModel.helium_conservation_row="dominant"` (solver argument
  `conservation_row`): particle conservation replaces the equation of the
  most populous helium state instead of the He III equation. In cool sdB
  surface layers (He III/He ~ 1e-18), replacing the He III equation either
  makes the system singular to roundoff or silently returns wrong
  populations. At J = B, a gray 16 kK atmosphere gave departures off by up
  to 20%. The sdB driver uses `dominant`.
- `HotNLTEModel.helium_i_atom`: the coupled helium solver takes a He I
  model atom (`helium_i_atom.py`). The default is the TLUSTY 14-term atom,
  with bit-identical results (`research/helium_atom_refactor_harness.py`, 140
  arrays). `read_tlusty_helium_i_atom` loads TLUSTY's 24-term `he1.dat`, which
  resolves every LS term through n = 4. Its first 19 terms use the COLLHE
  collision fits term by term and the l-resolved OP photoionization fits.
  For sdB optical He I lines it changes core depths by at most 0.3% at 30 kK
  and by 0.5-1.5% in a few lines at 23 kK, so the 14-term atom remains the
  sdB default.

Also fixed in the shared coupled helium solver: He III is no longer recovered
by subtraction from the total. That subtraction lost all precision for a trace
ion. Particle conservation is now enforced by a common, conservation-checked
rescaling.

## Validation

- **HD 4539** (Schneider et al. 2018: 23,200 K, log g 5.20, log He/H -2.27;
  parameters fixed, not fitted), compared with co-added ESO X-shooter UVB/VIS
  spectra (`research/sdb_coadd_xshooter.py`,
  `research/sdb_compare_hd4539.py`). LTE H/He already fits the Balmer lines
  and most He I lines. NLTE halves the He I 5876/6678 core residuals (rms
  0.035 -> 0.018), the known NLTE strengthening of these lines.
- **Robustness grid** (`research/sdb_grid_check.sh`,
  `research/sdb_grid_report.py`, `research/sdb_grid_plot_spectra.py`):
  12 points at 20-40 kK, log g 5.0-6.2 and log He/H -4 to -1, all at 40 depths.
  - Every LTE structure and NLTE population solve converged; each model took
    9-21 min on one core.
  - One unpreconditioned update changes the converged populations by
    <= 9e-5 (`research/sdb_fixed_point_check.py`).
  - The trends are physical: He I peaks near 25 kK; He II 4686 appears at
    30 kK and reaches about 20% depth at 40 kK; Balmer wings broaden with
    gravity; the He-poor and He-rich spectra behave as expected.
  - NLTE effects grow with Teff: He I 5876 is 20-30% deeper than LTE at
    20-30 kK and about 2x deeper at 40 kK. Balmer cores are substantially
    deeper than LTE at 35-40 kK.

## Limitations and next steps

- No metals in the structure or line formation yet (next step). Iron-group
  blanketing needs Fe III-VI line opacity in the LTE structure.
- In both He I atoms, n = 5 is a superlevel coupled to n = 4 by
  Van Regemorter-type closures.
- Radiation pressure is not included in the hydrostatics. It is negligible
  for typical sdBs but is needed for hot sdOs.
- Developed at 40 depths. Production-depth (80) timing is still to be measured.
