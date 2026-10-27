# J1637 excited-parent reference

This version records the 2026-10-25 O I/C II line-physics change. Its seed is
copied exactly from the checksum-verified v4 capsule: Teff, log g,
composition, column mass, optical-depth grid, base temperature, 48 layers,
both 25,000-line budgets and the prescribed 1.5% perturbation are unchanged.

Physics added: levels above the first ionization limit that belong to a
series on an excited parent term (for example O I 2p3(2D)4d) take classical
electron widths and linear-Stark n-manifolds from their parent limit (they
previously had no electron width), and the D6 atom gains the OP-normalized
O I 2p3(2D)3p -> 4d, 5s 3Do multiplets. The v1-v4 controls recover the
previous behaviour inside `research/excited_parent_series_legacy.py`.

The fixed spectrum changes by at most 45.8% in narrow native line pixels
(5488.0 A, an O I 3p'-5d' core now Stark-broadened) and by -0.507% in
integrated flux over the stored grid. The two measured warm updates do not
establish cold convergence; a separate certified public cold start with
this physics (17 iterations, 139 minutes, all-depth flux 1.0e-8, local
energy 7.0e-5) agrees with the seed temperatures to 0.1% for
tau_R = 1e-4-10 and lowers the GTC chi2 from 12716 to 11976 (5151 pixels).
Source-convergence, transfer, profile and spectral tolerances are unchanged,
including the fixed-spectrum relative tolerance of 2e-6.
