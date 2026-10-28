# Hydrogen neutral defaults, 2026-10-27

From this date every hydrogen-bearing family uses the Montreal
Hummer--Mihalas neutral-hydrogen hard-sphere radius, r_n = 0.5 n^2 a_0 (as for
helium and for hydrogen in mixed H/He atmospheres), and Ali--Griem
self-broadening at every temperature. Before, the radius scale was 1 and the
public policy switched to Barklem at and above 10,000 K. The change was
reviewed against SPY/UVES Balmer profiles of 13 DAs (6450--11,500 K), the
Montreal convention (Bergeron 1991; Tremblay et al. 2010) and the observed
spectra of PG 1225-079 and G149-28; see the
[DA guide](../../../../docs/models/DA.md).

These are fixed-atmosphere outputs of the new defaults on the unchanged frozen
structures. The capture script `research/capture_montreal_h_radius_controls.py`
first recovers each previous approved spectrum inside
`previous_hydrogen_neutral_defaults` (the structure EOS reloaded there) at the
consuming test's own tolerance, then records the default. Tests check the new
outputs here and, under the same legacy scope, every older control and its
tolerance. No atmosphere is reconverged or certified, and no tolerance changed.

Largest changes in significant flux (manifest `change_from_previous`):

- DA 3000, 4000 and 20,000 K, DAB 20,000 K and six DAH targets: below 0.05%.
- DA 5000 K: 0.41% in the Hgamma core; DA 12,000 K (cubic control): 0.29%.
- DAH J1018+0111: 0.44% (default) and 1.8% at 3877 A (paper prescription);
  J1254+5612 and J1351+5419: 0.2--0.3%.
- DZ J0738+1835: 14% at 924 A in the Lyman series; the optical changes by
  less than 0.1%.
- DAZ G149-28: 5.4% at H8 (3890 A); Hgamma and Hdelta cores deepen slightly.
- DZ PG 1225-079: 13% in the Hgamma core (Barklem to Ali--Griem at 10,800 K).
  The new cores match the SPY/UVES spectrum better (Hbeta core RMS 0.164 to
  0.114, Hgamma 0.113 to 0.054 in normalized flux).

The DAH paper prescription is not exempt: its frozen paper spectra in
`tests/data/dah_paper` are recovered only under the previous defaults, and the
`dah_paper/` outputs here are the prescription's current controls.

## Cold references

The radius also enters the structure solve. Fresh protected cold starts moved
the DA 3000, 4000 and 5000 K structures by 5.5e-4, 3.6e-4 and 1.1e-4 in
pressure (limit 2e-5) and their spectra by at most 2.4e-3 on the stellar
energy scale. `cold/` holds replacement references for those three only,
captured by `research/capture_montreal_h_radius_cold_controls.py` from the
completed canary models, each a verified gray public cold start. Every other
hydrogen-bearing cold canary (DA 20,000 K, DAB 20,000 K, molecular DAB 7500,
9000 and 10,000 K, radiative DA G 76-48, DAH J1007+1237, J1254+5612 and the
J1018 mean-field case, the three DAZ cases, DZ PG 1225-079 and DAO 60,000 K)
passed against its existing reference, which is unchanged.
