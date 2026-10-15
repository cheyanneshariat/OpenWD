# Planned grid updates

The recent updates call for a DZ rerun, three DAZ replacements, and a small
optional H2 UV grid. They do not require rerunning the whole pure DA grid.
The required existing-coordinate replacements are verified below. Optional grid
expansions remain proposed and have not passed grid validation.

## Work started

The [DZ rerun](../dz-rerun-2026-10-12/README.md) finished all 322 existing
production requests under tested source `1386029`. Native checks passed for 312;
seven timed out, two failed convergence checks and one worker crashed. Selected
coverage is 321/322 because nine accepted older spectra remain where fresh
attempts failed. Old attempts and the four earlier planning deferrals are preserved.

A six-request local pilot used current main
`e09e7daeaa0695d34e14b55c557ca770fad326af`. Its first production DA at
12,000 K/logg 8, with H2 off, hit the 4 GiB RSS guard after 935.24 seconds.
The sampled peak was 5.546 GiB. No verified spectrum was adopted. Five
follow-on requests were never started. The DA resource stop is retained;
the model is not retried automatically. Nine H2 and DA/DB fixed-spectrum
checks passed before dispatch. Fixed-spectrum checks do not certify an atmosphere.

The user then authorized all required replacements overnight. An independent
three-request DAZ tier has verified replacements for G29-38, G149-28 and GALEX J1931 using the same
checked source. It preserves the old standard resolution, temperatures,
gravities, full abundance patterns and alpha, with the new default metal
profiles. Its wavelength array exactly matches the three old selected spectra.
Two additional DAZ fixed-spectrum checks passed before dispatch, bringing the
preliminary total to eleven. One numerical thread runs
at a time during that completed tier, with two-hour wall and 6 GiB RSS guards per request. Case-specific
failures remain explicit; source or runtime identity failures stop the branch.

Local work adds zero charged NERSC node-hours. An earlier Python environment
import failure stopped before any model calculation; its logs are retained.
The corrected launcher uses the checked validation environment. The morning
handoff reported unfinished work honestly; the last DZ allocation settled later.

[All three required DAZ replacements](../daz-replacements-2026-10-12/README.md)
are now verified and archived. Their matched old/new plots and remaining
limits are recorded. The DZ allocations are also terminal.

## Staged scope

| Stage | Coordinates and requests | Purpose |
| --- | --- | --- |
| DZ replacement | 322 existing requests: 312 passed, 7 timeouts, 2 nonconvergences, 1 interruption | Update the existing metal-pattern grid while preserving old attempts. |
| DAZ replacements | 3 existing anchors, 3 cold requests | Replace spectra affected by the updated metal physics. |
| DA H2 UV supplement | 15 grid nodes + 2 independent tests; 34 paired cold requests | Test H2-off/on spectra across the cool hydrogen regime. |
| DAZ first patches | 27 grid nodes + 6 independent tests; 55 cold requests | Test T/g interpolation at three fixed abundance patterns; pair H2 off/on for the cool patterns. |
| Conditional DAZ metal-scale extension | 81 nodes + 12 independent tests in total; 100 additional cold requests after reusing the first patches | Test interpolation in a shared metal scale at fixed relative abundances. |

The three DAZ replacements and the first DA/DAZ grid stages total 92 cold
requests. G29-38's standard H2-on pilot adds one diagnostic request. These
counts describe a proposed scope, not an approved resource estimate. Reuse
requires identical source, configurations, wavelength arrays and verified
native products. Standard DAZ pilot results cannot be counted as production
grid nodes.

The DA H2 temperatures are 10,000–14,000 K in 1,000 K steps, with logg 7.5,
8.0 and 8.5. Independent tests are 11,500 K/logg 7.75 and 13,500 K/logg 8.25.
Alpha remains 0.7. Above 12,000 K, H2 can also enable molecular chemistry in
the structure; compare that change explicitly. Use the native 900–30,000 Å
grid plus 1150–1800 Å sampled every 0.05 Å. Check a representative at 0.025 Å.

Each DAZ patch uses the anchor temperature plus/minus 1,000 K and gravity
plus/minus 0.25 dex. The anchors are G149-28 (8600 K/logg 8.1), G29-38
(11,820 K/8.4), and GALEX J1931 (20,890 K/7.9). Two fresh interior tests per
pattern check interpolation. The conditional abundance stage adds shared
metal scales of −0.5, 0 and +0.5 dex and tests nonzero interior scales. A
shared scale cannot recover arbitrary element-by-element abundances.

## Other families and limits

Retain accepted pure DA, DB, DAB and DQ products. Diagnose their numerical
failures before retries. DAB needs targeted interpolation work: ten of 32 fresh
tests miss at least one optical target for each refined method. Abundance
interpolation remains a separate test.

PG1159 needs a fixed-spectrum check of the host-ion microfield update and a
supported cold benchmark before a grid. Its one existing request failed
numerically. DAH needs full-Stokes qualification after the dispersion update;
its two existing cases have structure certificates but lack independent
final-spectrum checks. DO/DAO needs diagnosis of the four hot timeouts and
small supported tiers for the ten unattempted DAO requests. None of these
branches has a measured full-grid cost yet.

D6 remains one accepted J1637 coordinate with a conditional GTC comparison.
A broader D6 grid requires a defined composition and parameter range. The
observed comparison does not establish absolute SED accuracy, independent
depth convergence or unbiased parameter recovery.

## Spending and acceptance

The completed DZ replacement campaign used **11.2247981771 charged CPU node-hours out of 13**,
from all nine registered allocations, including the cancelled pending job at zero
charge. Full elapsed time, allocated billing and actual QOS factors are reconciled.
No live reservation remains. No optional cluster grid is submitted. Further
expansion requires a measured resource plan and an approved spending scope.

Before scaling, measure production runtimes and RSS across representative
temperature, gravity and composition extremes. Require unchanged inputs,
public cold starts, native atmosphere certificates and independent spectrum
checks. Compare old/new absolute spectra on identical wavelengths, and label
any resampling. Test depth and wavelength resolution separately. Fresh
interior targets test interpolation; observations test physical applicability.
Archive all outcomes, including numerical failures and resource stops, before
updating mapped selections.

Source evidence: [PR23](https://github.com/kareemelbadry/OpenWD/pull/23),
[PR24](https://github.com/kareemelbadry/OpenWD/pull/24),
[PR25](https://github.com/kareemelbadry/OpenWD/pull/25), and
[PR26](https://github.com/kareemelbadry/OpenWD/pull/26).
[Draft PR27](https://github.com/kareemelbadry/OpenWD/pull/27) contains the
authors' COS comparisons, including remaining metal-wing residuals and
unsupported elements. Those comparisons have not been independently reproduced here.

The separately authorized [October 15 DA/DB/DAB campaign](../da-db-dab-replacements-2026-10-15/README.md) finished its registered work with 4.6108680556 charged CPU node-hours. DAB has 653 fresh accepted results plus three retained older spectra; DA and DB bulk are held after failed required pilots. This charge is separate from the earlier 13-hour campaign. Optional grids remain unsubmitted.
