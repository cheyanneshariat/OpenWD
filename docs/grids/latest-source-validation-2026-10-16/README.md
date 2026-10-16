# Latest-source validation, October 16

The 23 cold models below are queued under source `00503a0dd22a028e38741b90d53d9e18a8731545`. None has been independently accepted yet. The selected table is unchanged. The earlier DA/DB/DAB campaign remains tied to `c103b53`.

| Check | Cold requests | Purpose |
|---|---:|---|
| DO/DAO | 5 | Repeat four older timeouts after the upstream solver changes, with one successful DO control. |
| DZ/DAZ | 5 | Check two DZ and three DAZ anchors with the new metal-line defaults. |
| DAB/DBA | 13 | Calculate nine interpolation inputs and four independent test spectra in one patch. |

The [full requests](REQUESTS.json) preserve configurations and roles. The four hot failure cases run only if the DO control passes. Required native tolerances remain unchanged. No automatic scientific retry or grid expansion is included. PG1159's limited spectrum qualification remains distinct from full convergence.

Fourteen focused tests passed on both macOS and Linux. Six fixed-atmosphere spectral controls passed locally. These checks protect software behavior and saved spectra; they do not certify cold-model convergence.

## Why some DA/DB solves stop without qualification

The saved review covers 51 native failures: 14 DA and 37 DB. In 36 cases, the accepted temperature step is below its limit while the recorded unrestricted correction fails. The final native certificate correctly rejects those results. The relevant solver and certificate files are unchanged between `c103b53` and latest main.

![Accepted steps and unrestricted corrections](stopping_check.png)

Circles show DA and triangles DB. Dashed lines mark the native limits. Thirty-eight saved failures have both a measured correction check and a positive accepted step; 36 lie left of the vertical line and above the horizontal line. Unmeasured checks and dense-DB qualification records are omitted. All plotted models failed their final qualification. [PDF](stopping_check.pdf) and [saved checks with the analytic example](DIAGNOSIS.json) are included.

A constructed two-variable problem reproduces the stopping disagreement in the current driver. Its equations have no real root. An existing explicit convergence callback prevents the false driver-level pass in that example. This identifies a stopping-contract weakness; it does not establish the physical cause or a successful atmosphere correction. Domain stops, invalid optical-depth grids and unfinished timeouts remain separate.

## Metal-line changes on identical atmospheres

![New and previous metal-line defaults](line_policy_changes.png)

Orange uses the previous Kurucz line defaults; blue uses latest main. Left panels show absolute surface flux and right panels the percentage change. DZ panels show the ultraviolet; DAZ panels show their available optical samples near 3700–4500 Å. All changes remain visible, including large fractional changes in weak UV flux. No spectrum was normalized or resampled. The two DZ sampled UV bands change by 0.676% and 3.78% in integrated absolute flux difference. These are model differences, not observational fit scores.

Both line policies passed their native source checks for all four saved atmospheres. The previous policy recovered its immutable controls at their existing tolerances. [PDF](line_policy_changes.pdf), [comparison records](COMPARISON.json) and the four saved array pairs ([PG1225](dz-pg1225.npz), [J0738](dz-j0738.npz), [G149−28](daz-g149_28.npz), [GALEX J1931](daz-galex1931.npz)) preserve the inputs and outputs. These fixed-state checks do not establish atmosphere equilibrium, depth convergence or physical accuracy.

## Independent interpolation and remaining checks

None of the 32 historical independent DAB targets was rerun in the completed replacement campaign. The old 22/32 result therefore cannot qualify the updated selection. This new tier calculates all nine inputs and four withheld targets under the same latest source, in the first targeted temperature/gravity patch. The targets stay outside the predictor mesh. A pass would apply to that patch only. Independent atmosphere-depth and observational checks remain separate work.

Job `59857587` has a nine-hour wall limit. The conservative full pending reservation is at most 3.375 CPU node-hours, below a new self-imposed four-hour guard; actual allocated billing and terminal charge remain unverified. The earlier 12.5616319444-hour DA/DB/DAB and 11.2247981771-hour DZ/DAZ charges remain separate. [Compact status](SUMMARY.json) records these denominators.

Monitoring was stopped at Cheyanne's request and remains off. Private permanent archival remains pending. No merge or release publication is included.

The portable [metal plotter](plot_metal_changes.py) and [stopping-check plotter](plot_stopping_check.py) read the saved arrays and checks. They use the included [plot style](plot-style.mplstyle) and perform no atmosphere calculations.
