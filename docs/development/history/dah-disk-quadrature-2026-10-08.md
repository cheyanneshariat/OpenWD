# DAH drift-resolved disk quadrature

[Research history](README.md) · [DAH guide](../../models/DAH.md)

This change adds one opt-in numerical setting,
`DAHConfig(disk_component_drift_angstrom=16)`, for dipole geometries.
It changes the visible-disk quadrature, not the atomic data, continuum,
line-broadening prescription or selected atmosphere physics. Release
defaults, historical spectra and all regression tolerances are unchanged.

## Why refine the field bins?

The existing public settings `disk_field_bins=N` and `disk_field_bins=None`
remain available. The latter keeps every raw surface cell. Increasing `N`
does not increase the raw surface-grid resolution. On the published
geometries, requesting 267 equal-weight bins yields 147 occupied bins for
J1018 and 126 for J1351; their uncompressed default grids have 190 and 162
cells. These are geometry checks, not new spectrum calculations. The
lower-level `dipole_surface_cells` function also accepts explicit dense
angular grids. The new public option combines a dense grid with a
component-drift binning target. Use the existing settings when they are
already adequate; this note does not claim that every existing setting fails.

The default compresses the visible surface into 21 projected-weight bins.
Each bin is synthesized at its mean field. Rapid component motion between
bins can leave discrete spectral copies rather than a smooth disk integral.
The new option uses a dense surface grid and estimates the largest
component wavelength derivative on a 4001-point field grid. It places edges
at equal increments of accumulated drift. Each bin retains projected
weight, mean field, rms field–ray cosine and mean limb cosine. Bins never
merge fields at or below 1 MG with fields above it; continuous surface
bounds remain independent of compression.

The component screen uses `energy × dipole-strength` relative to the
strongest component of each parent, with a 1% cutoff and an output-window
margin of 100 Å. It omits temperature-dependent populations. The requested
distance is an estimated within-bin drift budget, not a rigorous bound on
flux error, every component, or adjacent representative fields. Tightening
the distance does not independently test the raw surface mesh or angular
compression. Extremely fine requests fail at 4096 drift intervals rather
than being silently made coarser.

## Fixed-atmosphere evidence and cost

The diagnostic spectra use the immutable paper atmospheres and published
geometries for J1018+0111 and J1351+5419. Only disk quadrature changes;
magnetic continuum/EOS options remain at their defaults. These are saved
atmosphere syntheses, not cold-start calculations or magnetic equilibrium
certificates. The archived runs use one low-priority process, one numerical
thread and 3401 output wavelengths. Their measured synthesis times are:

| Object | Default 21 bins | 32 Å target | 16 Å target | 8 Å target |
| --- | ---: | ---: | ---: | ---: |
| J1018+0111 | 46.1 s | 65 bins / 153.6 s | 130 / 307.7 s | 260 / 496.3 s |
| J1351+5419 | 43.3 s | 134 bins / 279.2 s | 267 / 646.0 s | 533 / 993.0 s |

This is an accuracy–cost tradeoff, not a speedup. Wall times depend on the
machine and runtime; they are not performance guarantees. The 8-Å calculation
is a finer reference on the same surface prescription, not exact truth.
The diagnostic data and an independent comparison script are in
[`research/dah_disk_quadrature`](../../../research/dah_disk_quadrature/README.md).
The script compares convolved spectra without independently fitting a flux
scale to each variant. Its JSON report defines the comparison mask and
reports maximum and RMS relative differences against the 8-Å reference.

An independent re-comparison of the archived spectra, with no per-variant
flux rescaling, gives:

| Object | 21 bins: max / RMS | 32 Å target | 16 Å target |
| --- | ---: | ---: | ---: |
| J1018+0111 | 7.68% / 1.77% | 2.03% / 0.41% | 0.63% / 0.13% |
| J1351+5419 | 5.10% / 1.06% | 1.07% / 0.18% | 0.46% / 0.05% |

These numbers concern compression of the surface integral at the bundled
observational resolution, not an error against a known exact physical
spectrum. The comparison keeps the baseline velocity fixed (96 km/s for
J1018, 0 km/s for J1351) and includes every positive finite reference pixel
in the stated interval; it does not select only improved features.

The disk correction does not solve the high-field continuum mismatch or
establish a simultaneous Stark–Zeeman treatment. Literature review,
additional benchmark stars and physical continuum/line-width experiments
remain separate work. No experimental RWA/EOS combination is promoted by
this change.

## Validation scope

The focused tests cover component derivatives, default-path identity,
weight/mean-field conservation, interval construction, the exact 1-MG
boundary, the zero-field limit, invalid requests and preservation of the
positional configuration interface. Existing frozen-paper tests retain
their original absolute-flux tolerances. Fast and fixed-spectrum checks
complement, but do not replace, the protected PR qualification workflow.
