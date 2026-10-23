# J1637 warm trajectory and fixed spectrum

`warm_v1/j1637.npz` is an immutable full-resolution regression reference for
the standard J1637 request: 48 layers, 25,000-line budgets for structure and
synthesis, and the normal optical spectrum grid. `manifest.json` records its hashes, equations,
configuration, archived source certificate, and measured build timings.
`warm_v1-seed.npz` contains the corresponding immutable input; both hashes
are checked, and its arrays must exactly match those in the reference.

The source is a previously certified standard J1637 **cold** atmosphere.
The source certificate applies to its recorded code revision; it does not
certify convergence under the current code. The offline builder perturbs its
temperature smoothly by up to 1.5% in the line-forming layers and uses the
current conservative equations to freeze two measured solver steps, including
complete temperature, pressure, density and electron-density profiles and
flux/energy/correction/source residuals. The frozen optical spectrum uses the
unperturbed archived atmosphere.

The trajectory is intentionally stopped through the iteration callback.
Its success is regression consistency, not a convergence certificate or a
new cold-start qualification. Starting from the frozen input also recomputes
the selected structural lines and wavelength grid; the reference therefore
describes this warm solve, not a claim to exactly resume the historical cold
trajectory. EOS populations, opacity, transfer and corrections are freshly
computed by each test. Tests never regenerate the reference.

`warm_v2_feii` adds the default Kurucz Fe II UV transitions introduced on
2026-10-12. D6's UV blanketing includes these transitions even when its
requested output spectrum is optical. The added transitions can change the
flux-ranked structural line set and wavelength grid; their decay branches
also enter inferred natural widths, and the added observed-energy level
enters the Fe II partition function. The trace-hydrogen width correction
does not apply to this hydrogen/helium-free composition.

Both versions use exactly the same archived cold source, published parameters,
abundances, 1.5% perturbation, 48 layers and 25,000-line budgets. The old
reference and its tolerances remain unchanged. The warm test checks `warm_v1`
with the Fe II supplement disabled in a test-only scope, then checks
`warm_v2_feii` with the defaults at its approval, clearing cached atoms
between scopes. Both now use their corresponding historical scopes.
At the time of the v2 approval, the optical fixed-atmosphere spectrum also
remained within the original `warm_v1` tolerance. Later versions have their
own reviewed spectra; historical cases retain their original physics scopes.

The v2 manifest records the independently verified legacy recovery and the
complete-profile differences caused by the new atom. Its source certificate
remains the historical certificate above; neither trajectory supplies a new
cold-start certificate.

The legacy temperature profiles are recovered within `7.2e-13` relative;
gas pressure is identical, and density/electron profiles agree within
`1.5e-12`. With current Fe II defaults, the maximum temperature change is
6.653 K (`1.76e-4` relative), density changes by `3.08e-4` and electron
density by `1.29e-4`. The fixed optical spectrum changes by at most
`1.24e-6` relative, within its original `2e-6` tolerance.

The Linux capture reevaluated the same exponential perturbation: initial
temperature layer 21 differs from the macOS v1 seed by one float64 ULP
(`1.82e-12` K). The raw captured seed and reference remain unchanged and
exactly match each other. Cross-version seed checks require exact equality
of column mass, optical depth and base temperature, and at most one ULP for
the perturbed initial temperature. All physical trajectory tolerances remain
unchanged. Future frozen-source builds reuse the archived initial-temperature
array verbatim to avoid this platform roundoff.

Run the bounded trajectory and independent fixed spectrum with:

```bash
python tools/validate.py regression --case d6-j1637 --jobs 1
python tools/validate.py spectra --case d6-j1637 --jobs 1
```

The full public cold-start canary remains available through
`python tools/validate.py cold --case d6-j1637 --jobs 1`; weekly and manually
dispatched CI still run it. A successful warm regression does not satisfy
`full_qualification` in the local validation report.

An intentional reviewed numerical change requires a separately versioned
reference and explicit review of its differences. Build a new directory
offline with a certified standard J1637 atmosphere:

```bash
PYTHONPATH=src python tools/build_d6_regression.py SOURCE_ATMOSPHERE.npz NEW_DIRECTORY
```

The builder refuses to overwrite an existing directory. Preserve previous
references and their provenance when accepting a new one.

## Current RAP reference

`warm_v3_independent_kurucz` retains the independently qualified Kurucz atom.
`warm_v4_rap_absolute` adds the Fe RAP reader repair taken from PR 37:
the supplied excitation already contains the ground-term centroid, and the
cross-section grid is an absolute photon grid. Counting that excitation once
and retaining the sampled support changes the fixed optical spectrum by at
most 0.0219181%, at 4883.3 Angstrom. Integrated flux over the stored grid
changes by -0.0052548%.

The v4 approval was explicitly requested for the cool-DZ release on
2026-10-23. Its seed is copied exactly from the checked v3 capsule.
Temperature, column mass, optical depth, abundances, 48 layers and both
25,000-line budgets are retained. The offline builder freshly computes two
measured warm steps and the fixed spectrum; all original comparison
tolerances remain unchanged. The old v1, v2 and v3 capsules and manifests
remain byte-for-byte unchanged. Their tests use their original atomic
policies and the preceding RAP reader, with caches cleared on both scope
boundaries. The current v4 tests use the corrected reader.

As for every earlier version, the archived cold certificate qualifies its
source revision only. Neither the frozen optical spectrum nor two perturbed
warm steps establish a new cold-start certificate.

Explicit capture from the checked-in source capsule:

```bash
PYTHONPATH=src python tools/build_d6_regression.py \
  tests/data/d6_regressions/warm_v3_independent_kurucz NEW_DIRECTORY \
  --from-frozen-reference --spectrum-first
```

The Fe II revision can also be captured from the checked-in, checksum-verified
seed capsule, without transporting the original cold-atmosphere file:

```bash
PYTHONPATH=src python tools/build_d6_regression.py \
  tests/data/d6_regressions/warm_v1 NEW_DIRECTORY \
  --from-frozen-reference --verify-legacy-uv-controls
```

This explicitly verifies the old trajectory at its original tolerances before
capturing the new default trajectory. It is expensive: four measured updates
plus a fixed spectrum. Use a hosted runner if local memory is insufficient.
