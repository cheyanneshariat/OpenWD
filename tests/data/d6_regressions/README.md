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
`warm_v2_feii` with current defaults, clearing cached atoms between scopes.
The optical fixed-atmosphere test still compares current defaults to the
original `warm_v1` spectrum at its original tolerance.

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
