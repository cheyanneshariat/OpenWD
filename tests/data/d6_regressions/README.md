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
