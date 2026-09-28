# DB module

[Model guide](README.md) · [Getting started](../getting-started.md)

`compute_db` solves a homogeneous pure-helium LTE atmosphere. It combines a
Hummer--Mihalas/Q-MHD helium EOS, He I/II/III continuum opacity, corrected
Doppler-convolved Beauchamp25-LD He I Stark profiles, Schoening/SYNSPEC He II
profiles, Unsold neutral-He broadening, and ML2/alpha=1.25 convection.

```bash
python examples/one_shot_db.py --teff 20000 --logg 8.0 \
  --quality standard --output results/db-20000-8.0
```

Typical standard calculations take roughly 3--10 minutes on a current laptop;
cool neutral-line models can be slower. No previous atmosphere is needed.

## Cool helium

Use `run_model(DBConfig(...), output_directory)` to select the experimental
dense-neutral helium treatment when indicated by the local material screen.
It combines the tabulated bulk EOS with approximate chemical potentials and
trace-ion chemistry, without inserting that closure into warm ionized helium.
The command-line example uses the same automatic selection as `run_model`.
The lower-level `compute_db` preset remains an explicit atomic-physics interface.

Protected cold starts cover the established prescription at 10000 and 22000 K
and the dense workflow at 5000 and 8000 K. See [tested points](../tested-temperature-ranges.md),
[cool-model setup](../getting-started.md#cool-helium-and-mixed-atmospheres), and
[physical limitations](../limitations.md#physical-approximations).
