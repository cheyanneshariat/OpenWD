# DAB/DBA module

[Model guide](README.md) · [Getting started](../getting-started.md)

`compute_dab` defaults to one homogeneous atomic H/He layer in LTE; it is not a
stratified thin-hydrogen-layer calculation. Hydrogen and helium share the
charge-neutrality solution and nonideal occupation-probability EOS. The module
combines the DA hydrogen opacity/profile treatment with the DB helium profiles
and uses ML2/alpha=1.25 convection.

```bash
python examples/one_shot_dab.py --teff 20000 --logg 8.0 --log-h-he -2 \
  --quality standard --output results/dab-20000-8.0
```

`--log-h-he` means `log10[N(H)/N(He)]`. Standard calculations typically take
3--20 minutes, with cool mixtures and low-gravity production models slower.

## Cool mixtures

Use `run_model(DABConfig(...), output_directory)` for automatic molecular
workflow selection. That workflow includes H2, H2+, H-, H3+, H/He ionization,
H2-He/H2-H2 collision-induced absorption, and neutral Ly-alpha wings. The command-line example uses the same automatic selection as `run_model`.
The lower-level `compute_dab` preset remains an explicit atomic-physics interface.

Fresh molecular calculations have been qualified at 7500, 8000, 9000, and
10000 K for log g = 8 and log10 N(H)/N(He) = -2. These are cold-start points,
not a guaranteed interval; 7250 and 5000 K remain unqualified. Setup requires
production quality, the normal package installation, and additional public data. See
[cool-model setup](../getting-started.md#cool-helium-and-mixed-atmospheres),
[tested points](../tested-temperature-ranges.md), and
[physical limitations](../limitations.md#physical-approximations).
