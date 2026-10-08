# OpenWD documentation

OpenWD turns a white dwarf's temperature, surface gravity, and composition
into a model atmosphere and surface spectrum. Start with one fresh calculation,
inspect its convergence status, then change the parameters for your application.

## Using OpenWD

1. [Get started](getting-started.md): install, run a model, and read the outputs.
2. [Interactive notebook](../examples/generate_spectrum.ipynb): edit parameters
   and plot the resulting spectrum. A second notebook
   [fits DZ abundances to an observed spectrum](../examples/fit_dz_spectrum.ipynb).
3. [Choose a model](models/README.md): DA, DAZ, DB, DAB/DBA, DZ/DBZ, DQ,
   DO/DAO, PG 1159, D6, DAH and [sdB](models/sdB.md).
4. [Caveats and limitations](limitations.md): convergence, accuracy, and
   [tested temperatures and compositions](tested-temperature-ranges.md).

## Developing OpenWD

- [Development and regression policy](development/README.md): local workflow
  and checks that protect existing models.
- [Performance](development/performance.md): compiled acceleration and threads.
- [Solver diagnostics](development/solver-telemetry.md): interpreting iteration
  logs and convergence evidence.
- [Hot DA/DAO trace metals](development/hot-daz-trace-metals.md): experimental
  fixed-host NLTE metals (C, N, O, Al, Si, P, S, Fe, Ni) and the G191-B2B
  benchmark; development record, not a validated preset.
- [Niobium, zinc, copper and nickel lines: HS 0209+0832](development/niobium.md):
  bundled Nb I--VI data, optional Zn, Cu and Ni IV--VI lines, and a nine-metal
  H/He model compared with the FUSE and STIS spectra.
- [Hot subdwarf (sdB) H/He models](development/sdb-hybrid.md):
  LTE-structure + NLTE H/He hybrid for 20-40 kK sdBs, with MALI and opt-in
  helium-solver options, numerical checks and observed comparisons.
- [Research history](development/history/README.md): dated investigations,
  rejected experiments, and detailed validation records. These are not the
  current user instructions.

See also the [data attribution and licenses](../THIRD_PARTY_NOTICES.md).
