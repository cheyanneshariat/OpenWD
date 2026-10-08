# Model guide

[Documentation home](../README.md) · [Getting started](../getting-started.md)

Use the configuration for your composition with `run_model`. It selects the
implemented material treatment before solving a new atmosphere. The experimental
sdB workflow uses the research drivers described in its guide.

Each guide includes the paper's comparison figures, with inline previews,
downloadable PDFs, and an explanation of the model inputs and display
processing. The figures preserve the paper snapshots; they are not regenerated
when defaults change. The [figure manifest](../assets/paper-figures.json)
records source filenames and checksums. Current convergence qualifications
remain listed separately under [tested points](../tested-temperature-ranges.md).

| Configuration | Composition | Populations | Details |
| --- | --- | --- | --- |
| `DAConfig` | Pure hydrogen | LTE | [DA physics](DA.md) |
| `DAZConfig` | Hydrogen with metals; abundances relative to H | LTE | [DAZ physics](DAZ.md) |
| `DBConfig` | Pure helium | LTE | [DB physics](DB.md) |
| `DABConfig` | Homogeneous H/He; `log_hydrogen_to_helium` sets log10 N(H)/N(He) | LTE | [DAB/DBA physics](DAB.md) |
| `DZConfig` | Helium with metals and optional trace hydrogen | LTE | [DZ/DBZ physics](DZ.md) |
| `DQConfig` | Helium with trace carbon and C₂; refractive transfer | LTE | [DQ physics and setup](DQ.md) |
| `DOConfig` | Pure helium, hot | NLTE He; LTE charge closure | [DO/DAO physics and data](DO-DAO.md) |
| `DAOConfig` | Homogeneous H/He, hot | NLTE H and He; LTE charge closure | [DO/DAO physics and data](DO-DAO.md) |
| `PG1159Config` | Helium, carbon and oxygen with trace elements (mass fractions) | NLTE He/C/O; trace-element NLTE line formation | [PG 1159 physics and data](PG1159.md) |
| `D6Config` | Hydrogen/helium-free C/O-dominated mixture; abundances relative to C | LTE | [D6 physics and validation](D6.md) |
| `DAHConfig` | Pure hydrogen with a uniform or offset-dipole magnetic field | LTE | [DAH physics and validation](DAH.md) |
| sdB research drivers | Lower-gravity H/He with trace metals | LTE structure; NLTE H/He and selected metals | [sdB method, examples and limitations](sdB.md) |

All configurations describe plane-parallel atmospheres. They predict spectra
for specified parameters; they do not fit observations. Applicability differs
between classes and between the warm and cool workflows of DB and DAB; see
[limitations](../limitations.md) and [tested points](../tested-temperature-ranges.md).
A high temperature in `DAConfig`, `DBConfig` or `DABConfig` does not switch to
NLTE; request `DOConfig`, `DAOConfig` or `PG1159Config` explicitly.

The individual guides also describe the `compute_*` functions and
command-line scripts. The explicit `compute_db` and `compute_dab` presets do
not switch to the cool dense/molecular workflows; use `run_model` (the
[automatic interface](../getting-started.md)) for that. `compute_dq`,
`compute_do`, `compute_dao` and `compute_pg1159` run the same calculation as
`run_model` with the corresponding configuration.
