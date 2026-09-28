# Explicit cool-model workflows

The cool DB/DAB implementations are packaged in `wd_spectra._cool`.
This directory retains component tests and compatibility command entry points.
`run_model` and the normal model CLI select the packaged recipes automatically. The
qualified recipes below are experimental physical models with numerical
validation at discrete points; see [tested temperatures](../../docs/tested-temperature-ranges.md).

The low-level drivers also retain explicitly selected comparison/proposal
experiments needed by the investigation. **Their availability is not evidence
that they improve convergence.** Do not enable additional flags and inherit
the qualifications of the recipes below. In particular, the later compatible
coordinates, secant and chemistry-precision experiments did not solve the
5000 K DAB problem and are not enabled here.

## Setup

Run from the repository root:

```sh
python -m pip install -e '.[test]'
export PYTHONPATH=src:research/cool_models
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENWD_NUM_THREADS=1
```

Run different models in separate processes. These research scopes temporarily
replace callbacks and are **not thread safe**. They restore callbacks on exit;
tests exercise exception restoration. Per-iteration logs, rejected proposals,
initialization progress and source identities are written while runs execute.
Keep output directories separate; drivers refuse to overwrite an experiment.
There is no automatic checkpoint lookup or failed-model substitution.

### Additional molecular-DAB data

The usual atomic data, He-REOS.3 and H2-H2 CIA are already bundled. Molecular
DAB additionally requires these exact, unmodified external tables:

| File | Source | SHA-256 |
| --- | --- | --- |
| `H2-He_2011.cia` (~140 MiB) | [HITRAN CIA download page](https://hitran.org/cia/), select this filename | `4f0eb9cd69a1c383f53a1431495bae0c01a30a41cc1b8433c2d726763ff45431` |
| `1H2__RACPPK.states.bz2` | [ExoMol RACPPK states](https://exomol.com/db/H2/1H2/RACPPK/1H2__RACPPK.states.bz2) | `276f5a36d094e7e1417c44f11c1d173417e92629c5f747b6a862bcce5c374a81` |

Set `OPENWD_RESEARCH_DATA` to the directory containing those two files,
or supply `research_data=...` to `run_model`. The default lookup is the
selected `ModelData.cache / "molecular-opacity"` directory. No data
are searched for in an adjacent checkout. Verify before running:

```sh
python -m wd_spectra._cool.check_data
```

The external tables are not redistributed here. Follow HITRAN's citation
policy and cite the Abel et al. H2-He calculation. RACPPK is the Roueff et al.
(2019, A&A 630 A58) ground-electronic-state H2 calculation; ExoMol distributes
it under [CC BY-SA 4.0](https://exomol.com/data/licence/). Its 302 bound levels
are used for both equilibrium and energy, not a fitted atmosphere correction.
An optional independent partition-table test uses `1H2__RACPPK.pf` from the
[same dataset](https://exomol.com/data/molecules/H2/1H2/RACPPK/).

## Dense pure-He DB: fresh 5000 or 8000 K

```sh
python research/cool_models/run_cool_db.py 5000 --output-root results/db-cold-check
# Run independently with 8000 and a new output root for the other tested point.
```

The wrapper implements the identical recipe at both temperatures: fresh
80-layer hydrostatic/discrete transport seed, at most 80 pseudo-time sweeps,
then at most 40 full static correction attempts. It uses the dense REOS/HNC
He/He+/He2+ model, exact dense material tangent, nonlinear ML2 proposal,
column-mass transfer, extended thermal quadrature, and smooth He-minus opacity
join. Local table/trace-ion guards remain active; no temperature switch
selects a different EOS.

It then runs an independent 8000-wavelength, 8/16-angle, longer-wavelength
audit. It exits nonzero unless numerical qualification passes. Failed outputs
are retained for exploration, not relabeled converged. These checks are not
a depth-grid or full-physics certificate. The generated HNC table and its
rebuild command are described in [data/README.md](data/README.md).

## Molecular DAB: fresh 8000 K

After verifying external data:

```sh
python research/cool_models/run_molecular_dab_mass_experiment.py \
  --physical-detuning-lyman --consistent-stark-edge --require-convergence \
  --pseudo-sweeps 40 --relax-bottom \
  8000 --log-h-he -2 --physical-only --stable-transfer \
  --step-method nonlinear-convection-current-energy --inner-scaling svd \
  --inner-max-evaluations 1000 --max-iterations 35 --no-continuations \
  --state-sum-h2 --output-root results/dab-cold-check
```

The separate `check_molecular_dab_result.py RUN_DIRECTORY` command checks
recorded structure-grid evidence: flux, local energy, independently checked
source closure, lower-boundary screening and a measured small correction in
a non-collapsed trust region. `--require-convergence` applies that check to
the run and exits nonzero on failure, retaining exploratory outputs.

The final spectrum matches the structure's mass-transfer law, angular order,
opacity and line-profile selections. It is not renormalized. A surface flux
integral alone is not sufficient for success.

## Cold-start DAB at 7500 K

The public workflow does not need a previously converged atmosphere:

```python
from wd_spectra import DABConfig, run_model
run_model(DABConfig(effective_temperature=7500, quality="production"),
          "results/dab-7500-fresh", require_convergence=True)
```

The September 7 fresh 80-layer run used the same final molecular/line physics,
40 provisional pseudo-time sweeps and 14 static iterations. It passed the
measured correction, all-depth flux, local energy, scattering and bottom-boundary
checks. An independent 16-angle / refined-wavelength audit also passed.
The earlier 166-layer continuation is historical evidence, not the public
reproduction recipe. The small archives in `checkpoints/` and prediction scripts
remain available for diagnostic comparisons only; they are not inputs to public
model generation or cold-start CI. See their [provenance](checkpoints/README.md).
The fresh calculation screens its lower boundary without either input.
The old `--refine-atmosphere` / `--extend-atmosphere` experiments remain
diagnostics only. Molecular 7250 and 5000 K remain unqualified.

### Independent DAB quadrature audit

After the 7500 K cold calculation above, hold its atmosphere fixed and independently
refine wavelength/angular sampling (no relaxation or normalization):

```sh
python research/cool_models/audit_molecular_dab_grid.py \
  --audit-output results/dab-7500-audit.json \
  --physical-detuning-lyman --consistent-stark-edge \
  --thermal-maximum 1e9 --angles 16 --lyman-core-subdivisions 10 \
  7500 --log-h-he -2 --physical-only --stable-transfer \
  --step-method nonlinear-convection-current-energy --inner-scaling svd \
  --inner-max-evaluations 1000 --max-iterations 1 --no-continuations \
  --structure-wavelength-count 6000 --state-sum-h2 \
  --resume-atmosphere results/dab-7500-fresh/worker/7500/atmosphere.npz \
  --output-root results/dab-7500-audit-work
```

Inspect `flux_and_local_energy_pass` and source closure in the JSON; this
diagnostic's process exit alone is not a convergence certificate. It cannot
replace the prior measured-step check, depth tests or missing physical effects.

## Tests and historical workspace

```sh
python -m pytest research/cool_models -o addopts=-ra
python -m pytest tests/test_protected_model_canaries.py -o addopts=-ra
```

GitHub runs the research **component** tests separately on Python 3.9 and
3.12 without downloading the large optional tables. Data-dependent tests
explicitly skip when absent. This is not a claim of cool-atmosphere execution
in ordinary CI. The slow production canaries remain a separate workflow.

In the original local research workspace, migrated `scripts/*.py` paths are
compatibility symlinks to these files, not second editable copies. Old reports
retain historical paths and “no push” statements describing their original
tranches. Use this README for the checkpoint's current reproduction commands.
