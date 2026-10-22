# Using OpenWD with an AI assistant

Start here to generate a spectrum or help someone use OpenWD. OpenWD computes
an atmosphere and emergent surface spectrum from stellar parameters. Prefer
the public configuration classes and `run_model` for model generation.

## Install and check the environment

For a new installation, use Python 3.9 or newer:

```bash
git clone https://github.com/kareemelbadry/OpenWD.git
cd OpenWD
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
```

If OpenWD is already installed, use its existing checkout and environment.
Run subsequent commands from the repository root. Confirm the active Python
and package path before running a model:

```bash
python -c "import sys, wd_spectra; print(sys.executable); print(wd_spectra.__file__)"
python -c "from wd_spectra import compiled_backend_available; print(compiled_backend_available())"
```

For an editable install, the package path should be inside this checkout's
`src/wd_spectra`. The second command reports optional C acceleration. `False`
means the Python implementation will be used; calculations can be slower.
See [installation details](docs/getting-started.md#install) and
[performance](docs/development/performance.md) if setup needs attention.

## Generate and plot your first spectrum

Save this as `first_spectrum.py` and run `python first_spectrum.py` in the same
environment. This DA model needs only bundled data. Allow minutes for the
calculation; runtime depends on the machine. Choose a new output directory
when repeating it: OpenWD refuses to overwrite an existing run.

```python
import matplotlib.pyplot as plt
from wd_spectra import DAConfig, run_model

run = run_model(
    DAConfig(effective_temperature=12_000, logg=8.0, quality="standard"),
    "results/my-first-da",
    require_convergence=True,
)
print("Numerical convergence verified:", run.convergence_verified)
print("Spectrum:", run.spectrum_path)
print("Run directory:", run.output_directory)

spectrum = run.spectrum
fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.plot(spectrum.wavelength_angstrom, spectrum.surface_flux_lambda)
ax.set_xlabel("Vacuum wavelength (Å)")
ax.set_ylabel(r"Surface $F_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)")
ax.set_title("DA: 12,000 K, log g = 8.0")
fig.savefig(run.output_directory / "spectrum.png", dpi=200)
fig.savefig(run.output_directory / "spectrum.pdf")
plt.close(fig)
```

Temperature is in kelvin; `logg` is log10 of gravity in cm s⁻². Wavelengths
are vacuum Angstroms. Flux is at the stellar surface, before distance scaling,
extinction, or instrumental broadening.

A successful run prints `Numerical convergence verified: True`. The directory
contains `spectrum.txt`, `atmosphere.npz`, `metadata.json`, `model-run.json`,
and the two plots. Inspect the plots and saved convergence diagnostics before
using the result. This example raises if numerical qualification fails;
retain any saved outputs and inspect the error instead of treating the
existence of a spectrum file as success.

## Choose the next task

| Task | Start here |
| --- | --- |
| Change parameters or use a notebook | [Getting started](docs/getting-started.md), [spectrum notebook](examples/generate_spectrum.ipynb) |
| Choose a composition or model class | [Model guides](docs/models/README.md), [tested points](docs/tested-temperature-ranges.md) |
| Fit an observed DZ spectrum | [DZ fitting example](docs/examples/fitting-dz.md), [fitting notebook](examples/fit_dz_spectrum.ipynb) |
| Change the implementation | [Development and regression policy](docs/development/README.md) |

Read the relevant model guide before adapting the DA example. Abundance
conventions, external data, runtime, and qualification rules differ by class.
For example, PG 1159 distinguishes `spectrum-qualified` from `converged`;
the strict check above accepts only the latter. The sdB workflow has separate
instructions in its [model guide](docs/models/sdB.md).

## When helping a user

- Use documented options first. Do not modify the solver merely to run a model.
- Start with one model before a grid or fit. `quality="quick"` is an interface
  smoke test, not evidence of a converged scientific model.
- Preserve inputs, existing results, and failed-run diagnostics. Do not change
  physics, relax acceptance checks, or rescale flux to make a failed run pass.
- Report the input parameters, source commit (`git rev-parse HEAD`), any local
  source changes, output location, and actual convergence status.
- Numerical convergence does not establish physical accuracy or independent
  depth convergence. Read the [limitations](docs/limitations.md) before making
  scientific claims; report relevant limits alongside the result.

## When changing OpenWD

The public API is in `src/wd_spectra/__init__.py`; model configurations and
dispatch are in `src/wd_spectra/models/`. Read the development policy before
editing. Preserve frozen reference data and select checks appropriate to the
change. For documentation edits, install the test dependencies with
`python -m pip install -e '.[test]'` and run
`python -m pytest tests/test_documentation.py`. Solver or physics
changes need the additional validation described in the development guide.
