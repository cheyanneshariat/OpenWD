<p align="center">
  <img src="docs/assets/openwd_logo.png" alt="OpenWD logo" width="520">
</p>

# OpenWD

OpenWD calculates white-dwarf atmospheres and their emergent spectra from
temperature, surface gravity, and composition. It solves the atmospheric
structure and radiative transfer from scratch, rather than interpolating a
precomputed spectral grid.

OpenWD provides plane-parallel models for these classes:

| Class | Atmosphere | Populations | Guide |
| --- | --- | --- | --- |
| DA | Hydrogen | LTE | [DA](docs/models/DA.md) |
| DAZ | Hydrogen with metals | LTE | [DAZ](docs/models/DAZ.md) |
| DB | Helium | LTE | [DB](docs/models/DB.md) |
| DAB/DBA | Homogeneous hydrogen–helium mixture | LTE | [DAB/DBA](docs/models/DAB.md) |
| DZ/DBZ | Helium with metals | LTE | [DZ/DBZ](docs/models/DZ.md) |
| DQ | Helium with trace carbon and C₂, refractive transfer | LTE | [DQ](docs/models/DQ.md) |
| DO/DAO | Hot helium or hydrogen–helium | NLTE hydrogen and helium (LTE charge closure) | [DO/DAO](docs/models/DO-DAO.md) |
| PG 1159 | Hot helium–carbon–oxygen | NLTE He, C and O; trace elements in the line formation | [PG 1159](docs/models/PG1159.md) |
| D6 | Hydrogen/helium-free carbon–oxygen with heavier elements | LTE | [D6](docs/models/D6.md) |
| DAH | Magnetic hydrogen spectra, normalized Kurucz/Griem profiles on a DA structure | LTE | [DAH](docs/models/DAH.md) |
| sdB | Lower-gravity hydrogen–helium with trace metals | LTE structure; NLTE H/He and selected metals | [sdB](docs/models/sdB.md) |

The white-dwarf presets are calculated from a cold start and report whether
the result passed its numerical convergence checks. The sdB workflow first
calculates an LTE structure, then solves NLTE populations and line formation
on that fixed structure. The guides describe each class's
physics and options; the [tested points](docs/tested-temperature-ranges.md)
list the temperatures and compositions that have been run from a cold start.

## Get started

Clone the repository and install it with Python 3.9 or newer:

```bash
git clone https://github.com/kareemelbadry/OpenWD.git
cd OpenWD
python -m pip install -e .
```

Generate a hydrogen-atmosphere spectrum:

```python
from wd_spectra import DAConfig, run_model

run = run_model(
    DAConfig(effective_temperature=12_000, logg=8.0, quality="standard"),
    "results/my-first-da",  # use a new directory for each run
)

wavelength = run.spectrum.wavelength_angstrom
surface_flux = run.spectrum.surface_flux_lambda
print("Numerical convergence verified:", run.convergence_verified)
```

The calculation reports progress and saves its spectrum, diagnostics, and
input parameters. Wavelengths are vacuum Angstroms; flux is the emergent
surface `F_lambda` in erg s⁻¹ cm⁻² Å⁻¹. No previous atmosphere is needed.

For an editable, plotting walkthrough, open the
[example notebook](examples/generate_spectrum.ipynb).
The [getting-started guide](docs/getting-started.md) covers installation,
changing composition, output files, and cool-model data requirements.

## How it works

OpenWD couples hydrostatic structure, equations of state, LTE or NLTE level
populations, opacity, radiative transfer, and ML2 convection, and solves them
with a shared nonlinear (trust-region Newton) solver. Line profiles, continuum
absorption and metal opacity are included as appropriate to the composition.
`run_model` selects the implemented physics from the configuration and local
material diagnostics before solving; it never changes physics in response to
a failed calculation.

An optional C extension accelerates the expensive transfer and opacity
kernels. See the [model guides](docs/models/README.md) for the physical
ingredients and the [performance guide](docs/development/performance.md)
for threading and benchmarks.

![OpenWD workflow](docs/assets/openwd_workflow.png)

## Caveats and limitations

OpenWD is research software under active development. Check convergence and
physical applicability for each result. A spectrum that did not pass its
module's checks is kept with a warning for exploratory work; pass
`require_convergence=True` to make qualification mandatory. The
[limitations guide](docs/limitations.md) explains what the checks establish,
known accuracy limits, and which temperatures and compositions have been
tested.

## Documentation and development

Start at the [documentation home](docs/README.md). User instructions are
separate from the [development and testing guide](docs/development/README.md)
and the [historical research notes](docs/development/history/README.md).

To run the ordinary tests:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

## Data and license

The atomic and constitutive data for all classes are bundled. The one
exception is the molecular cool DAB/DBA workflow, which needs additional
public tables described in the
[data instructions](research/cool_models/README.md#additional-molecular-dab-data).

OpenWD source is BSD-3-Clause licensed. Scientific tables retain their own
licenses and attribution; see [third-party notices](THIRD_PARTY_NOTICES.md).
