"""One unchanged automatic cold-start canary, used by local and CI runners."""

import argparse
from pathlib import Path
from wd_spectra import DBConfig, DABConfig, run_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        choices=(
            "dense-db-5000",
            "dense-db-8000",
            "molecular-dab-7500",
            "molecular-dab-9000",
            "molecular-dab-10000",
        ),
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    workflow, temperature = args.case.rsplit("-", 1)
    cls = DBConfig if workflow == "dense-db" else DABConfig
    result = run_model(
        cls(effective_temperature=int(temperature), quality="production"),
        args.output,
        require_convergence=True,
    )
    assert result.selection.workflow == workflow
    assert result.convergence_verified


if __name__ == "__main__":
    main()
