"""Conservative CI selection; unknown paths require full qualification."""

import json
import os
from pathlib import PurePosixPath
import subprocess


def requires_full(paths):
    for path in paths:
        p = PurePosixPath(path)
        if path in {"README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", ".gitignore"}:
            continue
        if p.parts[0] == "docs":
            continue
        if (
            p.parts[0] == "tests"
            and p.suffix == ".py"
            and p.name
            not in {
                "conftest.py",
                "test_protected_model_canaries.py",
                "test_spectral_regressions.py",
                "test_daz_regressions.py",
                "test_dq_spectral_regression.py",
                "test_dq_release.py",
                "test_hot_cold_canary.py",
                "test_released_family_canaries.py",
                "test_dah_paper.py",
                "test_radiative_da_convergence.py",
            }
        ):
            continue
        if p.parts[0] == "research" and p.name.startswith("test_"):
            continue
        if path == "examples/generate_spectrum.ipynb":
            continue
        return True
    return False


def plan(event_name, event, paths):
    if event_name in {"workflow_dispatch", "schedule"}:
        return True, True
    if event_name != "pull_request":
        raise ValueError(f"unsupported qualification event: {event_name}")
    pr = event["pull_request"]
    required = requires_full(paths)
    requested = not pr["draft"] and any(
        label["name"] == "full-validation" for label in pr["labels"]
    )
    return required, requested


def main():
    from pathlib import Path

    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    paths = []
    if os.environ["GITHUB_EVENT_NAME"] == "pull_request":
        pr = event["pull_request"]
        # Include both old and new names for renames. Failure must not certify
        # an empty diff. Never interpolate PR-controlled strings into a shell.
        paths = (
            subprocess.check_output(
                [
                    "git",
                    "diff",
                    "--no-renames",
                    "--name-only",
                    "-z",
                    pr["base"]["sha"] + "..." + pr["head"]["sha"],
                ],
            )
            .decode()
            .strip("\0")
            .split("\0")
        )
        paths = [path for path in paths if path]
    required, requested = plan(os.environ["GITHUB_EVENT_NAME"], event, paths)
    print(json.dumps(dict(required=required, requested=requested, changed=paths)))
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"required={str(required).lower()}\n")
        stream.write(f"requested={str(requested).lower()}\n")


if __name__ == "__main__":
    main()
