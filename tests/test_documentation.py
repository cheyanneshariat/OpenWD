"""Keep the user guide navigable as research records are reorganized."""

import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.rglob("*.md")) + [ROOT / "README.md", ROOT / "examples/data/README.md"]
NOTEBOOK = ROOT / "examples/generate_spectrum.ipynb"
FIT_NOTEBOOK = ROOT / "examples/fit_dz_spectrum.ipynb"


def markdown(path):
    if path.suffix == ".ipynb":
        return "\n".join(
            "".join(cell["source"])
            for cell in json.loads(path.read_text())["cells"]
            if cell["cell_type"] == "markdown"
        )
    return path.read_text()


def local_links(path):
    for target in re.findall(r"\]\(([^)\n]+)\)", markdown(path)):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith("/"):
            continue
        resolved = (
            (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
        )
        yield resolved, unquote(parsed.fragment)


def heading_anchors(path):
    anchors = set()
    counts = {}
    fenced = False
    for line in markdown(path).splitlines():
        if line.startswith(("```", "~~~")):
            fenced = not fenced
        if fenced or not re.match(r"^#{1,6} ", line):
            continue
        heading = re.sub(r"^#+\s+", "", line).strip().lower()
        slug = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        count = counts.get(slug, 0)
        anchors.add(slug if not count else f"{slug}-{count}")
        counts[slug] = count + 1
    return anchors


def test_local_documentation_links_and_headings_resolve():
    problems = []
    for source in PAGES + [NOTEBOOK, FIT_NOTEBOOK]:
        for target, fragment in local_links(source):
            if not target.exists():
                problems.append(f"{source.relative_to(ROOT)} -> missing {target}")
            elif fragment and target.suffix == ".md":
                if fragment not in heading_anchors(target):
                    problems.append(
                        f"{source.relative_to(ROOT)} -> missing {target}#{fragment}"
                    )
    assert not problems, "\n".join(problems)


def test_every_documentation_page_is_reachable_from_the_home_page():
    seen = set()
    queue = [DOCS / "README.md"]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        queue.extend(
            target
            for target, _ in local_links(path)
            if target.is_file()
            and target.suffix == ".md"
            and DOCS in target.parents
            and target not in seen
        )
    assert not set(DOCS.rglob("*.md")) - seen


@pytest.mark.parametrize("relative", ["README.md", "docs/README.md",
    "docs/getting-started.md", "docs/models/README.md", "docs/limitations.md",
    "docs/tested-temperature-ranges.md", "examples/generate_spectrum.ipynb"])
def test_dq_is_discoverable_in_user_documentation(relative):
    assert "DQ" in markdown(ROOT / relative)


def test_dq_quick_start_builds_a_valid_cold_request_without_running_it(monkeypatch):
    from wd_spectra import DQConfig
    from wd_spectra.models.dq import validate_config
    calls = []

    def capture(config, output, **kwargs):
        assert isinstance(config, DQConfig)
        validate_config(config)
        assert kwargs == {"require_convergence": True}
        calls.append((config, output))

    monkeypatch.setattr("wd_spectra.run_model", capture)
    blocks = re.findall(r"```python\n(.*?)```", (DOCS / "getting-started.md").read_text(), re.S)
    examples = [code for code in blocks if "DQConfig(" in code and "run_model(" in code]
    assert len(examples) == 1
    exec(compile(examples[0], "<DQ quick start>", "exec"), {})
    assert len(calls) == 1
    config, output = calls[0]
    assert (config.effective_temperature, config.logg, config.log_carbon_to_helium) == (9347, 8.041, -4.107)
    assert output == "results/dq-9347"


@pytest.mark.parametrize("model", ["DA", "DB", "DAB", "DZ", "DQ"])
def test_notebook_configuration_cells_without_launching_models(model):
    from pathlib import Path
    from wd_spectra import DAConfig, DBConfig, DABConfig, DZConfig, DQConfig
    cells = {cell["id"]: "".join(cell["source"])
             for cell in json.loads(NOTEBOOK.read_text())["cells"]}
    namespace = dict(Path=Path, DAConfig=DAConfig, DBConfig=DBConfig,
                     DABConfig=DABConfig, DZConfig=DZConfig, DQConfig=DQConfig)
    exec(compile(cells["controls"], "<notebook controls>", "exec"), namespace)
    namespace.update(MODEL=model, TEFF=9347., LOGG=8.041)
    exec(compile(cells["request"], "<notebook request>", "exec"), namespace)
    assert isinstance(namespace["config"], namespace[model+"Config"])
    if model == "DQ":
        from wd_spectra.models.dq import validate_config
        validate_config(namespace["config"])
        assert namespace["config"].log_carbon_to_helium == -4.107
