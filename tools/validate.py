"""Tiered validation, bounded parallelism, live logs, and explicit evidence.

No solver checkpoints are loaded by this runner. --reuse reuses a completed
test record, never an atmosphere, and only under an identical numerical key.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SPECTRA = (
    "da-3000",
    "da-4000",
    "da-5000",
    "da-20000",
    "db-10000",
    "db-22000",
    "dab-9000",
    "dab-20000",
    "dz-pg1225",
    "dz-j0738",
    "daz-g149-28",
    "daz-galex1931",
    "dq-j1235",
)
CANARY = "tests/test_protected_model_canaries.py::"
COLD_TESTS = {
    "do-50000": "tests/test_hot_cold_canary.py::test_hot_public_model_converges_from_cold[do-50000-standard]",
    "dao-60000": "tests/test_hot_cold_canary.py::test_hot_public_model_converges_from_cold[dao-60000-standard]",
    **{
        case: f"tests/test_released_family_canaries.py::test_released_family_cold_start[{case}]"
        for case in ("dz-pg1225", "pg1159-pg1424", "d6-j1637")
    },
    "db-10000": CANARY
    + "test_protected_db_cold_starts_converge_without_fallback[10000.0-60]",
    "db-22000": CANARY
    + "test_protected_db_cold_starts_converge_without_fallback[22000.0-60]",
    "db-22000-standard": CANARY
    + "test_standard_db_22000_enters_exact_flux_verification",
    "da-5000": CANARY
    + "test_protected_da_cold_starts_converge_without_fallback[5000.0-45]",
    "da-20000": CANARY
    + "test_protected_da_cold_starts_converge_without_fallback[20000.0-45]",
    "da-3000": CANARY
    + "test_ultracool_da_cold_starts_converge_with_exact_flux_verification[3000.0]",
    "da-4000": CANARY
    + "test_ultracool_da_cold_starts_converge_with_exact_flux_verification[4000.0]",
    "dq-j1235": "tests/test_dq_release.py::test_j1235_public_true_cold_and_independent_spectrum",
}
COLD = tuple(COLD_TESTS) + (
    "dab-20000",
    "dense-db-5000",
    "dense-db-8000",
    "molecular-dab-7500",
    "molecular-dab-9000",
    "molecular-dab-10000",
    "daz-g29-38",
    "daz-g149-28",
    "daz-galex1931",
)


def commands(tier, cases=()):
    pytest = [sys.executable, "-u", "-m", "pytest", "-o", "addopts=-ra", "-s"]
    if tier == "fast":
        return {
            "components": pytest + ["tests", "-m", "not canary and not spectral"],
            "cool-components": pytest + ["research/cool_models"],
        }
    available = SPECTRA if tier == "spectra" else COLD
    if set(cases) - set(available):
        raise ValueError(f"unknown {tier} cases: {sorted(set(cases) - set(available))}")
    tasks = {}
    for case in cases or available:
        if tier == "spectra" and case == "dq-j1235":
            tasks[case] = pytest + [
                "tests/test_dq_spectral_regression.py::test_refractive_spectrum_retains_j1235_absolute_flux"
            ]
        elif tier == "spectra" and case.startswith("daz-"):
            name = "g149_28" if case == "daz-g149-28" else "galex1931"
            tasks[case] = pytest + [
                f"tests/test_daz_regressions.py::test_daz_paper_spectrum[{name}]"
            ]
        elif tier == "spectra":
            tasks[case] = pytest + [
                "tests/test_spectral_regressions.py::"
                f"test_checked_scattering_preserves_broad_spectral_controls[{case}]"
            ]
        elif case in COLD_TESTS:
            tasks[case] = pytest + [COLD_TESTS[case]]
        elif case == "dab-20000":
            tasks[case] = [
                sys.executable,
                "-u",
                "research/validate_public_cold_start.py",
                case,
                "{output}/model",
            ]
        elif case.startswith("daz-"):
            tasks[case] = [
                sys.executable,
                "-u",
                "research/validate_daz_cold_start.py",
                case,
                "{output}/model",
            ]
        else:
            tasks[case] = [
                sys.executable,
                "-u",
                "tools/validate_cool_case.py",
                case,
                "{output}/model",
            ]
    return tasks


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numerical_identity(root=ROOT):
    """Hash actual inputs, including dirty/untracked source and loaded binaries.

    Exclude prose and generated caches. Unit/doc tests always rerun, so prose
    edits cannot borrow a stale documentation pass. Numerical/checker changes
    conservatively invalidate all expensive results, not just guessed consumers.
    """
    inputs = {}
    for folder in ("src", "csrc", "research/cool_models", "tests/data", "tools"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix in {".md", ".pyc"}:
                continue
            inputs[str(path.relative_to(root))] = file_hash(path)
    for name in (
        "pyproject.toml",
        "setup.py",
        "research/validate_public_cold_start.py",
        "research/validate_daz_cold_start.py",
        "research/check_saved_bolometric_spectrum.py",
        "tests/test_protected_model_canaries.py",
        "tests/test_spectral_regressions.py",
        "tests/test_daz_regressions.py",
        "tests/test_dq_spectral_regression.py",
        "tests/test_dq_release.py",
        "tests/conftest.py",
    ):
        if (root / name).is_file():
            inputs[name] = file_hash(root / name)
    data = Path(
        os.environ.get("OPENWD_RESEARCH_DATA", root / ".cache/molecular-opacity")
    )
    for name in ("H2-He_2011.cia", "1H2__RACPPK.states.bz2"):
        inputs["external/" + name] = (
            file_hash(data / name) if (data / name).is_file() else None
        )
    # Custom data roots are deliberately ineligible for reuse until their
    # entire manifest is covered. They may still be tested and recorded.
    runtime = dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        machine=platform.machine(),
        packages=sorted(
            (d.metadata["Name"], d.version) for d in metadata.distributions()
        ),
        environment={
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("OPENWD_", "OMP_", "MKL_", "NUMEXPR_"))
            or key in {"OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"}
        },
    )
    payload = dict(inputs=inputs, runtime=runtime)
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    payload["reuse_eligible"] = not bool(
        os.environ.get("OPENWD_DATA") or os.environ.get("OPENWD_DQ_DATA")
    )
    return payload


def reusable(previous, identity, key, command):
    old = previous.get("tasks", {}).get(key, {})
    return (
        identity["reuse_eligible"]
        and previous.get("inputs_unchanged_during_run") is True
        and previous.get("identity", {}).get("sha256") == identity["sha256"]
        and old.get("status") == "passed"
        and old.get("returncode") == 0
        and old.get("command") == command
        and Path(old.get("log", "")).is_file()
    )


def run_task(command, folder, label, stop, timeout):
    folder.mkdir(parents=True)
    command = [part.replace("{output}", str(folder)) for part in command]
    started = time.monotonic()
    print(f"[{label}] START", flush=True)
    with (folder / "output.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=(os.name != "nt"),
            env=dict(os.environ, OPENWD_TEST_ARTIFACTS=str(folder / "diagnostics")),
        )

        def stream():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[{label}] {line.rstrip()}", flush=True)

        reader = threading.Thread(target=stream, daemon=True)
        reader.start()
        next_report = started + 30
        reason = None
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if stop.is_set() or elapsed > timeout:
                reason = "interrupted" if stop.is_set() else "timed-out"
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                break
            if time.monotonic() >= next_report:
                print(
                    f"[{label}] running, elapsed={elapsed:.0f}s; log={folder / 'output.log'}",
                    flush=True,
                )
                next_report += 30
            stop.wait(0.2)
        process.wait()
        reader.join()
    status = reason or ("passed" if process.returncode == 0 else "failed")
    print(
        f"[{label}] {status.upper()} after {time.monotonic() - started:.1f}s",
        flush=True,
    )
    return dict(
        status=status,
        returncode=process.returncode,
        elapsed_seconds=time.monotonic() - started,
        log=str(folder / "output.log"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", choices=("fast", "spectra", "cold", "full"))
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reuse", type=Path, help="prior summary.json; never reuses atmospheres"
    )
    parser.add_argument(
        "--plan", action="store_true", help="print commands without running or writing"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=7200,
        help="seconds per task; expiry is failure, never convergence",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.timeout <= 0:
        parser.error("jobs and timeout must be positive")
    if args.case and args.tier in {"fast", "full"}:
        parser.error("--case is only for spectra/cold; full must include every control")
    tiers = ("fast", "spectra", "cold") if args.tier == "full" else (args.tier,)
    try:
        stages = {tier: commands(tier, args.case) for tier in tiers}
    except ValueError as exc:
        parser.error(str(exc))
    if args.plan:
        print(json.dumps(stages, indent=2))
        return 0
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OPENWD_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ["PYTHONPATH"] = (
        str(ROOT / "src") + os.pathsep + str(ROOT / "research/cool_models")
    )
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("PYTEST_ADDOPTS", None)
    output = (
        args.output
        or ROOT
        / "results/validation"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    identity = numerical_identity()
    previous = json.loads(args.reuse.read_text()) if args.reuse else {}
    report = dict(
        tier=args.tier,
        identity=identity,
        tasks={},
        status="running",
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    stop = threading.Event()
    old_handlers = {
        s: signal.signal(s, lambda *_: stop.set())
        for s in (signal.SIGINT, signal.SIGTERM)
    }

    def save():
        temporary = output / "summary.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "summary.json")

    save()
    try:
        for tier, tasks in stages.items():
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                pending = {}
                for case, command in tasks.items():
                    key = tier + "/" + case
                    if tier != "fast" and reusable(previous, identity, key, command):
                        report["tasks"][key] = dict(
                            previous["tasks"][key],
                            reused_from=str(args.reuse.resolve()),
                        )
                        print(
                            f"[{key}] REUSED completed test evidence (identical numerical inputs)",
                            flush=True,
                        )
                        continue
                    report["tasks"][key] = dict(status="queued", command=command)

                    def execute(cmd=command, name=key):
                        if stop.is_set():
                            return dict(status="interrupted", returncode=None)
                        return run_task(cmd, output / name, name, stop, args.timeout)

                    pending[pool.submit(execute)] = (key, command)
                save()
                for future in as_completed(pending):
                    key, command = pending[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = dict(status="failed", returncode=None, error=repr(exc))
                    report["tasks"][key] = dict(result, command=command)
                    save()
            if stop.is_set() or any(
                task["status"] != "passed" for task in report["tasks"].values()
            ):
                break  # Never start an expensive stage after a failed cheap gate.
        unchanged = numerical_identity()["sha256"] == identity["sha256"]
        report["inputs_unchanged_during_run"] = unchanged
        report["status"] = (
            "passed"
            if unchanged
            and not stop.is_set()
            and all(task["status"] == "passed" for task in report["tasks"].values())
            else "failed"
        )
        # A pass only applies to this tier/selection. Only full is qualification.
        report["full_qualification"] = (
            args.tier == "full" and report["status"] == "passed"
        )
        save()
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    print(f"{report['status'].upper()}: {output / 'summary.json'}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
