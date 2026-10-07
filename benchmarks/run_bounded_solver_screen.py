"""Run an explicit JSON command plan serially with per-process wall limits.

Plans contain a list of {name, command, timeout_seconds, artifact} objects.
Commands are argument lists, never shell text. An artifact is an optional JSON
file written by the benchmark. stdout/stderr, outcomes and elapsed times are
retained even when a benchmark fails or times out. No convergence inference is
made by this runner.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
THREADS = ("OPENWD_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")


def numerical_hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "src/wd_spectra").rglob("*.py"))}


def save(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # The process can exit between poll() and killpg().
    try:
        process.wait(timeout=2.)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not isinstance(plan, list) or not plan:
        parser.error("plan must be a nonempty list")
    names = set()
    for task in plan:
        name = task.get("name")
        command = task.get("command")
        if not name or Path(name).name != name or name in names:
            parser.error("task names must be unique plain filenames")
        names.add(name)
        if not isinstance(command, list) or not command or not all(
                isinstance(argument, str) for argument in command):
            parser.error("commands must be nonempty string argument lists")
        if not 0 < task.get("timeout_seconds", 120) <= 600:
            parser.error("timeout must be positive and no greater than 600 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "run-summary.json"
    env = dict(os.environ, **{name: "1" for name in THREADS})
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "tests")
    initial_hashes = numerical_hashes()
    summary = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkout": str(ROOT), "plan": str(args.plan.resolve()),
        "scope": "Bounded serial solver segments; not cold release qualification",
        "thread_environment": {name: env[name] for name in THREADS},
        "numerical_source_hashes": initial_hashes,
        "status": "running", "records": [],
    }
    save(summary_path, summary)
    try:
        for task in plan:
            name = task["name"]
            print(f"[{name}] START", flush=True)
            stdout_path = args.output / (name + ".stdout")
            stderr_path = args.output / (name + ".stderr")
            started = time.monotonic()
            timeout = task.get("timeout_seconds", 120)
            timed_out = False
            last_progress = started
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                process = subprocess.Popen(task["command"], cwd=ROOT, env=env,
                    stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    while process.poll() is None:
                        now = time.monotonic()
                        if now - started >= timeout:
                            timed_out = True
                            stop_process(process)
                            break
                        if now - last_progress >= 30:
                            print(f"[{name}] running, elapsed={now-started:.0f}s", flush=True)
                            last_progress = now
                        time.sleep(.1)
                finally:
                    stop_process(process)
            record = {
                **task, "returncode": process.returncode,
                "status": "timed-out" if timed_out else (
                    "completed" if process.returncode == 0 else "failed"),
                "elapsed_seconds": time.monotonic() - started,
                "stdout": str(stdout_path.resolve()), "stderr": str(stderr_path.resolve()),
            }
            if task.get("artifact"):
                artifact = Path(task["artifact"])
                if not artifact.is_absolute():
                    artifact = ROOT / artifact
                if artifact.is_file():
                    record["artifact_present"] = True
                else:
                    record["artifact_present"] = False
            summary["records"].append(record)
            save(summary_path, summary)
            print(f"[{name}] {record['status'].upper()} after {record['elapsed_seconds']:.1f}s", flush=True)
    finally:
        summary["numerical_source_unchanged"] = numerical_hashes() == initial_hashes
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        summary["status"] = "completed" if len(summary["records"]) == len(plan) else "interrupted"
        save(summary_path, summary)
    print(str(summary_path.resolve()), flush=True)
    if (not summary["numerical_source_unchanged"] or
            any(record["status"] != "completed" for record in summary["records"])):
        sys.exit(1)


if __name__ == "__main__":
    main()
