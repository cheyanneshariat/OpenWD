"""Run one scientific process at a time with RSS and elapsed-time cutoffs."""
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rss-gib', type=float, default=2)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cwd', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','OPENWD_NUM_THREADS',
                'MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        environment[key] = '1'
    start = time.monotonic()
    peak = 0
    reason = None
    with (args.output / 'run.log').open('w') as log:
        child = subprocess.Popen(command, cwd=args.cwd, env=environment,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print('START', child.pid, str(args.output), flush=True)
        try:
            while child.poll() is None:
                measurement = subprocess.run(['/bin/ps','-o','rss=','-p',str(child.pid)],
                    capture_output=True,text=True)
                if measurement.returncode == 0 and measurement.stdout.strip():
                    peak = max(peak,int(measurement.stdout.strip()))
                elapsed = time.monotonic() - start
                if peak > args.rss_gib*1024**2 or elapsed > args.seconds:
                    reason = 'RSS cutoff' if peak > args.rss_gib*1024**2 else 'elapsed-time cutoff'
                    raise RuntimeError(reason)
                (args.output/'watchdog.json').write_text(json.dumps(dict(
                    pid=child.pid,peak_RSS_KiB=peak,elapsed_seconds=elapsed,
                    RSS_limit_GiB=args.rss_gib,time_limit_seconds=args.seconds,
                    threads=1,status='running'))+'\n')
                time.sleep(.25)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
            raise
        finally:
            (args.output/'watchdog.json').write_text(json.dumps(dict(
                pid=child.pid,peak_RSS_KiB=peak,elapsed_seconds=time.monotonic()-start,
                RSS_limit_GiB=args.rss_gib,time_limit_seconds=args.seconds,
                threads=1,exit_code=child.poll(),stopped_reason=reason,status='finished'))+'\n')
        print('EXIT',child.returncode,'peak MiB',round(peak/1024),flush=True)
        if child.returncode: raise SystemExit(child.returncode)

if __name__ == '__main__': main()
