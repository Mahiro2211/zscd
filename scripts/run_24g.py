#!/usr/bin/env python
"""Pure-Python launcher for the full experiment suite (no shell scripts needed).

24GB-GPU optimal configuration:
  - zero-shot: sliding-window with all 16 windows per forward; the four
    baselines (semantic/notext/absdiff_raw/absdiff_norm) for the same weights
    share the GPU in parallel (~14GB total)
  - finally: aggregate everything into results/report.md

Usage (from anywhere):
  python scripts/run_24g.py                # everything: all zero-shot runs + report
  python scripts/run_24g.py zeroshot       # only the zero-shot runs
  python scripts/run_24g.py report         # just regenerate report.md
  python scripts/run_24g.py zeroshot --limit-per-domain 10   # quick smoke run
"""
import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)                 # the zscd package/repo dir
PROJ_ROOT = os.environ.get('ZSDG_ROOT', os.path.dirname(REPO_DIR))  # dir containing zscd/ + denseclip/
LOG_DIR = os.path.join(REPO_DIR, 'results', 'logs')
PY = sys.executable


def env():
    e = dict(os.environ)
    e['ZSDG_ROOT'] = PROJ_ROOT
    e['PYTHONPATH'] = PROJ_ROOT + os.pathsep + e.get('PYTHONPATH', '')
    return e


def launch(cmd, log_name):
    log = open(os.path.join(LOG_DIR, log_name), 'w')
    print(f'[launch] {os.path.basename(log_name)}: {" ".join(cmd)}', flush=True)
    return subprocess.Popen(cmd, cwd=REPO_DIR, env=env(), stdout=log,
                            stderr=subprocess.STDOUT), log


BASELINES = ['semantic', 'notext', 'absdiff_raw', 'absdiff_norm']


def run_zeroshot_all(common):
    for w in ['openai', 'remoteclip', 'georsclip']:
        base = [PY, '-m', 'zscd.run_zeroshot', '--weights', w,
                '--inference', 'slide', '--stride', '112', '--batch-windows', '16'] + common
        procs = []
        for b in BASELINES:
            procs.append(launch(base + ['--baseline', b], f'zs_{w}_{b}.log'))
        rcs = [p.wait() for p in procs]
        for _, l in procs:
            l.close()
        status = 'OK' if all(rc == 0 for rc in rcs) else \
            f'FAILED (rcs={rcs}) — see results/logs/'
        print(f'==== {w}: {"+".join(BASELINES)} {status} ====', flush=True)
        if any(rc != 0 for rc in rcs):
            sys.exit(1)


def make_report():
    from zscd import make_report  # same interpreter; cwd-independent
    make_report.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', nargs='?', default='all',
                    choices=['all', 'zeroshot', 'report'])
    ap.add_argument('--limit-per-domain', type=int, default=0)
    args, passthrough = ap.parse_known_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    common = []
    if args.limit_per_domain:
        common += ['--limit-per-domain', str(args.limit_per_domain)]
    # extra passthrough flags go to the zscd runners verbatim
    common += passthrough

    t0 = time.time()
    if args.stage in ('all', 'zeroshot'):
        run_zeroshot_all(common)
    if args.stage in ('all', 'report'):
        make_report()
    print(f'total {time.time() - t0:.0f}s — logs in {LOG_DIR}')


if __name__ == '__main__':
    sys.path.insert(0, PROJ_ROOT)   # so `from zscd import make_report` works
    main()
