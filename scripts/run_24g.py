#!/usr/bin/env python
"""Pure-Python launcher for the full experiment suite (no shell scripts needed).

24GB-GPU optimal configuration:
  - zero-shot: sliding-window with all 16 windows per forward; the semantic and
    no-text runs for the same weights share the GPU in parallel (~7GB total)
  - TPT: original TPT setting of 64 augmented views (top-10% = 6 confident
    views), chunk 32 visual forwards
  - finally: aggregate everything into results/report.md

Usage (from anywhere):
  python scripts/run_24g.py                # everything: 6 zero-shot + 3 TPT + report
  python scripts/run_24g.py zeroshot       # only the 6 zero-shot runs
  python scripts/run_24g.py tpt            # only the 3 TPT runs
  python scripts/run_24g.py tpt --n-views 32       # lighter TPT variant
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


def run_zeroshot_all(common):
    for w in ['openai', 'remoteclip', 'georsclip']:
        base = [PY, '-m', 'zscd.run_zeroshot', '--weights', w,
                '--inference', 'slide', '--stride', '112', '--batch-windows', '16'] + common
        p1, l1 = launch(base + [], f'zs_{w}_semantic.log')
        p2, l2 = launch(base + ['--baseline', 'notext'], f'zs_{w}_notext.log')
        rc = p1.wait()
        p2.wait()
        l1.close()
        l2.close()
        status = 'OK' if rc == 0 and p2.returncode == 0 else \
            f'FAILED (rc={rc},{p2.returncode}) — see results/logs/'
        print(f'==== {w}: semantic + notext {status} ====', flush=True)
        if rc != 0 or p2.returncode != 0:
            sys.exit(1)


def run_tpt_all(common):
    for w in ['openai', 'remoteclip', 'georsclip']:
        cmd = [PY, '-m', 'zscd.run_tpt', '--weights', w,
               '--tta-steps', '1', '--lr', '5e-3',
               '--chunk', '32', '--batch-windows', '16'] + common
        p, l = launch(cmd, f'tpt_{w}.log')
        rc = p.wait()
        l.close()
        print(f'==== TPT {w}: {"OK" if rc == 0 else "FAILED"} ====', flush=True)
        if rc != 0:
            sys.exit(1)


def make_report():
    from zscd import make_report  # same interpreter; cwd-independent
    make_report.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', nargs='?', default='all',
                    choices=['all', 'zeroshot', 'tpt', 'report'])
    ap.add_argument('--n-views', type=int, default=64)
    ap.add_argument('--limit-per-domain', type=int, default=0)
    ap.add_argument('--sample-per-domain', type=int, default=0)
    ap.add_argument('--calib-frac', type=float, default=0.05)
    args, passthrough = ap.parse_known_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    common = []
    if args.limit_per_domain:
        common += ['--limit-per-domain', str(args.limit_per_domain)]
    if args.sample_per_domain:
        common += ['--sample-per-domain', str(args.sample_per_domain)]
    common += ['--calib-frac', str(args.calib_frac)]
    # extra passthrough flags go to the zscd runners verbatim
    common += passthrough

    t0 = time.time()
    if args.stage in ('all', 'zeroshot'):
        run_zeroshot_all(common)
    if args.stage in ('all', 'tpt'):
        run_tpt_all(common + ['--n-views', str(args.n_views)])
    if args.stage in ('all', 'report'):
        make_report()
    print(f'total {time.time() - t0:.0f}s — logs in {LOG_DIR}')


if __name__ == '__main__':
    sys.path.insert(0, PROJ_ROOT)   # so `from zscd import make_report` works
    main()
