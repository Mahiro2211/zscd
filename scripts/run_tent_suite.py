#!/usr/bin/env python
"""Launcher for the TTA comparison suite (motivation.md direction D).

Three stages, each a self-contained set of comparison runs (e2e512 protocol,
fixed-0.5 threshold, same HistAccumulator evaluation as the zero-shot suite):

  0  collapse smoke     8/domain on levir+dsifn. Plain TENT (--tta-div 0) is
                        expected to collapse toward all-'no change' (fixed
                        IoU1 -> ~0); --tta-div 1 and the quantile control
                        should stay healthy. Validates the anti-collapse term
                        before any grid spend.
  1  episodic grid      50/domain x 5 domains, per track (absdiff_norm,
                        notext): no-TTA baseline + quantile control + tent
                        over lr {5e-5, 1e-4, 3e-4} x ln {all, lastk-4}, plus
                        a div=0 collapse reference. Picks lr/ln per track.
  2  full run           all 3,906 pairs with the stage-1 winner: episodic
                        (clean) + continual (the streaming-domain-generalization
                        headline) + quantile control + no-TTA baseline.

Usage (from anywhere):
  python scripts/run_tent_suite.py 0                    # minutes
  python scripts/run_tent_suite.py 0 --limit-per-domain 4
  python scripts/run_tent_suite.py 1                    # ~2-3 h on a 24GB GPU
  python scripts/run_tent_suite.py 2 --lr 1e-4 --ln all --baseline absdiff_norm
  python scripts/run_tent_suite.py 1 --weights dinov2-reg

Notes
- tent runs are sequential (backward pass needs the whole GPU); stage-2
  continual expects ~5-20x the baseline forward time.
- stage 0/1 use --tta-warmup 2 (subsampled streams must leave warmup fast);
  stage 2 uses the default 8.
- unknown extra flags are passed through to zscd.run_zeroshot verbatim
  (e.g. --tta-steps 2, --tta-fp32).
"""
import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)                 # the zscd package/repo dir
PROJ_ROOT = os.environ.get('ZSDG_ROOT', os.path.dirname(REPO_DIR))
LOG_DIR = os.path.join(REPO_DIR, 'results', 'logs')
PY = sys.executable


def env():
    e = dict(os.environ)
    e['ZSDG_ROOT'] = PROJ_ROOT
    e['PYTHONPATH'] = PROJ_ROOT + os.pathsep + e.get('PYTHONPATH', '')
    return e


def run(cmd, log_name):
    """Run one experiment to completion; logs go to results/logs/<log_name>."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, log_name)
    print(f'[run] {log_name}: {" ".join(cmd)}', flush=True)
    t0 = time.time()
    with open(log_path, 'w') as log:
        rc = subprocess.call(cmd, cwd=REPO_DIR, env=env(),
                             stdout=log, stderr=subprocess.STDOUT)
    status = 'OK' if rc == 0 else f'FAILED rc={rc} — see {log_path}'
    print(f'==== {log_name} [{time.time() - t0:.0f}s] {status} ====', flush=True)
    return rc


def e2e_cmd(a, baseline, limit, out, tta=()):
    cmd = [PY, '-m', 'zscd.run_zeroshot', '--weights', a.weights,
           '--baseline', baseline, '--inference', 'resize', '--input-size', '512',
           '--batch', '4', '--save-vis', '0']
    if limit:
        cmd += ['--limit-per-domain', str(limit)]
    if a.domains:
        cmd += ['--domains', a.domains]
    cmd += ['--out', f'results/{out}.json']
    return cmd + list(tta) + a.passthrough


LRS = ('5e-5', '1e-4', '3e-4')
LNS = ('all', 'lastk')


def stage0(a):
    """Collapse smoke: baseline vs tent div=0 vs tent div=1 vs quantile."""
    lim = a.limit_per_domain or 8
    dom = ['--domains', a.domains or 'levir,dsifn']
    a.domains = None                                    # already injected
    tent = ['--tta', 'tent', '--tta-mode', 'episodic', '--tta-warmup', '2']
    runs = [
        ('zs_t0_baseline', e2e_cmd(a, a.baseline, lim, 'zs_t0_baseline', dom)),
        ('zs_t0_tent_div0', e2e_cmd(a, a.baseline, lim, 'zs_t0_tent_div0',
                                    dom + tent + ['--tta-div', '0'])),
        ('zs_t0_tent_div1', e2e_cmd(a, a.baseline, lim, 'zs_t0_tent_div1',
                                    dom + tent + ['--tta-div', '1', '--tta-debug', '2'])),
        ('zs_t0_quantile', e2e_cmd(a, a.baseline, lim, 'zs_t0_quantile',
                                   dom + ['--tta', 'quantile', '--tta-warmup', '2'])),
    ]
    return [run(cmd, f'{name}.log') for name, cmd in runs]


def stage1(a):
    """Episodic grid per track: baseline + quantile + tent(lr x ln) + div0."""
    lim = a.limit_per_domain or 50
    for baseline in ('absdiff_norm', 'notext'):
        tent = ['--tta', 'tent', '--tta-mode', 'episodic']
        runs = [
            (f'zs_t1_{baseline}_baseline',
             e2e_cmd(a, baseline, lim, f'zs_t1_{baseline}_baseline')),
            (f'zs_t1_{baseline}_quantile',
             e2e_cmd(a, baseline, lim, f'zs_t1_{baseline}_quantile',
                     ['--tta', 'quantile'])),
            (f'zs_t1_{baseline}_tent_div0',
             e2e_cmd(a, baseline, lim, f'zs_t1_{baseline}_tent_div0',
                     tent + ['--tta-div', '0'])),
        ]
        for lr in LRS:
            for ln in LNS:
                runs.append((f'zs_t1_{baseline}_tent_lr{lr}_{ln}',
                             e2e_cmd(a, baseline, lim,
                                     f'zs_t1_{baseline}_tent_lr{lr}_{ln}',
                                     tent + ['--tta-div', '1', '--tta-lr', lr,
                                             '--tta-ln', ln, '--tta-debug', '25'])))
        for name, cmd in runs:
            run(cmd, f'{name}.log')


def stage2(a):
    """Full 3,906-pair runs with the stage-1 winner (--lr/--ln/--baseline)."""
    tent = ['--tta', 'tent', '--tta-lr', a.lr, '--tta-ln', a.ln,
            '--tta-debug', '50']
    runs = [
        ('zs_t2_baseline', e2e_cmd(a, a.baseline, 0, 'zs_t2_baseline')),
        ('zs_t2_quantile', e2e_cmd(a, a.baseline, 0, 'zs_t2_quantile',
                                   ['--tta', 'quantile'])),
        ('zs_t2_tent_episodic',
         e2e_cmd(a, a.baseline, 0, 'zs_t2_tent_episodic',
                 tent + ['--tta-mode', 'episodic'])),
        ('zs_t2_tent_continual',
         e2e_cmd(a, a.baseline, 0, 'zs_t2_tent_continual',
                 tent + ['--tta-mode', 'continual'])),
    ]
    for name, cmd in runs:
        run(cmd, f'{name}.log')


def make_report():
    from zscd import make_report  # same interpreter; cwd-independent
    make_report.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', type=int, choices=[0, 1, 2],
                    help='0=collapse smoke, 1=episodic grid, 2=full winner runs')
    ap.add_argument('--weights', default='dinov2',
                    help='dinov2 (main) / dinov2-reg (ablation, insight 6)')
    ap.add_argument('--baseline', default='absdiff_norm',
                    help='track for stages 0/2 (stage 1 sweeps both tracks)')
    ap.add_argument('--lr', default='1e-4', help='stage 2: stage-1 winner lr')
    ap.add_argument('--ln', default='all', choices=LNS,
                    help='stage 2: stage-1 winner LN scope')
    ap.add_argument('--limit-per-domain', type=int, default=0)
    ap.add_argument('--domains', default='',
                    help='override domain list (default: all five / levir,dsifn in stage 0)')
    a, passthrough = ap.parse_known_args()
    a.passthrough = passthrough

    t0 = time.time()
    if a.stage == 0:
        stage0(a)
    elif a.stage == 1:
        stage1(a)
    else:
        stage2(a)
    make_report()
    print(f'total {time.time() - t0:.0f}s — logs in {LOG_DIR}')


if __name__ == '__main__':
    sys.path.insert(0, PROJ_ROOT)   # so `from zscd import make_report` works
    main()
