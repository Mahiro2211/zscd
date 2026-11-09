#!/usr/bin/env python
"""Ranking-quality diagnostic: DINOv2 vs CLIP under the per-sample analysis.

Runs scripts/analyze_samples.py for the requested config (default
dinov2-reg + absdiff_norm) on the SAME tile subsample as the reference CLIP
run (deterministic per-domain subsample: same sample-frac / seed), then
prints a side-by-side of the numbers that decided "dense feature quality vs
calibration" in the CLIP diagnosis:

  - per-tile AUC over change tiles (ranking quality, threshold-free)
  - good/mid/bad + no-change clean/mild/severe bucket distribution
  - aggregate fixed@0.5 metrics

Usage:
  python scripts/run_dino_diag.py                          # dinov2-reg absdiff_norm vs georsclip
  python scripts/run_dino_diag.py --weights dinov2 --baseline absdiff_raw
  python scripts/run_dino_diag.py --no-run                 # compare existing per_sample JSONs only
"""
import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(REPO_DIR))

from zscd.paths import RESULTS_DIR  # noqa: E402


def run_analysis(weights, baseline, sample_frac, passthrough):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, 'analyze_samples.py'),
           '--weights', weights, '--baseline', baseline,
           '--sample-frac', str(sample_frac)] + passthrough
    print('[run]', ' '.join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def load_tag(weights, baseline):
    p = os.path.join(RESULTS_DIR, 'per_sample', f'{weights}_{baseline}.json')
    if not os.path.exists(p):
        sys.exit(f'missing {p} — run the analysis first (drop --no-run)')
    with open(p) as f:
        return json.load(f)


def summarize(tag, d):
    recs = d['records']
    chg = [r for r in recs if r['gt_pos'] > 0 and r['auc'] is not None]
    aucs = sorted(r['auc'] for r in chg)
    n = len(aucs)
    med = aucs[n // 2] if n else float('nan')
    mean = sum(aucs) / n if n else float('nan')
    b = {k: sum(r['bucket'] == k for r in recs)
         for k in ('good', 'mid', 'bad', 'nc_clean', 'nc_mild', 'nc_severe')}
    agg = d['aggregate']
    return dict(tag=tag, n_tiles=len(recs), n_chg_auc=n,
                auc_mean=mean, auc_med=med,
                severe_ratio=b['nc_severe'] / max(len(recs) - b['good'] - b['mid'] - b['bad'], 1),
                buckets=b, mF1=agg['mF1'], IoU1=agg['IoU1'])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--weights', default='dinov2-reg')
    ap.add_argument('--baseline', default='absdiff_norm')
    ap.add_argument('--ref-weights', default='georsclip')
    ap.add_argument('--ref-baseline', default='absdiff_norm')
    ap.add_argument('--sample-frac', type=float, default=0.2,
                    help='MUST match the reference run for identical tiles')
    ap.add_argument('--no-run', action='store_true',
                    help='skip analyze_samples, compare existing JSONs only')
    args, passthrough = ap.parse_known_args()

    if not args.no_run:
        rc = run_analysis(args.weights, args.baseline, args.sample_frac, passthrough)
        if rc != 0:
            sys.exit(f'analyze_samples failed (rc={rc})')

    runs = [summarize(f'{args.ref_weights}_{args.ref_baseline}',
                      load_tag(args.ref_weights, args.ref_baseline)),
            summarize(f'{args.weights}_{args.baseline}',
                      load_tag(args.weights, args.baseline))]

    print(f'\n== ranking-quality diagnostic (slide 224/112, fixed thr 0.5, '
          f'sample_frac={args.sample_frac}) ==')
    print(f"{'config':<34}{'AUC mean':>9}{'AUC med':>9}{'mF1':>7}{'IoU1':>7}"
          f"{'good/mid/bad':>15}{'severe-FP%':>11}")
    for r in runs:
        b = r['buckets']
        gmb = f"{b['good']}/{b['mid']}/{b['bad']}"
        print(f"{r['tag']:<34}{r['auc_mean']:>9.3f}{r['auc_med']:>9.3f}"
              f"{r['mF1']:>7.3f}{r['IoU1']:>7.3f}{gmb:>15}"
              f"{r['severe_ratio'] * 100:>10.1f}%")
    print('\nAUC = per-tile threshold-free ranking (change tiles only); '
          'severe-FP% = no-change tiles with pred_pos > 0.2.')


if __name__ == '__main__':
    main()
