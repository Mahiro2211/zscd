#!/usr/bin/env python
"""Per-sample prediction-quality analysis for the zero-shot CD pipeline.

run_zeroshot.py compresses every tile into a 2x256 pixel histogram — fine for
paper-comparable aggregate mF1/IoU1, but blind to WHICH tiles work. This
script reruns the SAME protocol (fixed global threshold 0.5, sliding-window
native-resolution inference, no labels touched) while keeping per-tile
metrics, then answers:

  - which tiles are predicted well / badly, and in what proportion
  - where failures concentrate (domain, change-region size, over/under-pred)
  - per-tile threshold-free AUC vs thresholded IoU1 (ranking vs decision)

Bucketing (defaults, CLI-overridable):
  change tiles   : good  IoU1 >= 0.5 / mid 0.2-0.5 / bad < 0.2
  no-change tiles: clean pred_pos <= 0.05 / mild <= 0.2 / severe > 0.2

Outputs (under results/, git-ignored):
  per_sample/{weights}_{baseline}.json      all per-tile records + rankings
  vis_analysis/{weights}_{baseline}/*.png   best / worst / worst-FP canvases
Cross-config markdown: scripts/make_sample_report.py

Usage:
  python scripts/analyze_samples.py --weights georsclip --baseline semantic
  python scripts/analyze_samples.py --weights openai --baseline notext --sample-frac 0.2
"""
import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
PROJ_ROOT = os.environ.get('ZSDG_ROOT', os.path.dirname(REPO_DIR))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from zscd.paths import RESULTS_DIR, DATA_ROOT
from zscd.data.cd_dataset import PairedChangeDataset
from zscd.models.cd_model import SemanticChangeCLIP
from zscd.inference import slide_change_prob
from zscd.eval.metrics import HistAccumulator
from zscd.run_zeroshot import save_vis


def subsample_by_domain(items, frac, seed):
    """Per-domain random subsample of (domain, name) items; >=1 per domain."""
    if frac >= 1.0:
        return sorted(items)
    rng = np.random.RandomState(seed)
    by_dom = {}
    for it in items:
        by_dom.setdefault(it[0], []).append(it)
    out = []
    for dom in sorted(by_dom):
        lst = by_dom[dom]
        n = max(1, int(round(len(lst) * frac)))
        out.extend(lst[i] for i in sorted(rng.permutation(len(lst))[:n]))
    return sorted(out)


def avg_ranks(p):
    """Ties-aware 1-based average ranks, vectorized (for per-tile AUC)."""
    order = np.argsort(p, kind='mergesort')
    sp = p[order]
    new = np.empty(len(p), bool)
    new[0] = True
    new[1:] = sp[1:] != sp[:-1]
    starts = np.flatnonzero(new)
    counts = np.diff(np.append(starts, len(p)))
    ranks = np.empty(len(p))
    ranks[order] = np.repeat(starts + 1 + (counts - 1) / 2.0, counts)
    return ranks


def tile_metrics(prob, lab, thr):
    """Per-tile confusion at `thr` plus IoU1/F1/OA, threshold-free AUC and
    mean-prob separation (change minus no-change pixels)."""
    p, l = prob.ravel(), lab.ravel()
    pred = p >= thr
    pos, neg = l == 1, l == 0
    tp = int((pred & pos).sum()); fp = int((pred & neg).sum())
    fn = int((~pred & pos).sum()); tn = int((~pred & neg).sum())
    n = tp + fp + fn + tn
    n1, n0 = tp + fn, fp + tn
    denom = tp + fp + fn
    iou1 = tp / denom if denom else 1.0        # empty GT + empty pred -> perfect
    auc = None
    if n1 and n0:
        auc = float((avg_ranks(p)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    sep = float(p[pos].mean() - p[neg].mean()) if n1 and n0 else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=n,
                iou1=float(iou1), f1c=2 * tp / (denom + tp) if denom else 1.0,
                oa=(tp + tn) / n,
                prc=tp / max(tp + fp, 1), rc=tp / max(tp + fn, 1),
                gt_pos=n1 / n, pred_pos=(tp + fp) / n, auc=auc, sep=sep)


def bucket_of(m, args):
    if m['gt_pos'] == 0:
        if m['pred_pos'] <= args.fp_clean:
            return 'nc_clean'
        return 'nc_mild' if m['pred_pos'] <= args.fp_mild else 'nc_severe'
    if m['iou1'] >= args.iou_good:
        return 'good'
    return 'mid' if m['iou1'] >= args.iou_mid else 'bad'


def tile_prob(model, ds, dom, name, args):
    """Sliding-window change probability for one pair (CPU tensors out)."""
    im1, im2, lab, _, _ = ds[ds.item_index(dom, name)]
    im1, im2 = im1.to('cuda'), im2.to('cuda')
    prob = slide_change_prob(model, im1, im2, win=args.win,
                             stride=args.stride, baseline=args.baseline,
                             batch_windows=args.batch_windows)
    return im1.cpu(), im2.cpu(), lab, prob.cpu()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--weights', required=True, choices=['openai', 'remoteclip', 'georsclip'])
    ap.add_argument('--baseline', default='semantic',
                    choices=['semantic', 'notext', 'absdiff_raw', 'absdiff_norm'])
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--stride', type=int, default=112)
    ap.add_argument('--batch-windows', type=int, default=16)
    ap.add_argument('--sample-frac', type=float, default=0.2,
                    help='per-domain eval subsample fraction (1.0 = full test2)')
    ap.add_argument('--sample-seed', type=int, default=0)
    ap.add_argument('--iou-good', type=float, default=0.5)
    ap.add_argument('--iou-mid', type=float, default=0.2)
    ap.add_argument('--fp-clean', type=float, default=0.05)
    ap.add_argument('--fp-mild', type=float, default=0.2)
    ap.add_argument('--topk', type=int, default=8, help='per ranking list')
    ap.add_argument('--domains', default='levir,second,cnam,dsifn,clcd')
    ap.add_argument('--limit-per-domain', type=int, default=0)
    args = ap.parse_args()
    args.win = args.input_size

    domains = args.domains.split(',')
    tag = f'{args.weights}_{args.baseline}'
    out_json = os.path.join(RESULTS_DIR, 'per_sample', f'{tag}.json')
    vis_dir = os.path.join(RESULTS_DIR, 'vis_analysis', tag)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    ds = PairedChangeDataset(args.data_root, domains, args.limit_per_domain)
    items = subsample_by_domain(ds.items, args.sample_frac, args.sample_seed)
    ds.items = items
    print(f'{tag}: {len(items)} sampled pairs '
          f'(sample_frac={args.sample_frac}, seed={args.sample_seed})')

    model = SemanticChangeCLIP(args.weights).to('cuda').eval()

    # ---- single pass: per-tile records at the fixed 0.5 threshold ---------
    thr = 0.5
    acc = HistAccumulator(domains)
    records, t0 = [], time.time()
    for k, (dom, name) in enumerate(items):
        _, _, lab, prob = tile_prob(model, ds, dom, name, args)
        acc.update(prob, lab, dom)
        m = tile_metrics(prob.numpy(), lab.numpy(), thr)
        m.update(domain=dom, name=name, bucket=bucket_of(m, args))
        records.append(m)
        if k % 50 == 0:
            print(f'  eval  [{k + 1}/{len(items)}] {time.time() - t0:.0f}s', flush=True)
    eval_seconds = time.time() - t0

    # pixel-level aggregate at the same fixed threshold (sanity vs main run)
    aggregate = acc.report()['overall']['fixed']

    # ---- rankings + pass 3: visualize only the selected tiles -------------
    change = [r for r in records if r['gt_pos'] > 0]
    nochange = [r for r in records if r['gt_pos'] == 0]
    best = sorted(change, key=lambda r: (-r['iou1'], -r['f1c']))[:args.topk]
    worst = sorted(change, key=lambda r: (r['iou1'], r['f1c']))[:args.topk]
    worst_fp = sorted(nochange, key=lambda r: -r['pred_pos'])[:args.topk]

    t0 = time.time()
    for kind, lst in (('best', best), ('worst', worst), ('worstfp', worst_fp)):
        for r in lst:
            im1, im2, lab, prob = tile_prob(model, ds, r['domain'], r['name'], args)
            stem = r['name'][:-4]
            r['img'] = f'vis_analysis/{tag}/{kind}_{stem}_vis.png'
            save_vis(os.path.join(RESULTS_DIR, r['img']), im1, im2, lab, prob, thr)
    print(f'rendered {len(best) + len(worst) + len(worst_fp)} canvases in {time.time() - t0:.0f}s')

    # console bucket summary (per domain)
    print(f'\n{tag} buckets @ thr={thr:.3f}:')
    print(f"{'domain':<9}{'good':>8}{'mid':>8}{'bad':>8} |{'clean':>8}{'mild':>8}{'severe':>8}")
    for d in domains + ['overall']:
        rs = records if d == 'overall' else [r for r in records if r['domain'] == d]
        c = sum(r['bucket'].startswith(('good', 'mid', 'bad')) for r in rs)
        nc = len(rs) - c
        row = [sum(r['bucket'] == b for r in rs) for b in ('good', 'mid', 'bad')]
        row += [sum(r['bucket'] == b for r in rs) for b in ('nc_clean', 'nc_mild', 'nc_severe')]
        print(f'{d:<9}{row[0]:>6}/{c:<2}{row[1]:>6}/{c:<2}{row[2]:>6}/{c:<2} |'
              f'{row[3]:>6}/{nc:<2}{row[4]:>6}/{nc:<2}{row[5]:>6}/{nc:<2}')

    out = dict(weights=args.weights, baseline=args.baseline,
               win=args.win, stride=args.stride,
               sample_frac=args.sample_frac, sample_seed=args.sample_seed,
               threshold=thr, bucket_def=dict(iou_good=args.iou_good, iou_mid=args.iou_mid,
                                              fp_clean=args.fp_clean, fp_mild=args.fp_mild),
               n_pairs=len(items),
               eval_seconds=eval_seconds,
               aggregate=dict(mF1=aggregate['mF1'], IoU1=aggregate['IoU1'],
                              F1_change=aggregate['F1_change']),
               rankings=dict(best=best, worst=worst, worst_fp=worst_fp),
               records=records)
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=1)
    print('saved ->', out_json)


if __name__ == '__main__':
    main()
