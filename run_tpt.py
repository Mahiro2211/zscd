"""Dense TPT experiments on WHU-GCD test set II — protocol-aligned runner.

Same protocol as run_zeroshot.py: sliding-window native-resolution prediction
(with the per-pair tuned prompt, mode='context'), per-domain stratified
calibration split selects the single global threshold; metrics at the frozen
threshold are the paper-comparable numbers.

Usage:
  python -m zscd.run_tpt --weights georsclip --n-views 32
  python -m zscd.run_tpt --weights openai --sample-per-domain 300   # budget mode
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, '/home/dhm/zero_shot_dg')
import torch

from zscd.models.cd_model import SemanticChangeCLIP
from zscd.data.cd_dataset import PairedChangeDataset
from zscd.tpt.dense_tpt import tpt_predict_pair
from zscd.inference import slide_change_prob
from zscd.run_zeroshot import save_vis, split_calib_eval
from zscd.eval.metrics import HistAccumulator, format_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, choices=['openai', 'remoteclip', 'georsclip'])
    ap.add_argument('--data-root', default='/home/dhm/Dataset/test2')
    ap.add_argument('--win', type=int, default=224)
    ap.add_argument('--stride', type=int, default=112)
    ap.add_argument('--n-views', type=int, default=32)
    ap.add_argument('--tta-steps', type=int, default=1)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--selection-p', type=float, default=0.1)
    ap.add_argument('--crop-size', type=int, default=224)
    ap.add_argument('--crop-scale-min', type=float, default=0.3)
    ap.add_argument('--calib-frac', type=float, default=0.05)
    ap.add_argument('--calib-seed', type=int, default=0)
    ap.add_argument('--sample-per-domain', type=int, default=0,
                    help='if >0: stratified subset (seed 0) of this many pairs per domain first')
    ap.add_argument('--domains', default='levir,second,cnam,dsifn,clcd')
    ap.add_argument('--save-vis', type=int, default=3)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    device = 'cuda'
    domains = args.domains.split(',')
    ds = PairedChangeDataset(args.data_root, domains, args.sample_per_domain)
    calib_items, eval_items = split_calib_eval(ds, args.calib_frac, args.calib_seed)
    print(f'pairs: {len(ds)} = calib {len(calib_items)} + eval {len(eval_items)}; '
          f'TPT views={args.n_views}, steps={args.tta_steps}')

    model = SemanticChangeCLIP(args.weights).to(device).eval()
    acc_calib = HistAccumulator(domains)
    acc_eval = HistAccumulator(domains)
    vis_saved = {d: 0 for d in domains}
    vis_dir = f'/home/dhm/zero_shot_dg/zscd/results/vis_{args.weights}_tpt'
    os.makedirs(vis_dir, exist_ok=True)

    def process(items, acc, save_vis_flag):
        t0 = time.time()
        loss_log = []
        for i, (dom, name) in enumerate(items):
            im1, im2, lab, _, _ = ds[ds.item_index(dom, name)]
            im1, im2 = im1.to(device), im2.to(device)
            losses, _ = tpt_predict_pair(model, im1, im2, n_views=args.n_views,
                                         tta_steps=args.tta_steps, lr=args.lr,
                                         selection_p=args.selection_p,
                                         crop_size=args.crop_size,
                                         crop_scale=(args.crop_scale_min, 1.0))
            prob = slide_change_prob(model, im1, im2, mode='context',
                                     win=args.win, stride=args.stride)   # tuned prompt
            acc.update(prob.cpu(), lab, dom)
            loss_log.append(losses)
            if save_vis_flag and vis_saved[dom] < args.save_vis:
                save_vis(os.path.join(vis_dir, f'{name[:-4]}_vis.png'),
                         im1.cpu(), im2.cpu(), lab, prob.cpu())
                vis_saved[dom] += 1
            if i % 100 == 0:
                print(f'  [{i + 1}/{len(items)}] {time.time() - t0:.0f}s '
                      f'loss={losses[-1]:.4f}', flush=True)
        return loss_log, time.time() - t0

    print('calibration split:')
    process(calib_items, acc_calib, save_vis_flag=False)
    print('eval split:')
    loss_log, dur = process(eval_items, acc_eval, save_vis_flag=True)

    report = {}
    merged_calib = HistAccumulator(['overall'])
    merged_calib.hist['overall'] = sum(acc_calib.hist.values())
    for d in domains:
        report[d] = acc_eval.report_with_calib(acc_calib, domain=d)
    report['overall'] = acc_eval.report_with_calib(merged_calib, domain='overall')

    print('\n' + format_table(report))
    import numpy as np
    ll = np.array(loss_log)
    out = args.out or (f'/home/dhm/zero_shot_dg/zscd/results/tpt_{args.weights}_v{args.n_views}'
                       f'_s{args.tta_steps}_slide.json')
    with open(out, 'w') as f:
        json.dump(dict(weights=args.weights, n_views=args.n_views, tta_steps=args.tta_steps,
                       lr=args.lr, selection_p=args.selection_p, win=args.win, stride=args.stride,
                       calib_frac=args.calib_frac, calib_seed=args.calib_seed,
                       n_calib=len(calib_items), n_eval=len(eval_items),
                       seconds=dur, loss_mean_last=ll[:, -1].mean() if ll.size else None,
                       protocol='calib-threshold global, slide native-res, TPT context-mode',
                       report=report), f, indent=2)
    print('saved ->', out)


if __name__ == '__main__':
    main()
