"""Zero-shot change detection on WHU-GCD test set II (~/Dataset/test2).

Protocol (aligned with Zan et al. 2025 Sec 5.2):
  - full 512x512 tiles evaluated pixel-wise, metrics mF1 / IoU1 per their Eqs (7)-(8)
  - native-resolution prediction: sliding 224 windows (CLIP stays at its native
    input; no whole-image downsampling)
  - threshold: a SINGLE global threshold selected on a per-domain stratified
    calibration split (5%, fixed seed) of test2 — never on eval pixels. Metrics
    at this frozen threshold are the paper-comparable numbers ("calibrated").
    fixed@0.5 and oracle* (test-label sweep, analysis only) are also reported.

Usage:
  python -m zscd.run_zeroshot --weights georsclip                 # semantic, slide
  python -m zscd.run_zeroshot --weights openai --baseline notext  # no-text ablation
"""
import argparse
import json
import os
import sys
import time

import os as _os, sys as _sys
_ROOT = _os.environ.get('ZSDG_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from zscd.models.cd_model import SemanticChangeCLIP
from zscd.data.cd_dataset import PairedChangeDataset, clip_denormalize
from zscd.inference import slide_change_prob
from zscd.eval.metrics import HistAccumulator, format_table


def save_vis(path, im1, im2, label, prob, thr=0.5):
    from PIL import Image
    h, w = label.shape
    rows = []
    for img in (im1, im2):
        t = (clip_denormalize(img).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rows.append(t)
    gt = (np.stack([label.numpy()] * 3, -1) * 80).astype(np.uint8)
    pred = (np.stack([(prob.numpy() >= thr)] * 3, -1) * 80).astype(np.uint8)
    heat = (np.stack([prob.numpy()] * 3, -1) * 255).astype(np.uint8)
    canvas = [np.concatenate(rows, axis=1), np.concatenate([gt, pred], axis=1),
              np.concatenate([heat, np.zeros_like(heat)], axis=1)]
    Image.fromarray(np.concatenate(canvas, axis=0)).save(path)


def split_calib_eval(ds, calib_frac, seed):
    """Per-domain stratified split by item name. Returns two item lists."""
    rng = np.random.RandomState(seed)
    by_dom = {}
    for it in ds.items:
        by_dom.setdefault(it[0], []).append(it)
    calib, eval_ = [], []
    for dom, lst in by_dom.items():
        idx = rng.permutation(len(lst))
        n_calib = int(round(len(lst) * calib_frac))
        calib.extend(lst[i] for i in sorted(idx[:n_calib]))
        eval_.extend(lst[i] for i in sorted(idx[n_calib:]))
    return sorted(calib), sorted(eval_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, choices=['openai', 'remoteclip', 'georsclip'])
    ap.add_argument('--data-root', default='/home/dhm/Dataset/test2')
    ap.add_argument('--input-size', type=int, default=224, help='CLIP window size')
    ap.add_argument('--stride', type=int, default=112)
    ap.add_argument('--inference', default='slide', choices=['slide', 'resize'])
    ap.add_argument('--mode', default='ensemble', choices=['ensemble', 'context'])
    ap.add_argument('--baseline', default='semantic', choices=['semantic', 'notext'])
    ap.add_argument('--batch-windows', type=int, default=16,
                    help='windows per forward in slide mode (24GB: 16 = all windows in one batch)')
    ap.add_argument('--calib-frac', type=float, default=0.05)
    ap.add_argument('--calib-seed', type=int, default=0)
    ap.add_argument('--batch', type=int, default=4, help='pairs per forward (resize mode)')
    ap.add_argument('--limit-per-domain', type=int, default=0)
    ap.add_argument('--domains', default='levir,second,cnam,dsifn,clcd')
    ap.add_argument('--save-vis', type=int, default=3)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    device = 'cuda'
    domains = args.domains.split(',')
    ds = PairedChangeDataset(args.data_root, domains, args.limit_per_domain)
    calib_items, eval_items = split_calib_eval(ds, args.calib_frac, args.calib_seed)
    print(f'pairs: {len(ds)} = calib {len(calib_items)} + eval {len(eval_items)}; '
          f'inference={args.inference}, baseline={args.baseline}')

    model = SemanticChangeCLIP(args.weights).to(device).eval()

    acc_calib = HistAccumulator(domains)
    acc_eval = HistAccumulator(domains)
    vis_saved = {d: 0 for d in domains}
    vis_dir = f'/home/dhm/zero_shot_dg/zscd/results/vis_{args.weights}_{args.baseline}'
    os.makedirs(vis_dir, exist_ok=True)

    def process(items, acc, save_vis_flag):
        t0, t_fwd = time.time(), 0.0
        if args.inference == 'slide':
            for i, (dom, name) in enumerate(items):
                im1, im2, lab, _, _ = ds[ds.item_index(dom, name)]
                im1, im2 = im1.to(device), im2.to(device)
                tf = time.time()
                prob = slide_change_prob(model, im1, im2, mode=args.mode,
                                         win=args.input_size, stride=args.stride,
                                         baseline=args.baseline,
                                         batch_windows=args.batch_windows)
                t_fwd_l = time.time() - tf
                acc.update(prob.cpu(), lab, dom)
                if save_vis_flag and vis_saved[dom] < args.save_vis:
                    save_vis(os.path.join(vis_dir, f'{name[:-4]}_vis.png'),
                             im1.cpu(), im2.cpu(), lab, prob.cpu())
                    vis_saved[dom] += 1
                if i % 200 == 0:
                    print(f'  [{i + 1}/{len(items)}] {time.time() - t0:.0f}s', flush=True)
                t_fwd += t_fwd_l
        else:
            B = args.batch
            for i in range(0, len(items), B):
                chunk = items[i:i + B]
                batch = [ds[ds.item_index(d, n)] for d, n in chunk]
                im1 = torch.stack([b[0] for b in batch]).to(device)
                im2 = torch.stack([b[1] for b in batch]).to(device)
                H, W = im1.shape[-2:]
                if args.input_size != H:
                    im1 = F.interpolate(im1, size=(args.input_size, args.input_size), mode='bilinear', align_corners=False)
                    im2 = F.interpolate(im2, size=(args.input_size, args.input_size), mode='bilinear', align_corners=False)
                tf = time.time()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                    if args.baseline == 'semantic':
                        out = model(im1, im2, mode=args.mode)
                        change = out['change'].float()
                    else:
                        sim = SemanticChangeCLIP.no_text_baseline(im1, im2, model)
                        change = ((1.0 - sim) / 2.0).clamp(0, 1).float()
                t_fwd_l = time.time() - tf
                change = F.interpolate(change.unsqueeze(1), size=(H, W), mode='bilinear',
                                       align_corners=False).squeeze(1).cpu()
                for k, (_, _, lab, dom, name) in enumerate(batch):
                    acc.update(change[k], lab, dom)
                    if save_vis_flag and vis_saved[dom] < args.save_vis:
                        save_vis(os.path.join(vis_dir, f'{name[:-4]}_vis.png'),
                                 batch[k][0], batch[k][1], lab, change[k])
                        vis_saved[dom] += 1
                if (i // B) % 50 == 0:
                    print(f'  [{i + len(chunk)}/{len(items)}] {time.time() - t0:.0f}s', flush=True)
        return t_fwd

    print('calibration split:')
    process(calib_items, acc_calib, save_vis_flag=False)
    print('eval split:')
    fwd_s = process(eval_items, acc_eval, save_vis_flag=True)

    # per-domain + overall reports with calibrated threshold
    report = {}
    merged_calib = HistAccumulator(['overall'])
    merged_calib.hist['overall'] = sum(acc_calib.hist.values())
    for d in domains:
        report[d] = acc_eval.report_with_calib(acc_calib, domain=d)
    report['overall'] = acc_eval.report_with_calib(merged_calib, domain='overall')

    print('\n' + format_table(report))
    out = args.out or (f'/home/dhm/zero_shot_dg/zscd/results/zs_{args.weights}_{args.baseline}_'
                       f'{args.inference}.json')
    with open(out, 'w') as f:
        json.dump(dict(weights=args.weights, baseline=args.baseline, mode=args.mode,
                       inference=args.inference, win=args.input_size, stride=args.stride,
                       calib_frac=args.calib_frac, calib_seed=args.calib_seed,
                       n_calib=len(calib_items), n_eval=len(eval_items),
                       fwd_seconds=fwd_s, protocol='calib-threshold global, slide native-res',
                       report=report), f, indent=2)
    print('saved ->', out)


if __name__ == '__main__':
    main()
