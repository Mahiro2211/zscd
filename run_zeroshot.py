"""Zero-shot change detection on WHU-GCD test set II (~/Dataset/test2).

Protocol (aligned with Zan et al. 2025 Sec 5.2):
  - full 512x512 tiles evaluated pixel-wise, metrics mF1 / IoU1 per their Eqs (7)-(8)
  - native-resolution prediction: sliding 224 windows (CLIP stays at its native
    input; no whole-image downsampling)
  - threshold: the FIXED global threshold 0.5 — same operating convention as
    the supervised baselines' built-in 0.5/argmax decision; no labels are
    touched anywhere in the pipeline. Metrics at this threshold are the
    paper-comparable numbers ("fixed"). oracle* (test-label sweep, analysis
    only) is also reported.

Usage:
  python -m zscd.run_zeroshot --weights georsclip                 # semantic, slide
  python -m zscd.run_zeroshot --weights openai --baseline notext  # no-text ablation
  python -m zscd.run_zeroshot --weights georsclip --baseline absdiff_raw   # |d1-d2| rel-L1, raw feats
  python -m zscd.run_zeroshot --weights georsclip --baseline absdiff_norm  # |d1-d2| rel-L1, L2-normalized

Test-time adaptation (resize/e2e mode only; motivation.md direction D):
  python -m zscd.run_zeroshot --weights dinov2 --baseline absdiff_norm \\
      --inference resize --input-size 512 --tta tent --tta-mode episodic
  python -m zscd.run_zeroshot --weights dinov2 --baseline absdiff_norm \\
      --inference resize --input-size 512 --tta quantile            # calibration control
"""
import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from zscd.models.cd_model import SemanticChangeCLIP, change_score
from zscd.models.dino_tower import DenseChangeDINO, DINO_MODELS
from zscd.paths import RESULTS_DIR, DATA_ROOT
from zscd.data.cd_dataset import PairedChangeDataset, clip_denormalize
from zscd.inference import slide_change_prob
from zscd.eval.metrics import HistAccumulator, format_table
from zscd.tta import QuantileCalibrator, TentCD


def save_vis(path, im1, im2, label, prob, thr=0.5):
    """3-row canvas: [im1|im2] / [GT|pred] / [heat|stretch].

    GT/pred are pure black-white (0/255) so change regions are obvious at a
    glance. Row 3 keeps the raw probability heat plus a per-tile min-max
    stretched copy — raw probs often span a narrow band, which reads as a
    flat grey to the eye.
    """
    from PIL import Image
    rows = []
    for img in (im1, im2):
        t = (clip_denormalize(img).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rows.append(t)
    gt = (np.stack([label.numpy()] * 3, -1) * 255).astype(np.uint8)
    pred = (np.stack([(prob.numpy() >= thr)] * 3, -1) * 255).astype(np.uint8)
    p = prob.numpy()
    heat = (np.stack([p] * 3, -1) * 255).astype(np.uint8)
    lo, span = p.min(), p.max() - p.min()
    stretch = (np.stack([(p - lo) / (span + 1e-8)] * 3, -1) * 255).astype(np.uint8)
    canvas = [np.concatenate(rows, axis=1), np.concatenate([gt, pred], axis=1),
              np.concatenate([heat, stretch], axis=1)]
    Image.fromarray(np.concatenate(canvas, axis=0)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True,
                    choices=['openai', 'remoteclip', 'georsclip'] + list(DINO_MODELS))
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--input-size', type=int, default=224, help='CLIP window size')
    ap.add_argument('--stride', type=int, default=112)
    ap.add_argument('--inference', default='slide', choices=['slide', 'resize'])
    ap.add_argument('--baseline', default='semantic',
                    choices=['semantic', 'notext', 'absdiff_raw', 'absdiff_norm'])
    ap.add_argument('--batch-windows', type=int, default=16,
                    help='windows per forward in slide mode (24GB: 16 = all windows in one batch)')
    ap.add_argument('--batch', type=int, default=4, help='pairs per forward (resize mode)')
    ap.add_argument('--limit-per-domain', type=int, default=0)
    ap.add_argument('--domains', default='levir,second,cnam,dsifn,clcd')
    ap.add_argument('--save-vis', type=int, default=3)
    ap.add_argument('--out', default='')
    # test-time adaptation (motivation.md direction D; resize/e2e protocol only)
    ap.add_argument('--tta', default='none', choices=['none', 'tent', 'quantile'])
    ap.add_argument('--tta-lr', type=float, default=1e-4, help='tent: Adam lr on LN affine')
    ap.add_argument('--tta-steps', type=int, default=1, help='tent: adapt steps per batch')
    ap.add_argument('--tta-alpha', type=float, default=8.0,
                    help='tent: sharpening sigmoid(alpha*(c-0.5)) inside the loss')
    ap.add_argument('--tta-div', type=float, default=1.0,
                    help='tent: anti-collapse weight (marginal-entropy term); 0 = plain TENT')
    ap.add_argument('--tta-ln', default='all', choices=['all', 'lastk'],
                    help='tent: which LayerNorms to adapt')
    ap.add_argument('--tta-ln-k', type=int, default=4, help='tent: last-k blocks when --tta-ln lastk')
    ap.add_argument('--tta-mode', default='episodic', choices=['episodic', 'continual'],
                    help='tent: restore source LN params per batch, or adapt over the whole stream')
    ap.add_argument('--tta-q', type=float, default=0.5, help='quantile: anchored quantile')
    ap.add_argument('--tta-warmup', type=int, default=8,
                    help='quantile: maps passed through unchanged before the offset kicks in')
    ap.add_argument('--tta-fp32', action='store_true',
                    help='disable fp16 autocast for the adapted forwards (stability fallback)')
    ap.add_argument('--tta-debug', type=int, default=0,
                    help='tent: print loss / drift every N batches')
    args = ap.parse_args()

    # DINOv2 has no text tower: only the feature-difference tracks are valid
    if args.weights in DINO_MODELS and args.baseline == 'semantic':
        ap.error(f'--weights {args.weights} has no text tower; choose '
                 f'notext / absdiff_raw / absdiff_norm')
    if args.tta != 'none' and args.inference != 'resize':
        ap.error('--tta requires --inference resize (e2e protocol; sliding windows '
                 'would adapt per crop and break the batch-unit semantics)')
    if args.tta == 'tent' and args.baseline == 'semantic':
        ap.error('--tta tent has no differentiable text track; '
                 'use notext / absdiff_raw / absdiff_norm')

    device = 'cuda'
    domains = args.domains.split(',')
    ds = PairedChangeDataset(args.data_root, domains, args.limit_per_domain)
    items = sorted(ds.items)
    print(f'pairs: {len(ds)}; inference={args.inference}, baseline={args.baseline}, tta={args.tta}')

    base = (DenseChangeDINO(args.weights) if args.weights in DINO_MODELS
            else SemanticChangeCLIP(args.weights)).to(device).eval()
    if args.tta == 'tent':
        model = TentCD(base, baseline=args.baseline, lr=args.tta_lr,
                       steps=args.tta_steps, alpha=args.tta_alpha,
                       div_weight=args.tta_div,
                       last_k=args.tta_ln_k if args.tta_ln == 'lastk' else 0,
                       mode=args.tta_mode, debug_every=args.tta_debug)
        n_adapt = sum(p.numel() for p in model.params)
        print(f'tent: adapting {len(model.params)} LN tensors ({n_adapt} params, '
              f'{args.tta_ln if args.tta_ln == "all" else f"lastk-{args.tta_ln_k}"}), '
              f'mode={args.tta_mode}')
    elif args.tta == 'quantile':
        model = QuantileCalibrator(base, baseline=args.baseline,
                                   q=args.tta_q, warmup=args.tta_warmup)
    else:
        model = base

    acc = HistAccumulator(domains)
    vis_saved = {d: 0 for d in domains}
    vis_dir = os.path.join(RESULTS_DIR, f'vis_{args.weights}_{args.baseline}')
    os.makedirs(vis_dir, exist_ok=True)

    def process(items, acc, save_vis_flag):
        t0, t_fwd = time.time(), 0.0
        if args.inference == 'slide':
            for i, (dom, name) in enumerate(items):
                im1, im2, lab, _, _ = ds[ds.item_index(dom, name)]
                im1, im2 = im1.to(device), im2.to(device)
                tf = time.time()
                prob = slide_change_prob(model, im1, im2,
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
                if args.tta == 'none':
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                        change = change_score(model, im1, im2, baseline=args.baseline)
                else:
                    # adapters manage grad/no-grad contexts internally;
                    # fp16 autocast kept for speed unless --tta-fp32
                    amp = (torch.autocast('cuda', dtype=torch.float16)
                           if not args.tta_fp32 else nullcontext())
                    with amp:
                        change = model(im1, im2)
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

    fwd_s = process(items, acc, save_vis_flag=True)

    # per-domain + overall reports at fixed 0.5 (oracle sweep as analysis)
    report = acc.report()

    print('\n' + format_table(report))
    tta_tag = ''
    if args.tta == 'tent':
        tta_tag = (f'_tta-tent-{args.tta_mode}-lr{args.tta_lr:g}-a{args.tta_alpha:g}'
                   f'-d{args.tta_div:g}-{args.tta_ln}')
    elif args.tta == 'quantile':
        tta_tag = f'_tta-quantile-q{args.tta_q:g}'
    out = args.out or os.path.join(
        RESULTS_DIR, f'zs_{args.weights}_{args.baseline}_{args.inference}{tta_tag}.json')
    tta_cfg = None
    if args.tta == 'tent':
        tta_cfg = dict(kind='tent', lr=args.tta_lr, steps=args.tta_steps,
                       alpha=args.tta_alpha, div=args.tta_div, ln=args.tta_ln,
                       ln_k=args.tta_ln_k, mode=args.tta_mode, fp32=args.tta_fp32)
    elif args.tta == 'quantile':
        tta_cfg = dict(kind='quantile', q=args.tta_q, warmup=args.tta_warmup)
    with open(out, 'w') as f:
        json.dump(dict(weights=args.weights, baseline=args.baseline,
                       inference=args.inference, win=args.input_size, stride=args.stride,
                       n_pairs=len(items), tta=tta_cfg,
                       fwd_seconds=fwd_s, protocol='fixed-threshold 0.5, slide native-res',
                       report=report), f, indent=2)
    print('saved ->', out)


if __name__ == '__main__':
    main()
