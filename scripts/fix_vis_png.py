"""Remap old save_vis PNGs (0/80 masks, zero-filled panel) to the new format.

Old format (run_zeroshot.py before 2026-11): row 2 = GT|pred at 0/80 grey,
row 3 = raw heat|zeros. This script rewrites each PNG in place:
  - GT/pred panels: {0, 80} -> {0, 255}  (pure black/white)
  - row-3 right panel: zeros -> per-tile min-max stretch of the heat panel
Files whose mask panels are not exactly {0, 80} are skipped (already remapped
or unexpected layout).

Usage:
  python -m scripts.fix_vis_png result/zscd/results/vis_*/*_vis.png
"""
import argparse
import glob
import sys

import numpy as np
from PIL import Image

MASK_HI = 80


def remap(arr):
    h3, w2 = arr.shape[0] // 3, arr.shape[1] // 2
    gt, pred = arr[h3:2 * h3, :w2], arr[h3:2 * h3, w2:]
    heat = arr[2 * h3:, :w2]
    for name, a in (('gt', gt), ('pred', pred)):
        if not np.isin(np.unique(a), [0, MASK_HI]).all():
            return None, f'{name} panel has values outside {{0,{MASK_HI}}} - skip'
    out = arr.copy()
    out[h3:2 * h3] = np.where(out[h3:2 * h3] > 0, 255, 0).astype(np.uint8)
    lo, hi = heat.min(), heat.max()
    span = float(hi - lo)
    stretched = ((heat.astype(np.float32) - lo) / (span + 1e-8) * 255).astype(np.uint8)
    out[2 * h3:, w2:] = stretched
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('patterns', nargs='+', help='glob(s) of *_vis.png')
    args = ap.parse_args()

    paths = sorted(p for pat in args.patterns for p in glob.glob(pat))
    if not paths:
        sys.exit('no files matched')
    n_ok = n_skip = 0
    for p in paths:
        arr = np.array(Image.open(p))
        if arr.ndim != 3 or arr.shape[0] % 3 or arr.shape[1] % 2:
            print(f'skip (shape {arr.shape}): {p}')
            n_skip += 1
            continue
        out, err = remap(arr)
        if out is None:
            print(f'{err}: {p}')
            n_skip += 1
            continue
        Image.fromarray(out).save(p)
        n_ok += 1
    print(f'remapped {n_ok}, skipped {n_skip}')


if __name__ == '__main__':
    main()
