"""Pre-download the DINO-family checkpoints (thin CLI over zscd.weights_dl).

DinoTower auto-fetches on first use via the same module, so this script is
only for pre-warming (e.g. on a server before a batch sweep): ModelScope for
the timm variants, hf-mirror for SatDINO, huggingface.co as last resort.
Files land in checkpoints/ as DINOv2-ViT-L-14-reg4.safetensors /
DINOv2-ViT-L-14.safetensors / SatDINO-vit_base-16.pth — runs afterwards are
fully offline.

Usage:
  python scripts/dl_dino.py           # all variants
  python scripts/dl_dino.py dinov2-reg
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from zscd.models.dino_tower import DINO_MODELS, LOCAL_CKPTS  # noqa: E402
from zscd.weights_dl import ensure_satdino_ckpt, ensure_timm_ckpt  # noqa: E402

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    names = list(DINO_MODELS) if which == 'all' else [which]
    for n in names:
        dst = LOCAL_CKPTS[n]
        if os.path.exists(dst) and os.path.getsize(dst) > 1e8:
            print(n, 'already present:', dst)
            continue
        if DINO_MODELS[n]['fmt'] == 'satdino':
            ensure_satdino_ckpt(dst)
        else:
            ensure_timm_ckpt(dst, DINO_MODELS[n]['timm'])
        print(n, 'ready:', dst)
