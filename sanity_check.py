"""Sanity check for the three CLIP checkpoints.

For each weights: load, classify real 224x224 crops from test2 with the 6
land-cover prompts, verify (a) crops of obvious classes rank correctly,
(b) logits look sane (no NaN, reasonable spread).
"""
import os as _os, sys as _sys
_ROOT = _os.environ.get('ZSDG_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from zscd.models.cd_model import SemanticChangeCLIP
from zscd.data.cd_dataset import clip_normalize

ROOT = '/home/dhm/Dataset/test2'
# hand-picked obvious crops: (file, i, j) top-left of a 224x224 crop + expected class
PROBES = [
    ('levir_10_2', 'building'),      # LEVIR: building change area
    ('clcd_0', 'farmland'),          # CLCD: cropland/aquaculture
    ('second_100', 'building'),
]
TEXTS = ['building', 'road', 'water', 'bare land', 'forest', 'farmland']


def crop(name, i, j, which='im1'):
    img = Image.open(f'{ROOT}/{which}/{name}.png').convert('RGB').crop((j, i, j + 224, i + 224))
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.
    return clip_normalize(t).unsqueeze(0)


def check(wname):
    print(f'\n================ {wname} ================')
    model = SemanticChangeCLIP(wname).cuda().eval()
    tok_ids = model._tokenize_classes(TEXTS)[0].cuda()
    with torch.no_grad():
        ctx = model._context_from_template('a satellite image of a').cuda()
        tf = F.normalize(model.text.forward_with_context(tok_ids, ctx).float(), dim=-1)
        for name, expect in PROBES:
            x = crop(name, 64, 64).cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                g, d = model.visual(x)
            gn = F.normalize(g.float(), dim=-1)
            sims = (gn @ tf.t())[0]
            top = sims.argmax().item()
            print(f'{name} (expect≈{expect}): top1="{TEXTS[top]}" '
                  f'sim={sims[top]:.3f} | all={[f"{t}:{s:.3f}" for t, s in zip(TEXTS, sims.tolist())]}')
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    for w in (sys.argv[1:] or ['openai', 'remoteclip', 'georsclip']):
        check(w)
