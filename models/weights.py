"""Unified CLIP checkpoint loading for three weight families.

Supported:
  - openai    : OpenAI JIT .pt checkpoints (torch.jit.script modules)
                e.g. ~/.cache/clip/ViT-L-14.pt
  - openclip  : open_clip-format plain .pt state dicts
                e.g. RemoteCLIP-ViT-L-14.pt, RS5M/GeoRSCLIP ViT-L-14-336

All formats share the same layout for the towers themselves:
  visual: visual.conv1.weight / visual.class_embedding / visual.positional_embedding /
          visual.ln_pre.* / visual.transformer.resblocks.* / visual.ln_post.* / visual.proj
  text  : token_embedding.weight / positional_embedding / ln_final.* /
          text_projection / transformer.resblocks.*
  scalar: logit_scale
open_clip newer versions may name the patch embed `visual.patch_embed.proj.*`;
we rename it back to `visual.conv1.*`.
"""
import os

import torch

from ..paths import CKPT_DIR, OPENAI_CLIP_CACHE

# name -> (path, format, arch)
WEIGHT_SPECS = {
    'openai': dict(path=os.path.join(OPENAI_CLIP_CACHE, 'ViT-L-14.pt'), fmt='openai', arch='ViT-L-14'),
    'openai-b16': dict(path=os.path.join(OPENAI_CLIP_CACHE, 'ViT-B-16.pt'), fmt='openai', arch='ViT-B-16'),
    'remoteclip': dict(path=os.path.join(CKPT_DIR, 'RemoteCLIP-ViT-L-14.pt'),
                       fmt='openclip', arch='ViT-L-14'),
    'georsclip': dict(path=os.path.join(CKPT_DIR, 'GeoRSCLIP-ViT-L-14-336.pt'),
                      fmt='openclip', arch='ViT-L-14-336'),
}

# arch -> (patch, width, layers, heads, output_dim, source_input_res, text_width, text_heads, text_layers)
ARCHS = {
    'ViT-B-32':   dict(patch=32, width=768,  layers=12, heads=12, output_dim=512,
                       width_t=512,  heads_t=8,  layers_t=12),
    'ViT-B-16':   dict(patch=16, width=768,  layers=12, heads=12, output_dim=512,
                       width_t=512,  heads_t=8,  layers_t=12),
    'ViT-L-14':   dict(patch=14, width=1024, layers=24, heads=16, output_dim=768,
                       width_t=768,  heads_t=12, layers_t=12),
    'ViT-L-14-336': dict(patch=14, width=1024, layers=24, heads=16, output_dim=768,
                       width_t=768,  heads_t=12, layers_t=12),
}


def _load_raw(path, fmt):
    if fmt == 'openai':
        return torch.jit.load(path, map_location='cpu').state_dict()
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    return {k: v.float() for k, v in sd.items() if isinstance(v, torch.Tensor)}


def load_checkpoint(name_or_path, family=None, family_arch=None):
    """Return (state_dict, arch_cfg, logit_scale) with keys normalized:
    visual tower keys prefixed 'visual.', text tower keys bare."""
    if name_or_path in WEIGHT_SPECS:
        spec = WEIGHT_SPECS[name_or_path]
        path, fmt, arch = spec['path'], spec['fmt'], spec['arch']
    else:
        path = name_or_path
        fmt = family or 'openclip'
        arch = family_arch or 'ViT-L-14'
    raw = _load_raw(path, fmt)
    sd = {}
    for k, v in raw.items():
        # open_clip newer patch-embed naming -> OpenAI naming
        if k.startswith('visual.patch_embed.proj.'):
            k = k.replace('visual.patch_embed.proj.', 'visual.conv1.')
        sd[k] = v
    logit_scale = sd.pop('logit_scale', torch.tensor(4.6052)).reshape(()).clone()
    return sd, ARCHS[arch], float(logit_scale)


def split_towers(sd):
    """Split normalized state dict into (visual_sd, text_sd) with prefixes stripped
    so each matches the module parameter names directly."""
    visual, text = {}, {}
    for k, v in sd.items():
        if k.startswith('visual.'):
            visual[k[len('visual.'):]] = v
        elif k.startswith('transformer.') or k.startswith('token_embedding') \
                or k.startswith('ln_final') or k == 'positional_embedding' or k == 'text_projection':
            text[k] = v
    return visual, text
