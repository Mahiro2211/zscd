"""DINOv2 dense tower (timm-backed) with the DenseViT interface contract.

DINOv2 (Oquab et al. 2023, arXiv:2304.07193) is a self-supervised ViT with no
text tower — its patch features are the strongest training-free dense /
correspondence features among open ViTs, which is exactly what the notext /
absdiff change tracks consume (the semantic text track is CLIP-only).

The 'with registers' variant (Darcet et al. 2024, arXiv:2309.16588) is the
default: register tokens absorb the high-norm artifact tokens that otherwise
pollute dense feature maps (and it avoids the plain variant's RoPE-in-last-
two-blocks, which is the riskier readout to reproduce).

Contract (identical to DenseViT): forward(x) -> (global [B, D], dense [B, D, h, w])
so the three no-text change tracks in cd_model.change_score work unchanged.

Input images arrive CLIP-normalized (PairedChangeDataset is shared); we fold
"CLIP denorm -> ImageNet renorm" into one per-channel affine inside the tower,
so the dataset / visualisation / launcher stay untouched.

Weights land in checkpoints/ and are auto-fetched there on first use via
zscd.weights_dl (ModelScope for the timm variants, hf-mirror for SatDINO) —
timm's HF-hub channel is never used, so a bare run on a fresh clone stays on
CN mirrors; once populated, runs are fully offline.
"""
import os

import torch
import torch.nn as nn

from ..paths import CKPT_DIR
from ..weights_dl import ensure_satdino_ckpt, ensure_timm_ckpt

# DINO-family variants: timm architecture + checkpoint format.
#   fmt='timm-hub' : timm-format safetensors (DINOv2 releases)
#   fmt='satdino'  : facebook-DINO training checkpoint (dict['teacher']) from
#                    strakaj/SatDINO — fMoW-RGB self-supervised pretraining,
#                    the RS-finetuned DINO comparison point
DINO_MODELS = {
    'dinov2-reg': dict(timm='vit_large_patch14_reg4_dinov2.lvd142m', fmt='timm-hub'),
    'dinov2': dict(timm='vit_large_patch14_dinov2.lvd142m', fmt='timm-hub'),
    'satdino-b16': dict(timm='vit_base_patch16_224', fmt='satdino'),
}

# local checkpoint files (repo convention, offline-friendly — same layout as
# the CLIP checkpoints); downloaded by scripts/dl_dino.py. If absent, timm
# falls back to its HF-hub path (populated local cache works offline too).
LOCAL_CKPTS = {
    'dinov2-reg': os.path.join(CKPT_DIR, 'DINOv2-ViT-L-14-reg4.safetensors'),
    'dinov2': os.path.join(CKPT_DIR, 'DINOv2-ViT-L-14.safetensors'),
    'satdino-b16': os.path.join(CKPT_DIR, 'SatDINO-vit_base-16.pth'),
}

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)


class DinoTower(nn.Module):

    @staticmethod
    def _load_satdino(net, path):
        """facebook-DINO 'teacher' checkpoint -> timm ViT state dict.

        Block naming (cls_token / pos_embed / blocks.N.attn.qkv / norm) is
        already timm-compatible; three facebook-DINO quirks need fixing: the
        patch-embed conv key differs (and is bias-free while timm defaults to
        bias=True — zero it so the loaded model is exactly equivalent), and
        the distilled training scheme adds a dist_token whose pos_embed row
        must be dropped alongside it (pos rows are cls|dist|patches)."""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt['teacher'] if 'teacher' in ckpt else ckpt
        sd = {k.replace('patch_embed.weight', 'patch_embed.proj.weight'): v
              for k, v in sd.items()}
        sd.pop('dist_token', None)
        if sd['pos_embed'].shape[1] == net.pos_embed.shape[1] + 1:
            sd['pos_embed'] = torch.cat(
                [sd['pos_embed'][:, :1], sd['pos_embed'][:, 2:]], dim=1)
        u, w = net.load_state_dict(sd, strict=False)
        miss = [k for k in u if k != 'patch_embed.proj.bias' and 'head' not in k
                and 'dist_token' not in k]
        assert not miss, f'satdino checkpoint missing keys: {miss[:8]}'
        if 'patch_embed.proj.bias' in u:
            net.patch_embed.proj.bias.zero_()
        print(f'[DinoTower] satdino loaded from {path} '
              f'(ignored: {[k for k in w if "head" not in k][:4]})')

    def __init__(self, variant='dinov2-reg'):
        super().__init__()
        import timm  # local import: experiment-only dependency, not needed by CLIP paths
        assert variant in DINO_MODELS, f'unknown dino variant: {variant}'
        spec = DINO_MODELS[variant]
        if spec['fmt'] == 'satdino':
            # plain timm ViT; weights come solely from the local checkpoint
            self.net = timm.create_model(spec['timm'], pretrained=False,
                                         num_classes=0, dynamic_img_size=True)
            ensure_satdino_ckpt(LOCAL_CKPTS[variant])   # CN mirrors if missing
            self._load_satdino(self.net, LOCAL_CKPTS[variant])
        else:
            # num_classes=0 drops the head; dynamic_img_size=True interpolates
            # pos-embeds to whatever grid the input yields (224->16x16, 512->36x36)
            ensure_timm_ckpt(LOCAL_CKPTS[variant], spec['timm'])
            # overlay => timm reads the local file; its HF-hub channel is never hit
            self.net = timm.create_model(spec['timm'], pretrained=True,
                                         num_classes=0, dynamic_img_size=True,
                                         pretrained_cfg_overlay=dict(file=LOCAL_CKPTS[variant]))
        self.num_prefix = self.net.num_prefix_tokens          # 1 cls (+4 registers)
        # x_in = (x_clip * clip_std + clip_mean - in_mean) / in_std
        #       = a * x_clip + b   (per channel, mathematically exact)
        a = torch.tensor(CLIP_STD) / torch.tensor(IN_STD)
        b = (torch.tensor(CLIP_MEAN) - torch.tensor(IN_MEAN)) / torch.tensor(IN_STD)
        self.register_buffer('_renorm_a', a.view(1, 3, 1, 1), persistent=False)
        self.register_buffer('_renorm_b', b.view(1, 3, 1, 1), persistent=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        """x: [B,3,H,W] CLIP-normalized (H,W arbitrary; grid = floor(H/patch)).
        Returns (global [B,D], dense [B,D,h,w]) — final-norm patch tokens."""
        x = self._renorm_a * x + self._renorm_b
        # timm asserts H,W divisible by patch size; DenseViT floor-crops instead
        # (conv1 drops the remainder) — match that so e2e 512 behaves identically
        p = self.net.patch_embed.patch_size[0]
        gh, gw = x.shape[-2] // p, x.shape[-1] // p
        x = x[..., :gh * p, :gw * p]
        feats = self.net.forward_features(x)                  # [B, 1(+reg)+N, C]
        global_emb = feats[:, 0]
        dense = feats[:, self.num_prefix:]
        B, _, C = dense.shape
        dense = dense.reshape(B, gh, gw, C).permute(0, 3, 1, 2)
        return global_emb, dense


class DenseChangeDINO(nn.Module):
    """DINOv2 counterpart of SemanticChangeCLIP: a frozen dense tower only.

    change_score's notext / absdiff_raw / absdiff_norm tracks access
    model.visual(im) exclusively; the semantic track calls model(im1, im2)
    and is rejected here (no text tower exists)."""

    def __init__(self, variant='dinov2-reg'):
        super().__init__()
        self.visual = DinoTower(variant)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, im1, im2):
        raise NotImplementedError(
            'DINOv2 has no text tower — the semantic baseline is CLIP-only; '
            'use change_score(..., baseline=notext|absdiff_raw|absdiff_norm)')
