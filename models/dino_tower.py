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

Weights come from checkpoints/ (downloaded by scripts/dl_dino.py: ModelScope
first, huggingface.co fallback) or, if absent, timm's HF-hub channel; either
way runs are fully offline once populated.
"""
import os

import torch
import torch.nn as nn

from ..paths import CKPT_DIR

# timm model name per --weights choice
DINO_MODELS = {
    'dinov2-reg': 'vit_large_patch14_reg4_dinov2.lvd142m',
    'dinov2': 'vit_large_patch14_dinov2.lvd142m',
}

# local checkpoint files (repo convention, offline-friendly — same layout as
# the CLIP checkpoints); downloaded by scripts/dl_dino.py. If absent, timm
# falls back to its HF-hub path (populated local cache works offline too).
LOCAL_CKPTS = {
    'dinov2-reg': os.path.join(CKPT_DIR, 'DINOv2-ViT-L-14-reg4.safetensors'),
    'dinov2': os.path.join(CKPT_DIR, 'DINOv2-ViT-L-14.safetensors'),
}

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IN_MEAN = (0.485, 0.456, 0.406)
IN_STD = (0.229, 0.224, 0.225)


class DinoTower(nn.Module):

    def __init__(self, variant='dinov2-reg'):
        super().__init__()
        import timm  # local import: experiment-only dependency, not needed by CLIP paths
        assert variant in DINO_MODELS, f'unknown dino variant: {variant}'
        # num_classes=0 drops the head; dynamic_img_size=True interpolates
        # pos-embeds to whatever grid the input yields (224->16x16, 512->36x36)
        kwargs = {}
        if os.path.exists(LOCAL_CKPTS[variant]):
            kwargs['pretrained_cfg_overlay'] = dict(file=LOCAL_CKPTS[variant])
        self.net = timm.create_model(DINO_MODELS[variant], pretrained=True,
                                     num_classes=0, dynamic_img_size=True,
                                     **kwargs)
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
        """x: [B,3,H,W] CLIP-normalized (H,W arbitrary; patch grid = floor(H/14)).
        Returns (global [B,1024], dense [B,1024,h,w]) — final-norm patch tokens."""
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
