"""Slim dense-capable CLIP ViT visual tower.

Reuses denseclip's Transformer/LayerNorm blocks (numerically identical to the port)
but, unlike denseclip's CLIPVisionTransformer, works for ANY patch size (14/16/32)
and returns only what zero-shot dense prediction needs:
    global embedding [B, D]  and  dense patch embeddings [B, D, h, w]
(= DenseCLIP's 5th backbone output, without the trained FPN pyramid).

Parameter names deliberately match OpenAI CLIP (conv1, class_embedding, ...)
so checkpoints load with a plain load_state_dict after prefix stripping.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
_ROOT = os.environ.get('ZSDG_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from denseclip.models.clip_models import Transformer, LayerNorm  # noqa: E402


class DenseViT(nn.Module):

    def __init__(self, patch_size, width, layers, heads, output_dim, drop_path_rate=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.width = width
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # placeholder; real values come from the checkpoint (grid inferred there)
        self.positional_embedding = nn.Parameter(scale * torch.randn(197, width))
        self.grid = 14  # sqrt(len(pos)-1), set on load
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.no_grad()
    def load_pretrained(self, visual_sd, source_input_res=None):
        pos = visual_sd.get('positional_embedding')
        if pos is not None and pos.dim() == 2:
            self.grid = int(round(math.sqrt(pos.shape[0] - 1)))
            self.positional_embedding = nn.Parameter(pos.clone())
        u, w = self.load_state_dict(visual_sd, strict=False)
        # every parameter must be covered except nothing — visual tower has no extras
        missing = [k for k in u if k != '']
        assert not missing, f'visual tower missing keys: {missing[:8]}'
        if w:
            print(f'[DenseViT] ignored unexpected keys: {w[:8]}')

    def forward(self, x: torch.Tensor):
        """x: [B,3,H,W] with H,W multiples of patch_size. Returns (global [B,D], dense [B,D,h,w])."""
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # B, HW, C
        cls = self.class_embedding.to(x.dtype) + torch.zeros(B, 1, C, dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        if H != self.grid or W != self.grid:
            spatial = F.interpolate(
                pos[1:].reshape(1, self.grid, self.grid, C).permute(0, 3, 1, 2),
                size=(H, W), mode='bilinear')
            spatial = spatial.reshape(1, C, H * W).permute(0, 2, 1)
            pos = torch.cat([pos[0:1].expand(B, 1, C), spatial.expand(B, H * W, C)], dim=1)
        else:
            pos = pos.unsqueeze(0).expand(B, H * W + 1, C)
        x = x + pos

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)          # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)          # LND -> NLD
        x = self.ln_post(x)
        x = x @ self.proj

        return x[:, 0], x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)
