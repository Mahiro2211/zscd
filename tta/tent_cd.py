"""Test-time adaptation for the zero-shot change tracks.

Two adapters, both driven purely by the test stream (no labels, no source
data, fixed-0.5 decision rule untouched):

- TentCD: entropy minimization in the TENT lineage (Wang et al., ICLR 2021,
  arXiv:2006.10726) transplanted to the binary change tracks. What is
  adapted: LayerNorm affine params only — ViTs have no BatchNorm, so there
  are no running statistics to refresh and predictions stay batch-
  independent (the batch-size-1 failure mode SAR, Niu et al. ICLR 2023,
  documents for BN-based TENT cannot occur by construction). What is
  minimized: per-pixel binary entropy of a sharpened pseudo-probability,
  plus an anti-collapse term that MAXIMIZES the per-image marginal entropy.
  Plain entropy's trivial minimum on imbalanced binary CD is "no change
  everywhere" (change pixels are 2-10% of a tile); the marginal term is what
  rules that out (SHOT-style diversity, Liang et al. 2020, arXiv:2002.08546).

- QuantileCalibrator: the calibration-only control (motivation.md direction
  A, minimal version). Streams the score distribution into a 256-bin
  histogram and shifts each new map so the running q-quantile sits on the
  0.5 decision anchor. If TentCD cannot beat this one-scalar baseline, its
  parameter budget is not paying for itself — that comparison is the point.

Both wrap the frozen tower and keep the score construction of change_score()
(cd_model.py) unchanged: adaptation moves the scores, not the decision rule.
"""
import re
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.cd_model import change_score

# matches 'blocks.5.' (timm ViT) / 'resblocks.5.' (DenseViT) anywhere in a
# qualified module name; the trailing dot is appended by callers because
# module names end in the layer name ('...blocks.5.norm1').
_BLOCK_RE = re.compile(r'(?:^|\.)(blocks|resblocks)\.(\d+)\.')


def collect_params_ln(model, last_k=0):
    """Collect the LayerNorm affine (weight, bias) params to adapt.

    last_k=0: every LayerNorm in the model (timm blocks.N.norm{1,2} + final
    norm; DenseViT's resblocks.N.ln_{1,2} + ln_pre/ln_post alike — any
    q_norm/k_norm inside attention are LayerNorms too and are included).
    last_k=k: only the last k transformer blocks; non-block norms (the final
    norm, and ln_pre for the CLIP tower) stay included in both modes.

    Returns (params, names); empty if the model has no LayerNorm at all.
    """
    n_blocks = 0
    for nm, _ in model.named_modules():
        m = _BLOCK_RE.search(nm + '.')
        if m:
            n_blocks = max(n_blocks, int(m.group(2)) + 1)
    params, names = [], []
    for nm, mod in model.named_modules():
        if not isinstance(mod, nn.LayerNorm):
            continue
        if last_k > 0:
            m = _BLOCK_RE.search(nm + '.')
            in_last_blocks = m is not None and int(m.group(2)) >= n_blocks - last_k
            is_global_norm = m is None
            if not (in_last_blocks or is_global_norm):
                continue
        for np_, p in mod.named_parameters():
            if np_ in ('weight', 'bias'):
                params.append(p)
                names.append(f'{nm}.{np_}')
    return params, names


class TentCD(nn.Module):
    """TENT adapter for the no-text change tracks (notext / absdiff_*).

    forward(im1, im2) adapts on the batch (entropy + anti-collapse), then
    re-forwards WITHOUT grad for the returned score map — unlike tent.py's
    forward_and_adapt, which returns the pre-update outputs. episodic mode
    restores the source LN params and optimizer state after every batch
    (adaptation never leaks across test samples); continual keeps updating
    over the whole stream (motivation.md direction D).
    """

    def __init__(self, model, baseline='absdiff_norm', lr=1e-4, steps=1,
                 alpha=8.0, div_weight=1.0, last_k=0, mode='episodic',
                 debug_every=0):
        super().__init__()
        assert baseline in ('notext', 'absdiff_raw', 'absdiff_norm'), \
            f'tent has no differentiable track for baseline {baseline!r}'
        assert mode in ('episodic', 'continual')
        self.model = model
        self.baseline = baseline
        self.steps = steps
        self.alpha = alpha
        self.div_weight = div_weight
        self.mode = mode
        self.debug_every = debug_every
        self._b = 0

        # ViT counterpart of tent.py's configure_model: stay in eval mode —
        # no BatchNorm running stats exist to refresh — and open grads only
        # on the LayerNorm affines.
        self.model.eval()
        self.model.requires_grad_(False)
        self.params, self.param_names = collect_params_ln(self.model, last_k)
        assert self.params, 'no LayerNorm affine params found to adapt'
        for p in self.params:
            p.requires_grad_(True)
        self.optimizer = torch.optim.Adam(self.params, lr=lr)

        # episodic restore point: source weights + empty optimizer moments
        self._src = [p.detach().clone() for p in self.params]
        self._opt0 = deepcopy(self.optimizer.state_dict())

    def train(self, mode=True):
        """Keep the tower in eval forever (dropout/BN must not wake up)."""
        return self

    # -- differentiable twin of cd_model's no-text score tracks -------------
    def _score(self, im1, im2):
        """Score map c in [0,1], [B,h,w] — same constructions as
        SemanticChangeCLIP.no_text_baseline / absdiff_baseline, but WITHOUT
        the internal no_grad (cd_model.py wraps those, which would cut the
        adaptation graph)."""
        _, d1 = self.model.visual(im1)
        _, d2 = self.model.visual(im2)
        if self.baseline == 'notext':
            n1 = F.normalize(d1.float(), dim=1)
            n2 = F.normalize(d2.float(), dim=1)
            sim = (n1 * n2).sum(dim=1)
            return ((1.0 - sim) / 2.0).clamp(0, 1)
        d1, d2 = d1.float(), d2.float()
        if self.baseline == 'absdiff_norm':
            d1 = F.normalize(d1, dim=1)
            d2 = F.normalize(d2, dim=1)
        diff = (d1 - d2).abs().sum(dim=1)
        denom = d1.abs().sum(dim=1) + d2.abs().sum(dim=1)
        return (diff / denom.clamp_min(1e-6)).clamp(0, 1)

    def tent_loss(self, c):
        """Binary entropy (minimized) + marginal entropy (maximized).

        The sharpened pseudo-probability p = sigmoid(alpha*(c-0.5)) exists
        only inside this loss; evaluation keeps thresholding the raw score c
        at 0.5, so the protocol is untouched. div = -H(m) where m is the
        per-image mean of p: minimizing div pushes m toward 0.5, which is
        exactly what rules out the all-'no change' collapse (m->0).
        """
        z = self.alpha * (c.float() - 0.5)
        p = torch.sigmoid(z)
        ent = -(p * torch.log(p.clamp_min(1e-8)) +
                (1 - p) * torch.log((1 - p).clamp_min(1e-8)))   # [B,h,w]
        m = p.flatten(1).mean(dim=1)                            # per-image
        div = (m * torch.log(m.clamp_min(1e-8)) +
               (1 - m) * torch.log((1 - m).clamp_min(1e-8)))    # = -H(m), <=0
        loss = ent.mean() + self.div_weight * div.mean()
        return loss, float(ent.mean().detach()), float(div.mean().detach())

    def _delta_w(self):
        with torch.no_grad():
            return float(torch.sqrt(sum(((p - s).float() ** 2).sum()
                                        for p, s in zip(self.params, self._src))))

    def forward(self, im1, im2):
        """Adapt `steps` times on the batch, then predict post-update."""
        loss_v = (float('nan'),) * 3
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            with torch.enable_grad():                # works under outer no_grad
                c = self._score(im1, im2)
                loss, ent_v, div_v = self.tent_loss(c)
            loss.backward()
            self.optimizer.step()
            loss_v = (float(loss.detach()), ent_v, div_v)
        with torch.no_grad():
            out = self._score(im1, im2)              # post-update prediction
        self._b += 1
        if self.debug_every and self._b % self.debug_every == 0:
            print(f'[tent] batch {self._b}: loss={loss_v[0]:.4f} '
                  f'ent={loss_v[1]:.4f} div={loss_v[2]:.4f} '
                  f'|dW|={self._delta_w():.3e}', flush=True)
        if self.mode == 'episodic':
            for p, s in zip(self.params, self._src):
                p.data.copy_(s)
            self.optimizer.load_state_dict(self._opt0)
        return out


class QuantileCalibrator(nn.Module):
    """Streaming quantile calibration (motivation.md direction A, minimal).

    Accumulates a 256-bin histogram of past score maps; each new map is
    shifted so the running q-quantile lands on the 0.5 decision anchor:
    c' = clamp(c - (q_hat - 0.5), 0, 1). The histogram is updated AFTER
    predicting (no peeking at the current map); until `warmup` maps have
    been seen, maps pass through unchanged.
    """

    def __init__(self, model, baseline='absdiff_norm', q=0.5, warmup=8,
                 num_bins=256):
        super().__init__()
        self.model = model
        self.baseline = baseline
        self.q = q
        self.warmup = warmup
        # histogram lives on CPU on purpose: it is float64, tiny, and this
        # keeps the wrapper device-agnostic (it may wrap an already-moved
        # tower without itself being .to(device)'d)
        self.register_buffer('hist', torch.zeros(num_bins, dtype=torch.float64),
                             persistent=False)
        self.n_maps = 0

    def train(self, mode=True):
        return self

    def _offset(self):
        """q_hat - 0.5 from the histogram; 0 during warmup / on empty hist."""
        if self.n_maps < self.warmup:
            return 0.0
        cum = torch.cumsum(self.hist, 0)
        total = cum[-1]
        if total == 0:
            return 0.0
        idx = int(torch.searchsorted(cum, self.q * total))
        idx = min(idx, self.hist.numel() - 1)
        return float((idx + 0.5) / self.hist.numel() - 0.5)

    @torch.no_grad()
    def forward(self, im1, im2):
        c = change_score(self.model, im1, im2, baseline=self.baseline)
        c = c.detach().float()
        off = self._offset()
        out = (c - off).clamp(0, 1) if off != 0.0 else c
        b = (c * (self.hist.numel() - 1)).round().long().flatten().cpu()
        self.hist += torch.bincount(b, minlength=self.hist.numel()).to(self.hist.dtype)
        self.n_maps += c.shape[0]                   # maps, not forward calls
        return out
