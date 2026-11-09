"""SemanticChangeCLIP: frozen CLIP, dense pixel-text similarity, semantic-consistency
change detection (Zan et al. 2025's 6 land-cover classes).

Change probability per pixel:  change = 1 - sum_c P1(c) * P2(c)
where P1/P2 are the per-date land-cover distributions from pixel-text cosine
similarity (DenseCLIP's score_map, without any trained FPN head).

Text features are zero-shot ensembles: full-sentence prompts, template-averaged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from zscd.vendor.clip_blocks import Transformer, LayerNorm
from zscd.vendor.tokenizer import tokenize, _tokenizer

from .weights import load_checkpoint, split_towers
from .clip_tower import DenseViT
from .text_prompts import LAND_COVER_CLASSES, TEMPLATES


class TextTower(nn.Module):
    """CLIP text tower supporting both plain sentences and context insertion."""

    def __init__(self, width, heads, layers, embed_dim, vocab_size=49408):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(77, width))
        self.transformer = Transformer(width, layers, heads)
        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))

    @torch.no_grad()
    def load_pretrained(self, text_sd):
        pos = text_sd.get('positional_embedding')
        if pos is not None and pos.shape[0] >= 77:
            text_sd = dict(text_sd)
            text_sd['positional_embedding'] = pos[:77]
        u, w = self.load_state_dict(text_sd, strict=False)
        assert not u, f'text tower missing keys: {u[:8]}'

    def _causal_mask(self, L, device):
        m = torch.empty(L, L, device=device)
        m.fill_(float('-inf'))
        m.triu_(1)
        return m

    def _run(self, x, eos_indx):
        """x: [N, L, W] embeddings; eos_indx: [N] pooled positions. Returns [N, D]."""
        N, L, W = x.shape
        mask = self._causal_mask(L, x.device)
        for blk in self.transformer.resblocks:
            blk.attn_mask = mask
        x = x + self.positional_embedding[:L].to(x.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(N, device=x.device), eos_indx] @ self.text_projection
        return x

    def forward_sentences(self, text_ids):
        """text_ids: [K, N] tokenized sentences. Returns [K, D]."""
        x = self.token_embedding(text_ids)
        eos = text_ids.argmax(dim=-1)
        return self._run(x, eos)


class SemanticChangeCLIP(nn.Module):

    def __init__(self, checkpoint, class_names=None, device='cuda', tau=0.07):
        super().__init__()
        sd, arch, logit_scale = load_checkpoint(checkpoint)
        visual_sd, text_sd = split_towers(sd)

        self.visual = DenseViT(arch['patch'], arch['width'], arch['layers'],
                               arch['heads'], arch['output_dim'])
        self.visual.load_pretrained(visual_sd)
        self.text = TextTower(arch['width_t'], arch['heads_t'], arch['layers_t'],
                              arch['output_dim'])
        self.text.load_pretrained(text_sd)
        self.logit_scale = min(float(logit_scale), 4.6052)            # exp <= 100
        # temperature for per-pixel posteriors: CLIP's 100x logit scale makes
        # softmax near one-hot (change prob saturates); DenseCLIP (Zhou et al.,
        # ECCV 2022, arXiv:2112.01071)'s identity loss uses score_map/0.07 —
        # we follow that calibration.
        self.tau = tau

        for p in self.parameters():
            p.requires_grad_(False)

        self.class_names = list(class_names or LAND_COVER_CLASSES)
        self.device = device

        # --- prompts -------------------------------------------------------
        # zero-shot ensemble text features (fixed, per-template averaged);
        # template-mean prompt ensembling follows Radford et al. 2021 (CLIP)
        self.register_buffer('ensemble_text_features',
                             self._build_ensemble_features(), persistent=False)

    # ---------------------------------------------------------------- prompts
    @torch.no_grad()
    def _build_ensemble_features(self):
        feats = []
        for t in TEMPLATES:
            sents = [t.format(c) for c in self.class_names]
            n = max(1 + len(_tokenizer.encode(s)) + 1 for s in sents)
            ids = tokenize(sents, context_length=n)
            f = self.text.forward_sentences(ids)
            feats.append(F.normalize(f.float(), dim=-1))
        return F.normalize(torch.stack(feats).mean(0), dim=-1)          # [K, D]

    def text_features(self):
        """Normalized template-averaged text features [K, D]."""
        return self.ensemble_text_features

    # ---------------------------------------------------------------- forward
    @staticmethod
    def class_logits(dense_feat, text_feat, scale):
        fd = F.normalize(dense_feat.float(), dim=1)
        return scale * torch.einsum('bchw,kc->bkhw', fd, text_feat)

    def forward(self, im1, im2):
        """im1/im2: [B,3,H,W] (H,W multiples of patch size, CLIP-normalized).
        Returns dict: logits1/logits2 [B,K,h,w], p1/p2 [B,K,h,w], change [B,h,w]."""
        with torch.no_grad():
            _, d1 = self.visual(im1)
            _, d2 = self.visual(im2)
        tf = self.text_features()
        scale = 1.0 / self.tau
        logits1 = self.class_logits(d1, tf, scale)
        logits2 = self.class_logits(d2, tf, scale)
        p1 = logits1.softmax(dim=1)
        p2 = logits2.softmax(dim=1)
        change = 1.0 - (p1 * p2).sum(dim=1)
        return dict(logits1=logits1, logits2=logits2, p1=p1, p2=p2, change=change)

    @staticmethod
    def no_text_baseline(im1, im2, model):
        """Cosine similarity of the two dates' dense features (no text involved)."""
        with torch.no_grad():
            _, d1 = model.visual(im1)
            _, d2 = model.visual(im2)
        n1 = F.normalize(d1.float(), dim=1)
        n2 = F.normalize(d2.float(), dim=1)
        return (n1 * n2).sum(dim=1)                                      # [B,h,w]

    @staticmethod
    def absdiff_baseline(im1, im2, model, normalize=False):
        """Supervised-style abs-difference as a relative L1 distance of the two
        dates' dense features. The denominator self-normalizes the score into
        [0,1] (triangle inequality) — required by the 256-bin histogram
        protocol, which clips anything outside [0,1].
        normalize=False: raw features — norm drift between dates counts as
        change; True: per-pixel L2 normalization first, direction change only
        (approximately monotone in the no-text cosine distance)."""
        with torch.no_grad():
            _, d1 = model.visual(im1)
            _, d2 = model.visual(im2)
        d1, d2 = d1.float(), d2.float()
        if normalize:
            d1 = F.normalize(d1, dim=1)
            d2 = F.normalize(d2, dim=1)
        diff = (d1 - d2).abs().sum(dim=1)                                # [B,h,w]
        denom = d1.abs().sum(dim=1) + d2.abs().sum(dim=1)
        return diff / denom.clamp_min(1e-6)                              # [0,1]


def change_score(model, im1, im2, baseline='semantic'):
    """Dispatch a zero-shot change score in [0,1], [B,h,w] — shared by the
    sliding-window and resize inference paths."""
    if baseline == 'semantic':
        return model(im1, im2)['change'].float()
    if baseline == 'notext':
        sim = SemanticChangeCLIP.no_text_baseline(im1, im2, model)
        return ((1.0 - sim) / 2.0).clamp(0, 1)
    if baseline in ('absdiff_raw', 'absdiff_norm'):
        return SemanticChangeCLIP.absdiff_baseline(
            im1, im2, model, normalize=baseline == 'absdiff_norm')
    raise ValueError(f'unknown baseline: {baseline}')
