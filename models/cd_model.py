"""SemanticChangeCLIP: frozen CLIP, dense pixel-text similarity, semantic-consistency
change detection (Zan et al. 2025's 6 land-cover classes).

Change probability per pixel:  change = 1 - sum_c P1(c) * P2(c)
where P1/P2 are the per-date land-cover distributions from pixel-text cosine
similarity (DenseCLIP's score_map, without any trained FPN head).

Two text modes:
  - ensemble (zero-shot): full-sentence prompts, template-averaged text features
  - context  (TPT):       [SOT] + learnable ctx + class + [EOT], ctx initialized
                          from a hand-crafted template (TPT's ctx_init)
"""
import os, sys
_ROOT = os.environ.get('ZSDG_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import torch
import torch.nn as nn
import torch.nn.functional as F

from denseclip.models.clip_models import Transformer, LayerNorm      # noqa: E402
from denseclip.tokenizer import tokenize                              # noqa: E402

from .weights import load_checkpoint, split_towers
from .clip_tower import DenseViT
from .text_prompts import LAND_COVER_CLASSES, TEMPLATES, TPT_INIT_TEMPLATE


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

    def forward_with_context(self, text_ids, context):
        """text_ids: [K, N1]; context: [1, n_ctx, W]. Returns [K, D]."""
        K, N1, W = self.token_embedding(text_ids).shape
        n_ctx = context.shape[1]
        x_text = self.token_embedding(text_ids)                       # K,N1,W
        eos = (text_ids.argmax(dim=-1) + n_ctx)                       # shifted by insertion
        x = torch.cat([x_text[:, 0:1], context.expand(K, -1, -1), x_text[:, 1:]], dim=1)
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
        # softmax near one-hot (change prob saturates); DenseCLIP's identity
        # loss uses score_map/0.07 — we follow that calibration.
        self.tau = tau

        for p in self.parameters():
            p.requires_grad_(False)

        self.class_names = list(class_names or LAND_COVER_CLASSES)
        self.device = device

        # --- prompts -------------------------------------------------------
        texts, self.n1 = self._tokenize_classes(self.class_names)        # [K, N1]
        self.register_buffer('texts', texts, persistent=False)
        # learnable context (TPT), initialized from hand-crafted template
        init = self._context_from_template(TPT_INIT_TEMPLATE)
        self.contexts = nn.Parameter(init)                               # [1, n_ctx, W]
        self._contexts_init = init.detach().clone()

        # zero-shot ensemble text features (fixed, per-template averaged)
        self.register_buffer('ensemble_text_features',
                             self._build_ensemble_features(), persistent=False)

    # ---------------------------------------------------------------- prompts
    def _tokenize_classes(self, names):
        toks = [1 + len(sys.modules['denseclip.tokenizer']._tokenizer.encode(n)) + 1 for n in names]
        n1 = max(toks)
        return tokenize(names, context_length=n1), n1

    def _context_from_template(self, template):
        toks = tokenize([template], context_length=77)[0]
        eot = int((toks == 49407).nonzero()[0])
        ids = toks[1:eot].to(self.text.token_embedding.weight.device)
        return self.text.token_embedding(ids).detach().unsqueeze(0).clone()

    @torch.no_grad()
    def _build_ensemble_features(self):
        feats = []
        for t in TEMPLATES:
            sents = [t.format(c) for c in self.class_names]
            n = max(1 + len(sys.modules['denseclip.tokenizer']._tokenizer.encode(s)) + 1 for s in sents)
            ids = tokenize(sents, context_length=n)
            f = self.text.forward_sentences(ids)
            feats.append(F.normalize(f.float(), dim=-1))
        return F.normalize(torch.stack(feats).mean(0), dim=-1)          # [K, D]

    def reset(self):
        with torch.no_grad():
            self.contexts.copy_(self._contexts_init)

    def text_features(self, mode='ensemble', contexts=None):
        """Returns normalized [K, D]; grad flows in 'context' mode."""
        if mode == 'ensemble':
            return self.ensemble_text_features
        ctx = self.contexts if contexts is None else contexts
        f = self.text.forward_with_context(self.texts, ctx)
        return F.normalize(f.float(), dim=-1)

    # ---------------------------------------------------------------- forward
    @staticmethod
    def class_logits(dense_feat, text_feat, scale):
        fd = F.normalize(dense_feat.float(), dim=1)
        return scale * torch.einsum('bchw,kc->bkhw', fd, text_feat)

    def forward(self, im1, im2, mode='ensemble'):
        """im1/im2: [B,3,H,W] (H,W multiples of patch size, CLIP-normalized).
        Returns dict: logits1/logits2 [B,K,h,w], p1/p2 [B,K,h,w], change [B,h,w]."""
        with torch.no_grad():
            _, d1 = self.visual(im1)
            _, d2 = self.visual(im2)
        tf = self.text_features(mode)
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
