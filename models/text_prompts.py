"""Text prompts for zero-shot semantic-consistency change detection.

Class system follows Zan et al. 2025 (WHU-GCD): 6 land-cover categories
(building / road / water / barren / forest / agriculture), see notes_paper.md.
"""
import os as _os, sys as _sys
_ROOT = _os.environ.get('ZSDG_ROOT', _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import torch
from denseclip.tokenizer import tokenize  # noqa: E402

# WHU-GCD 6 land-cover classes (paper Table 1); farmland == "agriculture"
LAND_COVER_CLASSES = ['building', 'road', 'water', 'bare land', 'forest', 'farmland']

# extra classes seen in test2 domains (SECOND: playground/low vegetation; CNAM: impervious surface)
EXTRA_CLASSES = ['grass', 'tree', 'playground', 'parking lot', 'bridge']

# prompt templates: {} is replaced by the class name
TEMPLATES = [
    'a satellite image of a {}',
    'an aerial image of a {}',
    'a remote sensing image of a {}',
    'a photo of a {} from a bird\'s-eye view',
]

# template used to initialize the TPT learnable context (no {})
TPT_INIT_TEMPLATE = 'a satellite image of a'


def build_class_tokens(class_names, extra_context_len=0):
    """Tokenize each class name (SOT ... EOT, zero-padded) to a fixed length N1.
    Returns LongTensor [K, N1] and N1."""
    lengths = [1 + len(sys.modules['denseclip.tokenizer']._tokenizer.encode(c)) + 1 for c in class_names]
    n1 = max(lengths)
    return tokenize(class_names, context_length=n1), n1


def init_context_vectors(text_encoder, template=TPT_INIT_TEMPLATE):
    """Context vectors for the learnable prompt, initialized from the token
    embeddings of a hand-crafted template (TPT-style, ctx_init='a photo of a')."""
    toks = tokenize([template], context_length=64)[0]
    eot = int((toks == 49407).nonzero()[0])
    ids = toks[1:eot]                       # drop SOT / EOT
    with torch.no_grad():
        vecs = text_encoder.token_embedding(ids)   # [n_ctx, width]
    return vecs.unsqueeze(0).clone()        # [1, n_ctx, width]


def embed_templates(text_encoder, texts_per_template):
    """text_encoder: CLIPTextContextEncoder-like with forward(text[K,N1], context).
    For zero-shot ensembling: build full prompts 'template.format(class)', tokenize
    to one fixed length, use zero-length context via empty insertion is not possible,
    so we tokenize the FULL sentence and pass an empty context tensor."""
    raise NotImplementedError
