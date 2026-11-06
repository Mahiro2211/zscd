"""Vendored CLIP building blocks — makes zscd self-contained.

Copied verbatim from the local DenseCLIP port (denseclip/models/clip_models.py
and denseclip/tokenizer.py, themselves verbatim OpenAI CLIP code) so the zscd
repository runs without the sibling denseclip/ directory. Any change to the
numerics here changes checkpoint compatibility — do not modify.
"""
from .clip_blocks import Transformer, LayerNorm, ResidualAttentionBlock  # noqa: F401
from .tokenizer import tokenize, _tokenizer  # noqa: F401
