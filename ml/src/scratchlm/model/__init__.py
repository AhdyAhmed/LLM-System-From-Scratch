"""From-scratch decoder-only Transformer architecture, attention, LayerNorm, MLP, and initialization."""

from scratchlm.model.attention import (
    CausalMultiHeadSelfAttention,
    build_causal_mask,
    causal_scaled_dot_product_attention,
    stable_softmax,
)
from scratchlm.model.blocks import (
    PreNormTransformerBlock,
    ScratchMLP,
    TokenAndPositionEmbedding,
)
from scratchlm.model.gpt import ScratchGPT, masked_next_token_cross_entropy
from scratchlm.model.normalization import ScratchLayerNorm

__all__ = [
    "CausalMultiHeadSelfAttention",
    "PreNormTransformerBlock",
    "ScratchGPT",
    "ScratchLayerNorm",
    "ScratchMLP",
    "TokenAndPositionEmbedding",
    "build_causal_mask",
    "causal_scaled_dot_product_attention",
    "masked_next_token_cross_entropy",
    "stable_softmax",
]
