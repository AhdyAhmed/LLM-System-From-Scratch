"""
Day026: Pre-norm Transformer block with residual connections, 4d GELU feed-forward network,
and token + learned positional embeddings with explicit context-overflow protection.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from scratchlm.model.attention import CausalMultiHeadSelfAttention
from scratchlm.model.normalization import ScratchLayerNorm


class ScratchMLP(nn.Module):
    """Position-wise feed-forward network: d_model -> d_ff (4 * d_model) -> GELU -> d_model."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if d_ff != 4 * d_model:
            raise ValueError(f"Expected d_ff == 4 * d_model ({4 * d_model}), got {d_ff}")

        self.d_model = d_model
        self.d_ff = d_ff
        self.fc_up = nn.Linear(d_model, d_ff, bias=use_bias)
        self.fc_down = nn.Linear(d_ff, d_model, bias=use_bias)
        # Mark residual output projection for 1 / sqrt(2 * n_layers) initialization scaling
        setattr(self.fc_down, "_is_residual_out_proj", True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc_up(x))
        out = self.fc_down(hidden)
        return self.dropout(out)


class PreNormTransformerBlock(nn.Module):
    """
    Pre-norm decoder Transformer block:
      x = x + attention(ln1(x))
      x = x + mlp(ln2(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        use_bias: bool = True,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.ln1 = ScratchLayerNorm(d_model, eps=layer_norm_eps)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            use_bias=use_bias,
        )
        self.ln2 = ScratchLayerNorm(d_model, eps=layer_norm_eps)
        self.mlp = ScratchMLP(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            use_bias=use_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TokenAndPositionEmbedding(nn.Module):
    """
    Token embedding [V, d_model] + learned absolute positional embedding [C, d_model].
    Explicitly rejects sequence lengths or offsets that exceed max_seq_len (C) or token IDs outside [0, V).
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor, *, past_len: int = 0) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(f"Expected 2D token_ids [B, T], got shape {tuple(token_ids.shape)}")
        batch_size, seq_len = token_ids.shape
        if seq_len <= 0:
            raise ValueError("Sequence length T must be >= 1")
        if past_len < 0 or past_len + seq_len > self.max_seq_len:
            raise ValueError(
                f"Context length overflow: past_len ({past_len}) + seq_len ({seq_len}) > max_seq_len ({self.max_seq_len})"
            )

        min_id = int(token_ids.min().item())
        max_id = int(token_ids.max().item())
        if min_id < 0 or max_id >= self.vocab_size:
            raise ValueError(
                f"Token ID out of range [0, {self.vocab_size}): min={min_id}, max={max_id}"
            )

        positions = torch.arange(
            past_len,
            past_len + seq_len,
            device=token_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)  # [1, T]

        tok_emb = self.token_embedding(token_ids)  # [B, T, d_model]
        pos_emb = self.position_embedding(positions)  # [1, T, d_model]
        return self.dropout(tok_emb + pos_emb)
