"""
Day023 & Day024: Causal scaled dot-product attention, numerically stable softmax,
and multi-head Q/K/V/O attention implemented from scratch without nn.MultiheadAttention.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax implemented explicitly from tensor operations:
      shifted = logits - max(logits, dim=dim, keepdim=True)
      exp_x = exp(shifted)
      probs = exp_x / sum(exp_x, dim=dim, keepdim=True)
    """
    max_vals = logits.max(dim=dim, keepdim=True).values
    shifted = logits - max_vals
    exp_scores = torch.exp(shifted)
    denom = exp_scores.sum(dim=dim, keepdim=True)
    return exp_scores / denom


def build_causal_mask(
    query_len: int,
    key_len: int,
    past_len: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a boolean causal mask of shape [1, 1, T_q, T_k] where True means attending is allowed.
    Query index i in [0, T_q) may attend to key index j in [0, T_k) iff j <= past_len + i.
    """
    if query_len <= 0 or key_len <= 0:
        raise ValueError("query_len and key_len must be positive")
    if past_len < 0 or past_len + query_len != key_len:
        raise ValueError(
            f"Invalid KV lengths: past_len={past_len}, query_len={query_len}, key_len={key_len}"
        )

    q_idx = torch.arange(query_len, device=device).unsqueeze(1) + past_len  # [T_q, 1]
    k_idx = torch.arange(key_len, device=device).unsqueeze(0)  # [1, T_k]
    allowed = k_idx <= q_idx  # [T_q, T_k]
    return allowed.view(1, 1, query_len, key_len)


def causal_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    past_len: int = 0,
    dropout_layer: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute causal scaled dot-product attention over 4D tensors:
      q: [B, h, T_q, d_head]
      k: [B, h, T_k, d_head]
      v: [B, h, T_k, d_head]
    Returns:
      (context: [B, h, T_q, d_head], attn_weights: [B, h, T_q, T_k])
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be 4D tensors of shape [B, h, T, d_head]")

    b_q, h_q, t_q, d_head = q.shape
    b_k, h_k, t_k, d_head_k = k.shape
    if (b_q, h_q, d_head) != (b_k, h_k, d_head_k) or k.shape != v.shape:
        raise ValueError(
            f"Incompatible Q/K/V shapes: q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )

    scale = 1.0 / math.sqrt(float(d_head))
    # [B, h, T_q, d_head] @ [B, h, d_head, T_k] -> [B, h, T_q, T_k]
    scores = (q @ k.transpose(-2, -1)) * scale

    allowed_mask = build_causal_mask(
        query_len=t_q,
        key_len=t_k,
        past_len=past_len,
        device=q.device,
    )
    masked_scores = scores.masked_fill(~allowed_mask, float("-inf"))
    attn_weights = stable_softmax(masked_scores, dim=-1)

    if dropout_layer is not None:
        attn_weights_dropped = dropout_layer(attn_weights)
    else:
        attn_weights_dropped = attn_weights

    # [B, h, T_q, T_k] @ [B, h, T_k, d_head] -> [B, h, T_q, d_head]
    context = attn_weights_dropped @ v
    return context, attn_weights


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Day024: From-scratch multi-head causal self-attention with explicit Q, K, V, and output projections.
    Does not use torch.nn.MultiheadAttention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be positive and divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        # Mark residual output projection for 1 / sqrt(2 * n_layers) initialization scaling
        setattr(self.out_proj, "_is_residual_out_proj", True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [B, T, d_model] -> [B, n_heads, T, d_head]."""
        batch_size, seq_len, width = x.shape
        if width != self.d_model:
            raise ValueError(f"Expected last dimension {self.d_model}, got {width}")
        return x.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [B, n_heads, T, d_head] -> [B, T, d_model]."""
        batch_size, heads, seq_len, head_dim = x.shape
        if heads != self.n_heads or head_dim != self.d_head:
            raise ValueError(f"Unexpected head tensor shape: {tuple(x.shape)}")
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for input x of shape [B, T, d_model].
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected 3D input [B, T, {self.d_model}], got {tuple(x.shape)}")

        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))

        context_heads, attn_weights = causal_scaled_dot_product_attention(
            q,
            k,
            v,
            past_len=0,
            dropout_layer=self.attn_dropout,
        )
        merged = self.merge_heads(context_heads)
        out = self.resid_dropout(self.out_proj(merged))

        if return_weights:
            return out, attn_weights
        return out
