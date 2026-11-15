"""
Day057: Per-layer Key/Value cache (`LayerKVCache`, `ScratchKVCache`) for incremental autoregressive
decoding in `ScratchGPT`, preserving the uncached forward path as a mathematical reference and
verifying cached vs uncached logit parity within FP32 tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from scratchlm.model.gpt import ScratchGPT


@dataclass
class LayerKVCache:
    """Stores accumulated key and value tensors of shape [B, n_heads, T_cached, d_head] for one layer."""

    k: torch.Tensor | None = None
    v: torch.Tensor | None = None

    @property
    def past_len(self) -> int:
        if self.k is None:
            return 0
        return int(self.k.shape[2])

    def clear(self) -> None:
        self.k = None
        self.v = None


class ScratchKVCache:
    """
    Multi-layer KV cache for decoder-only `ScratchGPT`.
    Enforces:
      - Prefill (`past_len = 0`, input length T_q): stores initial K, V of shape [B, n_heads, T_q, d_head].
      - Incremental decode (`past_len = P`, new query length T_q): concatenates new K, V along dim=2
        to total key length T_k = P + T_q.
      - Explicitly rejects any step where P + T_q > max_seq_len.
    """

    def __init__(self, n_layers: int, max_seq_len: int) -> None:
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.layers: list[LayerKVCache] = [LayerKVCache() for _ in range(n_layers)]

    @property
    def past_len(self) -> int:
        """Return cached sequence length prior to the current forward pass (from layer 0)."""
        return self.layers[0].past_len

    def get_layer_length(self, layer_idx: int) -> int:
        self._validate_layer_idx(layer_idx)
        return self.layers[layer_idx].past_len

    def append(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Append `new_k` and `new_v` (each [B, n_heads, T_q, d_head]) to `layer_idx`.
        Returns `(full_k, full_v, prior_past_len)`.
        Raises ValueError if `prior_past_len + T_q > self.max_seq_len`.
        """
        self._validate_layer_idx(layer_idx)
        if new_k.ndim != 4 or new_v.ndim != 4 or new_k.shape != new_v.shape:
            raise ValueError(
                f"Expected 4D matching new_k/new_v [B, h, T_q, d_head], got {tuple(new_k.shape)} and {tuple(new_v.shape)}"
            )

        layer = self.layers[layer_idx]
        prior_past_len = layer.past_len
        t_q = int(new_k.shape[2])
        if prior_past_len + t_q > self.max_seq_len:
            raise ValueError(
                f"KV cache context overflow at layer {layer_idx}: "
                f"past_len ({prior_past_len}) + query_len ({t_q}) > max_seq_len ({self.max_seq_len})"
            )

        if layer.k is None or layer.v is None:
            layer.k = new_k
            layer.v = new_v
        else:
            if layer.k.shape[0] != new_k.shape[0] or layer.k.shape[1] != new_k.shape[1] or layer.k.shape[3] != new_k.shape[3]:
                raise ValueError(
                    f"Incompatible KV cache batch/head dims: cached={tuple(layer.k.shape)} vs new={tuple(new_k.shape)}"
                )
            layer.k = torch.cat([layer.k, new_k], dim=2)
            layer.v = torch.cat([layer.v, new_v], dim=2)

        return layer.k, layer.v, prior_past_len

    def reset(self) -> None:
        for layer in self.layers:
            layer.clear()

    def total_bytes(self) -> int:
        total = 0
        for layer in self.layers:
            if layer.k is not None:
                total += layer.k.numel() * layer.k.element_size()
            if layer.v is not None:
                total += layer.v.numel() * layer.v.element_size()
        return total

    def _validate_layer_idx(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, {self.n_layers})")


@dataclass(frozen=True)
class KVCacheParityReport:
    """Verification report comparing cached incremental decoding vs uncached full-prefix recomputation."""

    prompt_len: int
    decode_steps: int
    max_abs_logit_diff: float
    tolerance: float
    passed: bool
    greedy_tokens_match: bool


@torch.no_grad()
def verify_kv_cache_parity(
    model: ScratchGPT,
    prompt_ids: Sequence[int],
    *,
    decode_steps: int = 8,
    atol: float = 1e-5,
) -> KVCacheParityReport:
    """
    Verify that prefill + step-by-step KV-cached decoding produces identical next-token logits
    at every step compared to full-sequence uncached recomputation within FP32 tolerance `atol`.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if decode_steps < 1:
        raise ValueError("decode_steps must be >= 1")
    if len(prompt_ids) + decode_steps > model.config.max_seq_len:
        raise ValueError("len(prompt_ids) + decode_steps exceeds model.config.max_seq_len")

    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        kv_cache = ScratchKVCache(
            n_layers=model.config.n_layers,
            max_seq_len=model.config.max_seq_len,
        )

        seq_uncached = [int(t) for t in prompt_ids]
        seq_cached = [int(t) for t in prompt_ids]
        max_diff = 0.0
        greedy_match = True

        # Step 0: Prefill entire prompt into KV cache and compare last-position logits with uncached forward
        prompt_tensor = torch.tensor([seq_cached], dtype=torch.long, device=device)
        logits_cached, _ = model(prompt_tensor, kv_cache=kv_cache)
        logits_uncached, _ = model(prompt_tensor, kv_cache=None)

        step0_diff = float((logits_cached[:, -1, :] - logits_uncached[:, -1, :]).abs().max().item())
        max_diff = max(max_diff, step0_diff)

        next_token_cached = int(torch.argmax(logits_cached[0, -1, :]).item())
        next_token_uncached = int(torch.argmax(logits_uncached[0, -1, :]).item())
        if next_token_cached != next_token_uncached:
            greedy_match = False

        seq_cached.append(next_token_cached)
        seq_uncached.append(next_token_uncached)

        # Incremental decode steps 1 .. decode_steps - 1
        for _ in range(1, decode_steps):
            single_token_inp = torch.tensor([[seq_cached[-1]]], dtype=torch.long, device=device)
            full_seq_inp = torch.tensor([seq_uncached], dtype=torch.long, device=device)

            step_logits_cached, _ = model(single_token_inp, kv_cache=kv_cache)
            step_logits_uncached, _ = model(full_seq_inp, kv_cache=None)

            diff = float(
                (step_logits_cached[:, -1, :] - step_logits_uncached[:, -1, :]).abs().max().item()
            )
            max_diff = max(max_diff, diff)

            tok_c = int(torch.argmax(step_logits_cached[0, -1, :]).item())
            tok_u = int(torch.argmax(step_logits_uncached[0, -1, :]).item())
            if tok_c != tok_u:
                greedy_match = False

            seq_cached.append(tok_c)
            seq_uncached.append(tok_u)

        passed = (max_diff <= atol) and greedy_match
        return KVCacheParityReport(
            prompt_len=len(prompt_ids),
            decode_steps=decode_steps,
            max_abs_logit_diff=max_diff,
            tolerance=atol,
            passed=passed,
            greedy_tokens_match=greedy_match,
        )
    finally:
        if was_training:
            model.train()
