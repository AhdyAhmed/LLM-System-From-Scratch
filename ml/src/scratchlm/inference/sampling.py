"""
Day028: Autoregressive greedy and seeded temperature / top-k / top-p sampling
with EOS termination, context-limit enforcement, and stateful incremental UTF-8 streaming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

from scratchlm.model.attention import stable_softmax
from scratchlm.model.gpt import ScratchGPT
from scratchlm.tokenizer.byte_tokenizer import EOS_TOKEN_ID, ByteTokenizer


@dataclass(frozen=True)
class GenerationSampleResult:
    prompt_ids: list[int]
    output_ids: list[int]
    output_text: str
    finish_reason: str  # "stop" | "length"


def sample_next_token(
    logits_1d: torch.Tensor,
    *,
    temperature: float = 0.8,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """
    Sample a single token ID from a 1D vocabulary logit vector [V]:
      - If temperature == 0.0: deterministic greedy argmax.
      - Otherwise: scale by 1/temperature, apply top-k and nucleus top-p filtering
        (always keeping at least the highest-logit token), compute stable_softmax,
        and sample with torch.multinomial.
    """
    if logits_1d.ndim != 1:
        raise ValueError(f"Expected 1D logit vector [V], got {tuple(logits_1d.shape)}")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")

    vocab_size = logits_1d.shape[0]
    if temperature == 0.0:
        return int(torch.argmax(logits_1d).item())

    scaled = logits_1d / float(temperature)

    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        k = min(int(top_k), vocab_size)
        top_vals, _ = torch.topk(scaled, k)
        kth_val = top_vals[-1]
        scaled = torch.where(scaled < kth_val, torch.full_like(scaled, float("-inf")), scaled)

    if top_p is not None:
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0.0, 1.0]")
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            sorted_probs = stable_softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens where cumulative probability prior to this token exceeds top_p
            sorted_remove = (cumulative_probs - sorted_probs) >= float(top_p)
            # Always keep the #1 most probable token
            sorted_remove[0] = False

            remove_mask = torch.zeros_like(scaled, dtype=torch.bool)
            remove_mask.scatter_(dim=0, index=sorted_indices, src=sorted_remove)
            scaled = scaled.masked_fill(remove_mask, float("-inf"))

    probs = stable_softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(sampled.item())


@torch.no_grad()
def generate_tokens(
    model: ScratchGPT,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int = 1337,
    eos_token_id: int = EOS_TOKEN_ID,
    tokenizer: ByteTokenizer | None = None,
) -> GenerationSampleResult:
    """
    Autoregressively generate tokens from ScratchGPT in evaluation mode.
    Explicitly rejects context overflow if len(prompt_ids) + max_new_tokens > model.config.max_seq_len.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    max_seq_len = model.config.max_seq_len
    if len(prompt_ids) + max_new_tokens > max_seq_len:
        raise ValueError(
            f"Context overflow: len(prompt_ids) ({len(prompt_ids)}) + max_new_tokens ({max_new_tokens}) "
            f"> max_seq_len ({max_seq_len})"
        )

    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        gen = torch.Generator(device=device).manual_seed(seed)
        tok = tokenizer or ByteTokenizer()
        stream_decoder = tok.create_stream_decoder()

        seq: list[int] = [int(t) for t in prompt_ids]
        output_ids: list[int] = []
        text_chunks: list[str] = []
        finish_reason = "length"

        for _ in range(max_new_tokens):
            inp = torch.tensor([seq], dtype=torch.long, device=device)
            logits, _ = model(inp)
            next_logits = logits[0, -1, :]
            next_id = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=gen,
            )
            if next_id == eos_token_id:
                finish_reason = "stop"
                break

            seq.append(next_id)
            output_ids.append(next_id)
            delta = stream_decoder.step(next_id)
            if delta:
                text_chunks.append(delta)

        tail = stream_decoder.flush()
        if tail:
            text_chunks.append(tail)

        return GenerationSampleResult(
            prompt_ids=list(prompt_ids),
            output_ids=output_ids,
            output_text="".join(text_chunks),
            finish_reason=finish_reason,
        )
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def stream_generate_utf8(
    model: ScratchGPT,
    prompt_ids: Sequence[int],
    *,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int = 1337,
    eos_token_id: int = EOS_TOKEN_ID,
    tokenizer: ByteTokenizer | None = None,
) -> Iterator[str]:
    """
    Stream decoded UTF-8 string chunks during token-by-token generation using
    IncrementalUtf8TokenStreamDecoder so multi-byte characters split across token steps
    never emit corrupted UTF-8.
    """
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if len(prompt_ids) + max_new_tokens > model.config.max_seq_len:
        raise ValueError(
            f"Context overflow: {len(prompt_ids)} + {max_new_tokens} > {model.config.max_seq_len}"
        )

    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        gen = torch.Generator(device=device).manual_seed(seed)
        tok = tokenizer or ByteTokenizer()
        stream_decoder = tok.create_stream_decoder()

        seq: list[int] = [int(t) for t in prompt_ids]
        for _ in range(max_new_tokens):
            inp = torch.tensor([seq], dtype=torch.long, device=device)
            logits, _ = model(inp)
            next_id = sample_next_token(
                logits[0, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=gen,
            )
            if next_id == eos_token_id:
                break
            seq.append(next_id)
            delta = stream_decoder.step(next_id)
            if delta:
                yield delta

        tail = stream_decoder.flush()
        if tail:
            yield tail
    finally:
        if was_training:
            model.train()
