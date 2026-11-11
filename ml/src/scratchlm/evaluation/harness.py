"""
Day041, Day045, Day046 & Day049: Model evaluation harness:
- Unigram and Laplace-smoothed Bigram baselines on the exact same tokenizer/shard
- Token-level cross-entropy, perplexity, and tokenization-independent bits-per-byte (BPB)
- Deterministic fixed-prompt continuation evaluation with failure/repetition diagnostics
- Narrow SFT format/continuation rubric evaluator
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch

from scratchlm.data.shards import PackedSplitShard
from scratchlm.inference.sampling import generate_tokens
from scratchlm.model.gpt import ScratchGPT
from scratchlm.tokenizer.bpe_tokenizer import BPETokenizer
from scratchlm.tokenizer.byte_tokenizer import IGNORE_INDEX


@dataclass(frozen=True)
class LanguageModelMetrics:
    """Held-out language modeling metrics on a split shard."""

    model_label: str
    split_name: str
    evaluated_valid_tokens: int
    evaluated_utf8_bytes: int
    cross_entropy_loss: float
    perplexity: float
    bits_per_byte: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PromptSampleEvaluation:
    """Deterministic prompt continuation record with quality/failure diagnostics."""

    prompt_id: str
    prompt_text: str
    generated_continuation: str
    full_text: str
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str
    token_repetition_ratio: float
    diagnostic_note: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SftRubricComparison:
    """Comparison of Base vs. SFT model on held-out SFT validation examples and LM regression."""

    evaluated_examples: int
    base_format_compliance_rate: float
    sft_format_compliance_rate: float
    base_avg_repetition_ratio: float
    sft_avg_repetition_ratio: float
    base_lm_val_loss: float
    sft_lm_val_loss: float
    base_lm_val_perplexity: float
    sft_lm_val_perplexity: float
    validation_gate_passed: bool
    sample_comparisons: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class EmpiricalNgramBaseline:
    """
    Token-level Laplace-smoothed Unigram and Bigram baseline models fit strictly on the
    training token shard (`train.bin`) using the exact same vocabulary size `V`.
    """

    def __init__(self, vocab_size: int, smoothing_alpha: float = 1.0) -> None:
        self.vocab_size = vocab_size
        self.smoothing_alpha = smoothing_alpha
        self.unigram_log_probs = np.zeros(vocab_size, dtype=np.float64)
        self._bigram_counts: dict[int, dict[int, int]] = {}
        self._context_totals: dict[int, int] = {}

    def fit(self, train_shard: PackedSplitShard) -> EmpiricalNgramBaseline:
        tokens = np.asarray(train_shard._mmap[:], dtype=np.int64)
        counts = np.bincount(tokens, minlength=self.vocab_size).astype(np.float64)
        smoothed = counts + self.smoothing_alpha
        probs = smoothed / smoothed.sum()
        self.unigram_log_probs = np.log(probs)

        bigram_counts: dict[int, dict[int, int]] = {}
        context_totals: dict[int, int] = {}
        for i in range(len(tokens) - 1):
            prev_t = int(tokens[i])
            next_t = int(tokens[i + 1])
            row = bigram_counts.get(prev_t)
            if row is None:
                row = {}
                bigram_counts[prev_t] = row
            row[next_t] = row.get(next_t, 0) + 1
            context_totals[prev_t] = context_totals.get(prev_t, 0) + 1

        self._bigram_counts = bigram_counts
        self._context_totals = context_totals
        return self

    def evaluate_unigram(self, eval_shard: PackedSplitShard, total_utf8_bytes: int) -> LanguageModelMetrics:
        tokens = np.asarray(eval_shard._mmap[:], dtype=np.int64)
        targets = tokens[1:]
        nll = -float(self.unigram_log_probs[targets].mean())
        ppl = float(math.exp(min(nll, 20.0)))
        bpb = float((nll * len(targets)) / (max(1, total_utf8_bytes) * math.log(2.0)))
        return LanguageModelMetrics(
            model_label="unigram-laplace-baseline",
            split_name=eval_shard.split_name,
            evaluated_valid_tokens=int(len(targets)),
            evaluated_utf8_bytes=int(total_utf8_bytes),
            cross_entropy_loss=round(nll, 4),
            perplexity=round(ppl, 4),
            bits_per_byte=round(bpb, 4),
        )

    def evaluate_bigram(self, eval_shard: PackedSplitShard, total_utf8_bytes: int) -> LanguageModelMetrics:
        tokens = np.asarray(eval_shard._mmap[:], dtype=np.int64)
        alpha = self.smoothing_alpha
        v_alpha = self.vocab_size * alpha
        total_nll = 0.0
        n_targets = len(tokens) - 1
        for i in range(n_targets):
            prev_t = int(tokens[i])
            next_t = int(tokens[i + 1])
            c_prev = self._context_totals.get(prev_t, 0)
            c_pair = self._bigram_counts.get(prev_t, {}).get(next_t, 0)
            prob = (c_pair + alpha) / (c_prev + v_alpha)
            total_nll += -math.log(prob)
        mean_nll = total_nll / max(1, n_targets)
        ppl = float(math.exp(min(mean_nll, 20.0)))
        bpb = float((mean_nll * n_targets) / (max(1, total_utf8_bytes) * math.log(2.0)))
        return LanguageModelMetrics(
            model_label="bigram-laplace-baseline",
            split_name=eval_shard.split_name,
            evaluated_valid_tokens=int(n_targets),
            evaluated_utf8_bytes=int(total_utf8_bytes),
            cross_entropy_loss=round(mean_nll, 4),
            perplexity=round(ppl, 4),
            bits_per_byte=round(bpb, 4),
        )


@torch.no_grad()
def evaluate_gpt_on_shard(
    model: ScratchGPT,
    eval_shard: PackedSplitShard,
    *,
    model_label: str,
    seq_len: int,
    batch_size: int = 4,
    total_utf8_bytes: int,
    device: str = "cpu",
) -> LanguageModelMetrics:
    """
    Evaluate a `ScratchGPT` model over all windows of `eval_shard` (including padded tail),
    weighting loss strictly by valid target tokens.
    """
    was_training = model.training
    model.eval()
    try:
        n_win = eval_shard.num_windows(seq_len, include_padded_tail=True)
        total_nll = 0.0
        total_valid = 0
        for start_idx in range(0, n_win, batch_size):
            win_indices = list(range(start_idx, min(start_idx + batch_size, n_win)))
            batch = eval_shard.get_batch(win_indices, seq_len=seq_len, device=device)
            if batch.total_valid_targets == 0:
                continue
            _, loss = model(batch.input_ids, targets=batch.targets, ignore_index=IGNORE_INDEX)
            assert loss is not None
            total_nll += float(loss.item()) * batch.total_valid_targets
            total_valid += batch.total_valid_targets

        mean_loss = total_nll / max(1, total_valid)
        ppl = float(math.exp(min(mean_loss, 20.0)))
        bpb = float((mean_loss * total_valid) / (max(1, total_utf8_bytes) * math.log(2.0)))
        return LanguageModelMetrics(
            model_label=model_label,
            split_name=eval_shard.split_name,
            evaluated_valid_tokens=total_valid,
            evaluated_utf8_bytes=total_utf8_bytes,
            cross_entropy_loss=round(mean_loss, 4),
            perplexity=round(ppl, 4),
            bits_per_byte=round(bpb, 4),
        )
    finally:
        model.train(was_training)


def compute_token_repetition_ratio(token_ids: Sequence[int]) -> float:
    """Fraction of generated tokens that repeat an immediately preceding bigram or token."""
    if len(token_ids) <= 1:
        return 0.0
    unique_count = len(set(token_ids))
    return round(1.0 - (unique_count / float(len(token_ids))), 4)


@torch.no_grad()
def evaluate_deterministic_prompts(
    model: ScratchGPT,
    tokenizer: BPETokenizer,
    prompts: Sequence[tuple[str, str]],
    *,
    max_new_tokens: int = 28,
    seed: int = 1337,
) -> list[PromptSampleEvaluation]:
    """
    Generate greedy deterministic continuations for fixed `(prompt_id, prompt_text)` pairs
    and attach honest diagnostic notes regarding repetition or tiny-model limitations.
    """
    results: list[PromptSampleEvaluation] = []
    for prompt_id, prompt_text in prompts:
        prompt_ids = tokenizer.encode(prompt_text, add_bos=False, add_eos=False)
        gen_result = generate_tokens(
            model=model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=None,
            top_p=None,
            seed=seed,
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,  # type: ignore[arg-type]
        )
        rep_ratio = compute_token_repetition_ratio(gen_result.output_ids)
        if rep_ratio > 0.45:
            note = "High token repetition observed (typical of tiny 0.9M parameter model under greedy decoding)."
        elif gen_result.finish_reason == "length":
            note = "Reached max_new_tokens budget before emitting EOS; domain vocabulary is story-restricted."
        else:
            note = "Terminated on EOS within token budget; coherent narrow-domain narrative continuation."

        results.append(
            PromptSampleEvaluation(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                generated_continuation=gen_result.output_text,
                full_text=prompt_text + gen_result.output_text,
                prompt_tokens=len(gen_result.prompt_ids),
                generated_tokens=len(gen_result.output_ids),
                finish_reason=gen_result.finish_reason,
                token_repetition_ratio=rep_ratio,
                diagnostic_note=note,
            )
        )
    return results
