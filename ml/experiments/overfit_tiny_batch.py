"""
Day029: Overfit a single tiny batch with ScratchGPT, compare against the empirical
unigram/byte-frequency cross-entropy baseline, verify greedy continuation memorization,
and prove full-model causality (future-token invariance).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from scratchlm.config import load_model_config
from scratchlm.inference.sampling import generate_tokens
from scratchlm.model.gpt import ScratchGPT
from scratchlm.tokenizer.byte_tokenizer import (
    IGNORE_INDEX,
    ByteTokenizer,
    build_shifted_next_token_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_CONFIG_PATH = REPO_ROOT / "ml" / "configs" / "tiny.json"


@dataclass(frozen=True)
class OverfitExperimentResult:
    seed: int
    steps: int
    unigram_baseline_loss: float
    initial_loss: float
    final_loss: float
    loss_curve: list[tuple[int, float]]
    causality_max_abs_diff: float
    greedy_memorized_text: str
    passed: bool


def compute_empirical_unigram_baseline_loss(target_ids: torch.Tensor) -> float:
    """
    Compute the optimal stationary unigram (byte-frequency) cross-entropy baseline
    H(p) = - sum_v p(v) log p(v) over all valid (non-ignored) target positions in the batch.
    No unigram model without context can achieve a lower cross-entropy on this batch.
    """
    valid_targets = [int(t) for t in target_ids.view(-1).tolist() if int(t) != IGNORE_INDEX]
    if not valid_targets:
        raise ValueError("No valid targets to compute unigram baseline")
    counts = Counter(valid_targets)
    total = float(len(valid_targets))
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy


@torch.no_grad()
def verify_full_model_causality(model: ScratchGPT, input_ids: torch.Tensor, pivot_pos: int = 10) -> float:
    """
    Verify that mutating tokens at positions > pivot_pos in evaluation mode produces
    ZERO change in logits at positions 0 .. pivot_pos.
    """
    was_training = model.training
    model.eval()
    try:
        mutated = input_ids.clone()
        mutated[:, pivot_pos + 1 :] = (mutated[:, pivot_pos + 1 :] + 37) % 256

        logits_orig, _ = model(input_ids)
        logits_mut, _ = model(mutated)

        prefix_diff = (logits_orig[:, : pivot_pos + 1, :] - logits_mut[:, : pivot_pos + 1, :]).abs().max()
        return float(prefix_diff.item())
    finally:
        if was_training:
            model.train()


def run_overfit_tiny_batch(
    *,
    steps: int = 140,
    lr: float = 3e-3,
    seed: int = 1337,
) -> OverfitExperimentResult:
    """
    Train the 843,520-parameter tiny ScratchGPT model (with dropout disabled for deterministic
    memorization) on a toy batch of storytelling + UTF-8 sequences and verify that:
      1. Initial loss is near log(264) ~ 5.58
      2. Final loss drops well below the empirical unigram entropy baseline (~3.35) to < 0.15
      3. Full-model causality max_abs_diff == 0.0
      4. Greedy decoding from a prefix reproduces the memorized continuation and stops on EOS
    """
    torch.manual_seed(seed)
    base_cfg = load_model_config(TINY_CONFIG_PATH)
    # Use dropout=0.0 for deterministic single-batch memorization verification
    cfg = type(base_cfg)(
        profile_name=base_cfg.profile_name,
        vocab_size=base_cfg.vocab_size,
        max_seq_len=base_cfg.max_seq_len,
        n_layers=base_cfg.n_layers,
        n_heads=base_cfg.n_heads,
        d_model=base_cfg.d_model,
        d_ff=base_cfg.d_ff,
        dropout=0.0,
        use_bias=base_cfg.use_bias,
        tie_word_embeddings=base_cfg.tie_word_embeddings,
        layer_norm_eps=base_cfg.layer_norm_eps,
    )

    tokenizer = ByteTokenizer()
    toy_texts = [
        "Once upon a time, the little robot counted 1, 2, 3.",
        "Blue stars shine brightly above the quiet valley.",
    ]
    encoded = [tokenizer.encode(t, add_bos=True, add_eos=True) for t in toy_texts]
    shifted = build_shifted_next_token_batch(encoded, max_seq_len=cfg.max_seq_len)

    unigram_baseline = compute_empirical_unigram_baseline_loss(shifted.target_ids)

    model = ScratchGPT(cfg, init_seed=seed)
    causality_diff = verify_full_model_causality(model, shifted.input_ids, pivot_pos=12)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loss_curve: list[tuple[int, float]] = []

    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(shifted.input_ids, shifted.target_ids)
        assert loss is not None
        loss_val = float(loss.item())
        if step == 1 or step % 20 == 0 or step == steps:
            loss_curve.append((step, loss_val))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    initial_loss = loss_curve[0][1]
    final_loss = loss_curve[-1][1]

    # Test greedy continuation from prefix "Once upon a time"
    prefix_ids = tokenizer.encode("Once upon a time", add_bos=True, add_eos=False)
    sample_res = generate_tokens(
        model,
        prefix_ids,
        max_new_tokens=45,
        temperature=0.0,
        seed=seed,
        tokenizer=tokenizer,
    )

    passed = (
        math.isfinite(initial_loss)
        and math.isfinite(final_loss)
        and final_loss < unigram_baseline
        and final_loss < 0.15
        and causality_diff < 1e-6
        and "little robot counted 1, 2, 3." in sample_res.output_text
        and sample_res.finish_reason == "stop"
    )

    return OverfitExperimentResult(
        seed=seed,
        steps=steps,
        unigram_baseline_loss=unigram_baseline,
        initial_loss=initial_loss,
        final_loss=final_loss,
        loss_curve=loss_curve,
        causality_max_abs_diff=causality_diff,
        greedy_memorized_text=sample_res.output_text,
        passed=passed,
    )


if __name__ == "__main__":
    res = run_overfit_tiny_batch()
    print(
        f"[overfit] initial_loss={res.initial_loss:.4f} unigram_baseline={res.unigram_baseline_loss:.4f} "
        f"final_loss={res.final_loss:.4f} causality_diff={res.causality_max_abs_diff:.2e} "
        f"greedy='{res.greedy_memorized_text}' passed={res.passed}"
    )
    for s, l_val in res.loss_curve:
        print(f"  step={s:3d} loss={l_val:.4f}")
    if not res.passed:
        raise SystemExit(1)
