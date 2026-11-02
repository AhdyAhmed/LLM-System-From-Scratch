"""
Day021: Tensor dimensions, broadcasting, batched matrix products, and numerical finite-difference
gradient verification on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class TensorBasicsReport:
    batch_size: int
    seq_len: int
    d_model: int
    n_heads: int
    d_head: int
    max_finite_diff_error: float
    passed: bool


def verify_tensor_shapes_and_broadcasting(
    batch_size: int = 2,
    seq_len: int = 6,
    d_model: int = 16,
    n_heads: int = 4,
    seed: int = 1337,
) -> dict[str, tuple[int, ...]]:
    """
    Demonstrate and assert [B, T, d] -> [B, h, T, d_head] reshaping, batched attention score
    computation [B, h, T, T], causal mask broadcasting [1, 1, T, T], and head merging [B, T, d].
    """
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    d_head = d_model // n_heads

    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((batch_size, seq_len, d_model), generator=gen, dtype=torch.float32)
    w_q = torch.randn((d_model, d_model), generator=gen, dtype=torch.float32) * 0.02
    w_k = torch.randn((d_model, d_model), generator=gen, dtype=torch.float32) * 0.02
    w_v = torch.randn((d_model, d_model), generator=gen, dtype=torch.float32) * 0.02

    # [B, T, d] @ [d, d] -> [B, T, d]
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    assert q.shape == (batch_size, seq_len, d_model)

    # Split heads: [B, T, h, d_head] -> [B, h, T, d_head]
    q_heads = q.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    k_heads = k.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    v_heads = v.view(batch_size, seq_len, n_heads, d_head).transpose(1, 2)
    assert q_heads.shape == (batch_size, n_heads, seq_len, d_head)

    # Scaled dot-product scores: [B, h, T, d_head] @ [B, h, d_head, T] -> [B, h, T, T]
    scores = (q_heads @ k_heads.transpose(-2, -1)) / (d_head ** 0.5)
    assert scores.shape == (batch_size, n_heads, seq_len, seq_len)

    # Broadcast [1, 1, T, T] lower-triangular causal mask across batch and heads
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).view(
        1, 1, seq_len, seq_len
    )
    masked_scores = scores.masked_fill(~causal_mask, float("-inf"))
    weights = torch.softmax(masked_scores, dim=-1)
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6)

    # Merge heads: [B, h, T, d_head] -> [B, T, h, d_head] -> [B, T, d]
    context_heads = weights @ v_heads
    merged = context_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    assert merged.shape == (batch_size, seq_len, d_model)

    return {
        "input": tuple(x.shape),
        "q_heads": tuple(q_heads.shape),
        "scores": tuple(scores.shape),
        "mask": tuple(causal_mask.shape),
        "merged": tuple(merged.shape),
    }


def central_finite_difference_grad(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute numerical gradient via central finite differences:
      df/dx_i approx (f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps)
    Uses float64 internally so truncation error does not dominate.
    """
    x64 = x.detach().to(torch.float64).clone()
    num_grad = torch.zeros_like(x64)
    flat_x = x64.view(-1)
    flat_grad = num_grad.view(-1)

    for idx in range(flat_x.numel()):
        orig = flat_x[idx].item()
        flat_x[idx] = orig + eps
        loss_plus = fn(x64).item()
        flat_x[idx] = orig - eps
        loss_minus = fn(x64).item()
        flat_x[idx] = orig
        flat_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)

    return num_grad


def run_tensor_basics_check(seed: int = 1337) -> TensorBasicsReport:
    """Run shape assertions and compare autograd against central finite differences."""
    shapes = verify_tensor_shapes_and_broadcasting(seed=seed)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((2, 3, 4), generator=gen, dtype=torch.float64, requires_grad=True)
    w = torch.randn((4, 4), generator=gen, dtype=torch.float64)

    def scalar_objective(inp: torch.Tensor) -> torch.Tensor:
        # Smooth non-linear projection + layer normalization style reduction
        centered = inp - inp.mean(dim=-1, keepdim=True)
        var = (centered * centered).mean(dim=-1, keepdim=True)
        normed = centered / torch.sqrt(var + 1e-5)
        proj = torch.tanh(normed @ w)
        return (proj * proj).mean()

    loss = scalar_objective(x)
    loss.backward()
    assert x.grad is not None
    analytic_grad = x.grad.detach().clone()
    numeric_grad = central_finite_difference_grad(scalar_objective, x, eps=1e-6)

    max_err = float((analytic_grad - numeric_grad).abs().max().item())
    passed = max_err < 1e-7 and shapes["merged"] == (2, 6, 16)
    return TensorBasicsReport(
        batch_size=2,
        seq_len=6,
        d_model=16,
        n_heads=4,
        d_head=4,
        max_finite_diff_error=max_err,
        passed=passed,
    )


if __name__ == "__main__":
    report = run_tensor_basics_check()
    print(
        f"[tensor_basics] shapes=PASS max_finite_diff_error={report.max_finite_diff_error:.3e} passed={report.passed}"
    )
    if not report.passed:
        raise SystemExit(1)
