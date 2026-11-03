"""
Day025: Trainable LayerNorm implemented explicitly from the mean/variance formula:
  mu = mean(x, dim=-1, keepdim=True)
  var = mean((x - mu)^2, dim=-1, keepdim=True)
  x_hat = (x - mu) / sqrt(var + eps)
  y = weight * x_hat + bias
"""

from __future__ import annotations

import torch
from torch import nn


class ScratchLayerNorm(nn.Module):
    """Explicit trainable LayerNorm over the channel dimension d_model."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"ScratchLayerNorm expected last dimension {self.d_model}, got {x.shape[-1]}"
            )
        mean = x.mean(dim=-1, keepdim=True)
        diff = x - mean
        var = (diff * diff).mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        normalized = diff * inv_std
        return normalized * self.weight + self.bias
