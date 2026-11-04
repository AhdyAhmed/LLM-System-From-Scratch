"""
Day027: Full decoder-only ScratchGPT model, final ScratchLayerNorm, tied input/output embeddings,
N(0, 0.02^2) weight initialization with 1 / sqrt(2 * n_layers) residual-projection scaling,
and masked next-token cross-entropy loss.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from scratchlm.config import ModelConfig
from scratchlm.model.blocks import PreNormTransformerBlock, TokenAndPositionEmbedding
from scratchlm.model.normalization import ScratchLayerNorm
from scratchlm.tokenizer.byte_tokenizer import IGNORE_INDEX


def masked_next_token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    Compute next-token cross-entropy loss averaged strictly over valid supervised positions
    (targets != ignore_index) using explicit log-softmax and gather operations.
    Raises ValueError if there are zero valid target tokens.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D logits [B, T, V], got {tuple(logits.shape)}")
    if targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise ValueError(
            f"Target shape {tuple(targets.shape)} must match logits [B, T] {tuple(logits.shape[:2])}"
        )

    valid_mask = targets != ignore_index
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        raise ValueError("Cannot compute next-token cross-entropy with zero supervised target tokens")

    # Numerically stable log_softmax: logits - max - log(sum(exp(logits - max)))
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    log_probs = shifted - torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))

    safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
    token_nll = -log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    masked_nll = token_nll * valid_mask.to(token_nll.dtype)
    return masked_nll.sum() / float(valid_count)


class ScratchGPT(nn.Module):
    """
    From-scratch decoder-only GPT-style pre-norm Transformer language model.
    """

    def __init__(self, config: ModelConfig, *, init_seed: int = 1337) -> None:
        super().__init__()
        self.config = config

        self.embeddings = TokenAndPositionEmbedding(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList(
            [
                PreNormTransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    use_bias=config.use_bias,
                    layer_norm_eps=config.layer_norm_eps,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = ScratchLayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.token_embedding.weight

        self.reset_parameters(seed=init_seed)

    def reset_parameters(self, seed: int = 1337) -> None:
        """
        Initialize all model parameters deterministically:
          - Embeddings & Linear weights: N(0, 0.02^2)
          - Residual output projections (W_output in attention and W_down in MLP):
            N(0, (0.02 / sqrt(2 * n_layers))^2)
          - Linear biases: 0.0
          - LayerNorm weight = 1.0, bias = 0.0
        """
        gen = torch.Generator(device="cpu").manual_seed(seed)
        base_std = 0.02
        residual_std = base_std / math.sqrt(2.0 * float(self.config.n_layers))

        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Embedding):
                    module.weight.normal_(mean=0.0, std=base_std, generator=gen)
                elif isinstance(module, nn.Linear):
                    if module is self.lm_head and self.config.tie_word_embeddings:
                        # Tied weight is already initialized via token_embedding
                        continue
                    is_res_proj = bool(getattr(module, "_is_residual_out_proj", False))
                    std = residual_std if is_res_proj else base_std
                    module.weight.normal_(mean=0.0, std=std, generator=gen)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, ScratchLayerNorm):
                    module.weight.fill_(1.0)
                    module.bias.zero_()

    def exact_num_parameters(self) -> int:
        """Count unique trainable parameters (accounting for weight tying)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        ignore_index: int = IGNORE_INDEX,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Compute forward pass:
          input_ids: [B, T]
          targets: optional [B, T]
        Returns:
          (logits: [B, T, V], loss: scalar Tensor or None)
        """
        x = self.embeddings(input_ids, past_len=0)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = masked_next_token_cross_entropy(logits, targets, ignore_index=ignore_index)
        return logits, loss
