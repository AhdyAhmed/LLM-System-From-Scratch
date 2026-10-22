"""Model configuration schema and parameter count calculator for the scratchlm Transformer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Immutable decoder-only GPT-2 style pre-norm Transformer configuration."""

    profile_name: str
    vocab_size: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    dropout: float
    use_bias: bool
    tie_word_embeddings: bool
    layer_norm_eps: float

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.d_ff != 4 * self.d_model:
            raise ValueError("d_ff must equal 4 * d_model for standard GPT-2 MLP")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0)")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def exact_parameter_count(self) -> int:
        """
        Compute the exact trainable parameter count under weight tying (W_lm_head = W_tok_emb)
        and use_bias=True (or use_bias=False).

        For tie_word_embeddings=True and use_bias=True with d_ff = 4 * d_model:
          N_params = V*d + C*d + L*(12*d^2 + 13*d) + 2*d
        """
        v = self.vocab_size
        c = self.max_seq_len
        l = self.n_layers
        d = self.d_model
        ff = self.d_ff

        embeddings = v * d + c * d
        attn_weights = 4 * d * d + (4 * d if self.use_bias else 0)
        mlp_weights = 2 * d * ff + ((ff + d) if self.use_bias else 0)
        ln_weights = 4 * d
        per_layer = attn_weights + mlp_weights + ln_weights

        final_ln = 2 * d
        lm_head = 0 if self.tie_word_embeddings else v * d

        return embeddings + l * per_layer + final_ln + lm_head


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate a ModelConfig from a JSON file."""
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return ModelConfig(
        profile_name=str(raw["profile_name"]),
        vocab_size=int(raw["vocab_size"]),
        max_seq_len=int(raw["max_seq_len"]),
        n_layers=int(raw["n_layers"]),
        n_heads=int(raw["n_heads"]),
        d_model=int(raw["d_model"]),
        d_ff=int(raw["d_ff"]),
        dropout=float(raw["dropout"]),
        use_bias=bool(raw["use_bias"]),
        tie_word_embeddings=bool(raw["tie_word_embeddings"]),
        layer_norm_eps=float(raw["layer_norm_eps"]),
    )
