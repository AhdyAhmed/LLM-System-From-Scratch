"""scratchlm: Custom decoder-only Transformer LM, tokenizer, serving, and retrieval engine."""

__version__ = "0.1.0"

from scratchlm.config import ModelConfig, load_model_config

__all__ = ["ModelConfig", "load_model_config", "__version__"]
