"""Autoregressive decoding, sampling strategies, and incremental UTF-8 streaming."""

from scratchlm.inference.sampling import (
    GenerationSampleResult,
    generate_tokens,
    sample_next_token,
    stream_generate_utf8,
)

__all__ = [
    "GenerationSampleResult",
    "generate_tokens",
    "sample_next_token",
    "stream_generate_utf8",
]
