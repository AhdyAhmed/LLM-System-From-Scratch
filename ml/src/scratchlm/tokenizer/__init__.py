"""From-scratch byte-level and BPE tokenizers, special tokens, and incremental UTF-8 decoding."""

from scratchlm.tokenizer.byte_tokenizer import (
    ASSISTANT_TOKEN_ID,
    BOS_TOKEN_ID,
    BYTE_VOCAB_SIZE,
    CONTEXT_TOKEN_ID,
    END_TURN_TOKEN_ID,
    EOS_TOKEN_ID,
    IGNORE_INDEX,
    PAD_TOKEN_ID,
    SPECIAL_TOKENS,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    ByteTokenizer,
    IncrementalUtf8TokenStreamDecoder,
    ShiftedBatch,
    build_shifted_next_token_batch,
)

__all__ = [
    "ASSISTANT_TOKEN_ID",
    "BOS_TOKEN_ID",
    "BYTE_VOCAB_SIZE",
    "CONTEXT_TOKEN_ID",
    "END_TURN_TOKEN_ID",
    "EOS_TOKEN_ID",
    "IGNORE_INDEX",
    "PAD_TOKEN_ID",
    "SPECIAL_TOKENS",
    "SYSTEM_TOKEN_ID",
    "USER_TOKEN_ID",
    "ByteTokenizer",
    "IncrementalUtf8TokenStreamDecoder",
    "ShiftedBatch",
    "build_shifted_next_token_batch",
]
