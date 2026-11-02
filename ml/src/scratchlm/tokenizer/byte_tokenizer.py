"""
Day022: UTF-8 byte-level tokenizer with explicit special-token IDs (256..263),
stateful incremental UTF-8 streaming decoder, and next-token sequence shifting.
Distinguishes padding (PAD_TOKEN_ID = 256) from raw 0x00 byte values (0).
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from typing import Sequence

import torch

PAD_TOKEN_ID = 256
EOS_TOKEN_ID = 257
BOS_TOKEN_ID = 258
SYSTEM_TOKEN_ID = 259
USER_TOKEN_ID = 260
ASSISTANT_TOKEN_ID = 261
CONTEXT_TOKEN_ID = 262
END_TURN_TOKEN_ID = 263

BYTE_VOCAB_SIZE = 264
IGNORE_INDEX = -100

SPECIAL_TOKENS: dict[str, int] = {
    "<|pad|>": PAD_TOKEN_ID,
    "<|eos|>": EOS_TOKEN_ID,
    "<|bos|>": BOS_TOKEN_ID,
    "<|system|>": SYSTEM_TOKEN_ID,
    "<|user|>": USER_TOKEN_ID,
    "<|assistant|>": ASSISTANT_TOKEN_ID,
    "<|context|>": CONTEXT_TOKEN_ID,
    "<|end_turn|>": END_TURN_TOKEN_ID,
}

ROLE_TO_SPECIAL_ID: dict[str, int] = {
    "system": SYSTEM_TOKEN_ID,
    "user": USER_TOKEN_ID,
    "assistant": ASSISTANT_TOKEN_ID,
    "context": CONTEXT_TOKEN_ID,
}


class IncrementalUtf8TokenStreamDecoder:
    """
    Stateful per-stream UTF-8 decoder that buffers incomplete multi-byte UTF-8 sequences
    across token boundaries (e.g., 2-byte Arabic or 4-byte emoji split across tokens)
    and emits only valid UTF-8 characters.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def step(self, token_id: int) -> str:
        """Feed one generated token ID and return any newly completed UTF-8 characters."""
        if token_id < 0 or token_id >= BYTE_VOCAB_SIZE:
            raise ValueError(f"Token ID {token_id} out of range [0, {BYTE_VOCAB_SIZE})")
        if token_id >= 256:
            # Special control tokens do not emit visible text characters
            return ""
        return self._decoder.decode(bytes([token_id]), final=False)

    def flush(self) -> str:
        """Flush any trailing incomplete byte sequence on stream termination."""
        return self._decoder.decode(b"", final=True)

    def reset(self) -> None:
        self._decoder.reset()


class ByteTokenizer:
    """
    Deterministic byte-level tokenizer mapping raw UTF-8 bytes to 0..255 and
    structural control tokens to 256..263 (vocab_size = 264).
    """

    vocab_size: int = BYTE_VOCAB_SIZE
    pad_token_id: int = PAD_TOKEN_ID
    eos_token_id: int = EOS_TOKEN_ID
    bos_token_id: int = BOS_TOKEN_ID

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """
        Encode arbitrary text as raw UTF-8 byte IDs (0..255).
        Note: literal special-token strings inside user text are intentionally encoded as raw bytes
        so untrusted user prompts cannot inject privileged role markers.
        """
        ids: list[int] = []
        if add_bos:
            ids.append(BOS_TOKEN_ID)
        ids.extend(text.encode("utf-8"))
        if add_eos:
            ids.append(EOS_TOKEN_ID)
        return ids

    def encode_chat_turns(
        self,
        turns: Sequence[tuple[str, str]],
        *,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """
        Structurally encode (role, content) turns using dedicated role token IDs (259..263)
        while encoding turn content strictly as raw UTF-8 bytes (0..255).
        """
        ids: list[int] = [BOS_TOKEN_ID]
        for role, content in turns:
            role_key = role.strip().lower()
            if role_key not in ROLE_TO_SPECIAL_ID:
                raise ValueError(f"Unsupported chat role: {role}")
            ids.append(ROLE_TO_SPECIAL_ID[role_key])
            ids.extend(content.encode("utf-8"))
            ids.append(END_TURN_TOKEN_ID)
        if add_generation_prompt:
            ids.append(ASSISTANT_TOKEN_ID)
        return ids

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """Decode a sequence of token IDs back into a UTF-8 string."""
        raw_bytes = bytearray()
        parts: list[str] = []
        id_to_special = {v: k for k, v in SPECIAL_TOKENS.items()}

        for tid in token_ids:
            if tid < 0 or tid >= BYTE_VOCAB_SIZE:
                raise ValueError(f"Token ID {tid} out of range [0, {BYTE_VOCAB_SIZE})")
            if tid < 256:
                raw_bytes.append(tid)
            elif not skip_special_tokens:
                if raw_bytes:
                    parts.append(raw_bytes.decode("utf-8", errors="replace"))
                    raw_bytes.clear()
                parts.append(id_to_special[tid])

        if raw_bytes:
            parts.append(raw_bytes.decode("utf-8", errors="replace"))
        return "".join(parts)

    def create_stream_decoder(self) -> IncrementalUtf8TokenStreamDecoder:
        return IncrementalUtf8TokenStreamDecoder()


@dataclass(frozen=True)
class ShiftedBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    loss_mask: torch.Tensor
    valid_target_count: int


def build_shifted_next_token_batch(
    token_sequences: Sequence[Sequence[int]],
    max_seq_len: int,
    pad_token_id: int = PAD_TOKEN_ID,
    ignore_index: int = IGNORE_INDEX,
) -> ShiftedBatch:
    """
    Construct next-token prediction inputs (tokens[:-1]) and targets (tokens[1:])
    padded with pad_token_id (256, distinct from byte 0x00 = 0).
    Positions where target == pad_token_id receive ignore_index (-100) and loss_mask = False.
    """
    if not token_sequences:
        raise ValueError("token_sequences must not be empty")
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be >= 1")

    batch_size = len(token_sequences)
    # Determine effective window length T <= max_seq_len
    max_raw_len = max(len(seq) for seq in token_sequences)
    if max_raw_len < 2:
        raise ValueError("Each sequence batch must contain at least 2 tokens for next-token shifting")
    seq_len = min(max_raw_len - 1, max_seq_len)

    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    target_ids = torch.full((batch_size, seq_len), ignore_index, dtype=torch.long)
    loss_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    for b_idx, seq in enumerate(token_sequences):
        if len(seq) < 2:
            raise ValueError(f"Sequence at index {b_idx} has length < 2")
        window = list(seq[: seq_len + 1])
        inp = window[:-1]
        tgt = window[1:]
        for pos, (i_tok, t_tok) in enumerate(zip(inp, tgt)):
            input_ids[b_idx, pos] = int(i_tok)
            if int(t_tok) != pad_token_id and int(i_tok) != pad_token_id:
                target_ids[b_idx, pos] = int(t_tok)
                loss_mask[b_idx, pos] = True

    valid_count = int(loss_mask.sum().item())
    return ShiftedBatch(
        input_ids=input_ids,
        target_ids=target_ids,
        loss_mask=loss_mask,
        valid_target_count=valid_count,
    )
