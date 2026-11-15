"""
Day058: Conversation prompt assembly, reserved output-token budget, explicit truncation/rejection
rules, and trained chat template formatting (`ContextPacker`, `ConversationTurn`, `PackedContextResult`).

Guarantees:
  1. Reserves `max_new_tokens` (1..64) from `context_limit` (128) so `prompt_token_budget = context_limit - max_new_tokens`.
  2. Formats conversations using the trained `ChatTemplateSpec` special token IDs:
       [SYSTEM_TOKEN_ID, *sys_ids, END_TURN_TOKEN_ID, USER_TOKEN_ID, *user_ids, END_TURN_TOKEN_ID, ASSISTANT_TOKEN_ID]
     and renders human-readable `<|system|> ... <|end_turn|>\n<|user|> ... <|end_turn|>\n<|assistant|>\n`.
  3. Never silently truncates or discards the newest user prompt or system instruction:
     if the newest user turn + system framing exceeds `prompt_token_budget`, raises `ContextWindowExceededError`.
  4. When historical conversation turns are supplied, packs newest turns first (preserving chronological
     order in the final formatted prompt) and deterministically drops oldest complete turns that do not fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scratchlm.tokenizer.bpe_tokenizer import BPETokenizer
from scratchlm.training.sft import ChatTemplateSpec

DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful storytelling and narrow-task assistant."


class ContextWindowExceededError(ValueError):
    """Raised when the required prompt (plus reserved output tokens) exceeds the model context window."""

    def __init__(
        self,
        message: str,
        *,
        input_tokens: int,
        reserved_output_tokens: int,
        context_limit: int,
        error_code: str = "CONTEXT_LIMIT_EXCEEDED",
    ) -> None:
        super().__init__(message)
        self.input_tokens = input_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.context_limit = context_limit
        self.error_code = error_code


@dataclass(frozen=True)
class ConversationTurn:
    """Single message turn in a conversation (`USER` or `ASSISTANT`)."""

    role: str  # "USER" | "ASSISTANT"
    content: str


@dataclass(frozen=True)
class PackedContextResult:
    """Deterministic output of packing a prompt + optional prior turns into a fixed context window."""

    formatted_prompt: str
    prompt_ids: list[int]
    input_tokens: int
    reserved_output_tokens: int
    available_prompt_budget: int
    context_limit: int
    included_history_turns: int
    dropped_older_turns: int


class ContextPacker:
    """
    Day058 prompt assembler and token-budget enforcer for `ScratchGPT` (`C = 128`).
    """

    def __init__(
        self,
        tokenizer: BPETokenizer,
        chat_template: ChatTemplateSpec | None = None,
        *,
        default_context_limit: int = 128,
        max_output_cap: int = 64,
        default_system_prompt: str = DEFAULT_SYSTEM_INSTRUCTION,
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_template = chat_template or ChatTemplateSpec()
        self.default_context_limit = default_context_limit
        self.max_output_cap = max_output_cap
        self.default_system_prompt = default_system_prompt

    def pack_prompt(
        self,
        user_prompt: str,
        *,
        max_new_tokens: int = 24,
        context_limit: int | None = None,
        history: Sequence[ConversationTurn] | None = None,
        system_prompt: str | None = None,
        apply_chat_template: bool = True,
    ) -> PackedContextResult:
        """
        Assemble and tokenize a prompt within `context_limit - max_new_tokens`.
        Raises `ContextWindowExceededError` if the newest `user_prompt` (with optional `system_prompt`
        and chat template markers) cannot fit inside the available prompt budget.
        """
        clean_prompt = user_prompt.strip()
        if not clean_prompt:
            raise ValueError("user_prompt must not be empty or whitespace-only")

        limit = int(context_limit if context_limit is not None else self.default_context_limit)
        if limit <= 0:
            raise ValueError(f"context_limit must be positive, got {limit}")
        if max_new_tokens < 1 or max_new_tokens > self.max_output_cap:
            raise ValueError(
                f"max_new_tokens ({max_new_tokens}) must be in [1, {self.max_output_cap}]"
            )
        if max_new_tokens >= limit:
            raise ContextWindowExceededError(
                f"max_new_tokens ({max_new_tokens}) leaves zero budget in context_limit ({limit})",
                input_tokens=0,
                reserved_output_tokens=max_new_tokens,
                context_limit=limit,
            )

        available_budget = limit - max_new_tokens

        if not apply_chat_template:
            raw_ids = self.tokenizer.encode(clean_prompt, add_bos=True, add_eos=False)
            if len(raw_ids) > available_budget:
                raise ContextWindowExceededError(
                    f"Prompt tokens ({len(raw_ids)}) + reserved max_new_tokens ({max_new_tokens}) "
                    f"exceeds context_limit ({limit})",
                    input_tokens=len(raw_ids),
                    reserved_output_tokens=max_new_tokens,
                    context_limit=limit,
                )
            return PackedContextResult(
                formatted_prompt=clean_prompt,
                prompt_ids=raw_ids,
                input_tokens=len(raw_ids),
                reserved_output_tokens=max_new_tokens,
                available_prompt_budget=available_budget,
                context_limit=limit,
                included_history_turns=0,
                dropped_older_turns=0,
            )

        sys_text = (
            system_prompt.strip()
            if system_prompt is not None
            else self.default_system_prompt
        )
        base_formatted, base_ids = self._render_and_encode_turns(
            system_text=sys_text,
            history_turns=[],
            latest_user_prompt=clean_prompt,
        )

        # If default system prompt pushes a medium user prompt over budget, fall back to compact user-only framing
        if len(base_ids) > available_budget and system_prompt is None:
            compact_formatted, compact_ids = self._render_and_encode_turns(
                system_text="",
                history_turns=[],
                latest_user_prompt=clean_prompt,
            )
            if len(compact_ids) <= available_budget:
                sys_text = ""
                base_formatted = compact_formatted
                base_ids = compact_ids

        if len(base_ids) > available_budget:
            raise ContextWindowExceededError(
                f"Newest prompt requires {len(base_ids)} tokens, exceeding available prompt budget "
                f"{available_budget} (context_limit={limit}, reserved_output_tokens={max_new_tokens})",
                input_tokens=len(base_ids),
                reserved_output_tokens=max_new_tokens,
                context_limit=limit,
            )

        if not history:
            return PackedContextResult(
                formatted_prompt=base_formatted,
                prompt_ids=base_ids,
                input_tokens=len(base_ids),
                reserved_output_tokens=max_new_tokens,
                available_prompt_budget=available_budget,
                context_limit=limit,
                included_history_turns=0,
                dropped_older_turns=0,
            )

        # Greedily include as many complete recent turns (from newest to oldest) as fit in available_budget
        accepted_newest_first: list[ConversationTurn] = []
        best_formatted = base_formatted
        best_ids = base_ids
        dropped_count = 0

        for idx in range(len(history) - 1, -1, -1):
            candidate_turns = [history[idx], *reversed(accepted_newest_first)]
            candidate_formatted, candidate_ids = self._render_and_encode_turns(
                system_text=sys_text,
                history_turns=candidate_turns,
                latest_user_prompt=clean_prompt,
            )
            if len(candidate_ids) <= available_budget:
                accepted_newest_first.append(history[idx])
                best_formatted = candidate_formatted
                best_ids = candidate_ids
            else:
                dropped_count = idx + 1
                break

        return PackedContextResult(
            formatted_prompt=best_formatted,
            prompt_ids=best_ids,
            input_tokens=len(best_ids),
            reserved_output_tokens=max_new_tokens,
            available_prompt_budget=available_budget,
            context_limit=limit,
            included_history_turns=len(accepted_newest_first),
            dropped_older_turns=dropped_count,
        )

    def _render_and_encode_turns(
        self,
        *,
        system_text: str,
        history_turns: Sequence[ConversationTurn],
        latest_user_prompt: str,
    ) -> tuple[str, list[int]]:
        text_parts: list[str] = []
        ids: list[int] = []
        tmpl = self.chat_template

        if system_text:
            text_parts.append(f"<|system|>\n{system_text}\n<|end_turn|>\n")
            ids.append(tmpl.system_token_id)
            ids.extend(self.tokenizer.encode(system_text, add_bos=False, add_eos=False))
            ids.append(tmpl.end_turn_token_id)

        for turn in history_turns:
            role_upper = turn.role.strip().upper()
            clean_turn = turn.content.strip()
            if role_upper == "USER":
                text_parts.append(f"<|user|>\n{clean_turn}\n<|end_turn|>\n")
                ids.append(tmpl.user_token_id)
            else:
                text_parts.append(f"<|assistant|>\n{clean_turn}\n<|end_turn|>\n")
                ids.append(tmpl.assistant_token_id)
            ids.extend(self.tokenizer.encode(clean_turn, add_bos=False, add_eos=False))
            ids.append(tmpl.end_turn_token_id)

        text_parts.append(f"<|user|>\n{latest_user_prompt}\n<|end_turn|>\n<|assistant|>\n")
        ids.append(tmpl.user_token_id)
        ids.extend(self.tokenizer.encode(latest_user_prompt, add_bos=False, add_eos=False))
        ids.append(tmpl.end_turn_token_id)
        ids.append(tmpl.assistant_token_id)

        return "".join(text_parts), ids
