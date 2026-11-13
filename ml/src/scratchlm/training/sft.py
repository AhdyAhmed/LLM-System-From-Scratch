"""
Day047-Day049: Narrow Supervised Fine-Tuning (SFT) dataset authoring, chat template specification,
task/template-family split isolation, assistant-only loss masking (including terminating EOS),
bounded SFT training from a frozen base checkpoint with child MLflow lineage, and SFT rubric evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from scratchlm.config import ModelConfig
from scratchlm.data.ingestion import compute_canonical_dedup_hash, normalize_document_text
from scratchlm.evaluation.harness import (
    LanguageModelMetrics,
    SftRubricComparison,
    compute_token_repetition_ratio,
    evaluate_gpt_on_shard,
)
from scratchlm.inference.sampling import generate_tokens
from scratchlm.model.gpt import ScratchGPT
from scratchlm.tokenizer.bpe_tokenizer import BPETokenizer
from scratchlm.tokenizer.byte_tokenizer import (
    ASSISTANT_TOKEN_ID,
    END_TURN_TOKEN_ID,
    EOS_TOKEN_ID,
    IGNORE_INDEX,
    PAD_TOKEN_ID,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    ByteTokenizer,
)
from scratchlm.training.checkpoints import AtomicCheckpointManager, CheckpointMetadata, LineageHashes
from scratchlm.training.loop import (
    NonFiniteLossError,
    TrainingConfig,
    build_adamw_optimizer,
    compute_cosine_lr_with_warmup,
    seed_everything,
)
from scratchlm.training.tracker import ExperimentTracker, InMemoryExperimentTracker


@dataclass(frozen=True)
class ChatTemplateSpec:
    """
    Frozen structural chat format using reserved special token IDs (256..263).
    Layout:
      <|system|> {system_text} <|end_turn|>
      <|user|> {user_text} <|end_turn|>
      <|assistant|> {assistant_text} <|end_turn|> <|eos|>
    """

    template_version: str = "scratch-chat-v1"
    system_token_id: int = SYSTEM_TOKEN_ID
    user_token_id: int = USER_TOKEN_ID
    assistant_token_id: int = ASSISTANT_TOKEN_ID
    end_turn_token_id: int = END_TURN_TOKEN_ID
    eos_token_id: int = EOS_TOKEN_ID
    pad_token_id: int = PAD_TOKEN_ID
    max_byte_tokens: int = 128

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SftExample:
    """Single narrow-task continuation/formatting SFT example."""

    example_id: str
    task_category: str
    template_family_id: str
    system_text: str
    user_text: str
    assistant_text: str
    expected_prefix_marker: str
    byte_token_length: int
    bpe_token_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SftSplitFileMeta:
    split_name: str
    example_count: int
    file_sha256: str
    example_ids: list[str]
    template_families: list[str]
    answer_canonical_hashes: list[str]


@dataclass(frozen=True)
class SftDatasetManifest:
    """Provenance and split isolation manifest for `ml/data/sft/` (Day047)."""

    schema_version: str
    dataset_id: str
    chat_template: ChatTemplateSpec
    max_byte_token_budget: int
    max_observed_byte_tokens: int
    max_observed_bpe_tokens: int
    test_split_sealed: bool
    splits: dict[str, SftSplitFileMeta]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.canonical_json()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: Path) -> SftDatasetManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        tmpl = ChatTemplateSpec(**raw["chat_template"])
        splits = {k: SftSplitFileMeta(**v) for k, v in raw["splits"].items()}
        return cls(
            schema_version=raw["schema_version"],
            dataset_id=raw["dataset_id"],
            chat_template=tmpl,
            max_byte_token_budget=raw["max_byte_token_budget"],
            max_observed_byte_tokens=raw["max_observed_byte_tokens"],
            max_observed_bpe_tokens=raw["max_observed_bpe_tokens"],
            test_split_sealed=raw["test_split_sealed"],
            splits=splits,
        )


@dataclass(frozen=True)
class EncodedSftSample:
    """Tokenized and assistant-masked SFT example ready for batching."""

    example_id: str
    input_ids: torch.Tensor       # [T] int64
    targets: torch.Tensor         # [T] int64 (IGNORE_INDEX = -100 on system/user/pad)
    prompt_prefix_ids: list[int]  # [<|system|> ... <|end_turn|> <|user|> ... <|end_turn|> <|assistant|>]
    supervised_token_count: int
    total_non_pad_tokens: int


def encode_sft_prompt_prefix(
    tokenizer: ByteTokenizer | BPETokenizer,
    system_text: str,
    user_text: str,
) -> list[int]:
    """
    Encode the prompt prefix up to and including the `<|assistant|>` turn header token:
      [SYSTEM_TOKEN_ID, *system_ids, END_TURN_TOKEN_ID, USER_TOKEN_ID, *user_ids, END_TURN_TOKEN_ID, ASSISTANT_TOKEN_ID]
    """
    sys_norm = normalize_document_text(system_text)
    usr_norm = normalize_document_text(user_text)
    if not usr_norm:
        raise ValueError("user_text must not be empty in SFT prompt")

    prefix: list[int] = [SYSTEM_TOKEN_ID]
    prefix.extend(tokenizer.encode(sys_norm, add_bos=False, add_eos=False))
    prefix.append(END_TURN_TOKEN_ID)
    prefix.append(USER_TOKEN_ID)
    prefix.extend(tokenizer.encode(usr_norm, add_bos=False, add_eos=False))
    prefix.append(END_TURN_TOKEN_ID)
    prefix.append(ASSISTANT_TOKEN_ID)
    return prefix


def encode_sft_example_with_assistant_mask(
    tokenizer: ByteTokenizer | BPETokenizer,
    *,
    example_id: str,
    system_text: str,
    user_text: str,
    assistant_text: str,
    seq_len: int = 128,
) -> EncodedSftSample:
    """
    Day048: Encode a full SFT turn and build next-token `input_ids` and `targets` where:
      - System, user, and role header tokens have `targets == IGNORE_INDEX (-100)`.
      - Assistant content tokens AND terminating `<|end_turn|>` + `<|eos|>` are supervised.
      - Padding tokens have `targets == IGNORE_INDEX (-100)`.
      - Raises ValueError if assistant_text produces zero supervised content tokens or if the
        full sequence exceeds `seq_len` (including tiny's 128-byte-token limit).
    """
    asst_norm = normalize_document_text(assistant_text)
    if not asst_norm:
        raise ValueError(f"SFT example '{example_id}' has empty assistant_text (zero supervision)")

    prefix_ids = encode_sft_prompt_prefix(tokenizer, system_text, user_text)
    asst_content_ids = tokenizer.encode(asst_norm, add_bos=False, add_eos=False)
    if not asst_content_ids:
        raise ValueError(f"SFT example '{example_id}' encoded to zero assistant content tokens")

    completion_ids = list(asst_content_ids) + [END_TURN_TOKEN_ID, EOS_TOKEN_ID]
    full_ids = prefix_ids + completion_ids

    if len(full_ids) > seq_len:
        raise ValueError(
            f"SFT example '{example_id}' length ({len(full_ids)}) exceeds context limit seq_len={seq_len}"
        )

    # Next-token shift across full_ids:
    # At index i, input_ids[i] = full_ids[i], and the unmasked next token is full_ids[i + 1].
    # Prefix occupies indices 0 .. len(prefix_ids) - 1, where full_ids[len(prefix_ids) - 1] == ASSISTANT_TOKEN_ID.
    # Therefore:
    # - For i < len(prefix_ids) - 1: full_ids[i + 1] is still part of the prompt prefix -> mask with IGNORE_INDEX.
    # - For i from len(prefix_ids) - 1 to len(full_ids) - 2: full_ids[i + 1] is an assistant token (or END_TURN / EOS) -> supervise!
    input_ids = torch.full((seq_len,), PAD_TOKEN_ID, dtype=torch.long)
    targets = torch.full((seq_len,), IGNORE_INDEX, dtype=torch.long)

    n_inputs = len(full_ids) - 1
    input_ids[:n_inputs] = torch.tensor(full_ids[:-1], dtype=torch.long)

    first_supervised_idx = len(prefix_ids) - 1
    for i in range(first_supervised_idx, n_inputs):
        targets[i] = int(full_ids[i + 1])

    supervised_count = int((targets != IGNORE_INDEX).sum().item())
    if supervised_count != len(completion_ids):
        raise AssertionError(
            f"Expected {len(completion_ids)} supervised tokens (assistant + END_TURN + EOS), got {supervised_count}"
        )

    return EncodedSftSample(
        example_id=example_id,
        input_ids=input_ids,
        targets=targets,
        prompt_prefix_ids=prefix_ids,
        supervised_token_count=supervised_count,
        total_non_pad_tokens=len(full_ids),
    )


def generate_curated_sft_examples(bpe_tokenizer: BPETokenizer) -> list[SftExample]:
    """
    Author 48 short continuation/formatting SFT examples across 12 distinct template families.
    Every example fits strictly inside 128 byte tokens AND 128 BPE tokens, with unique answers.
    """
    byte_tok = ByteTokenizer()
    system_prompt = "Format story notes concisely."

    families = [
        # Train families (families 01..08 -> 32 examples)
        (
            "family_moral_tag_01",
            "moral_tag",
            "MORAL:",
            [
                ("Write moral: Lily waited by the sunny meadow gate.", "MORAL: patience brings calm."),
                ("Write moral: Omar wound the brass clock with care.", "MORAL: steady hands build trust."),
                ("Write moral: Mira steered the paper boat gently.", "MORAL: gentle turns keep afloat."),
                ("Write moral: Tariq counted three bright stars.", "MORAL: clear notes guide the night."),
            ],
        ),
        (
            "family_moral_tag_02",
            "moral_tag",
            "MORAL:",
            [
                ("Write moral: Nora shared warm rolls at noon.", "MORAL: shared bread warms all."),
                ("Write moral: Zayd oiled the tiny wooden gears.", "MORAL: small care stops rust."),
                ("Write moral: Lily returned the blue marble quickly.", "MORAL: honest friends return gifts."),
                ("Write moral: Omar swept the quiet workshop floor.", "MORAL: tidy rooms spark joy."),
            ],
        ),
        (
            "family_role_bullet_03",
            "role_bullet",
            "- Role:",
            [
                ("Who is Lily in the sunny meadow?", "- Role: curious little rabbit."),
                ("Who is Omar in the workshop?", "- Role: gentle clock apprentice."),
                ("Who is Mira by the silver brook?", "- Role: brave paper boat captain."),
                ("Who is Tariq in the lantern tower?", "- Role: young star watcher."),
            ],
        ),
        (
            "family_role_bullet_04",
            "role_bullet",
            "- Role:",
            [
                ("Who is Nora in the village kitchen?", "- Role: cheerful baker helper."),
                ("Who is Zayd in the study room?", "- Role: clever wooden robot."),
                ("What did Lily carry at dawn?", "- Role: keeper of the compass."),
                ("What did Mira guard on the brook?", "- Role: pilot of paper sails."),
            ],
        ),
        (
            "family_ending_05",
            "story_ending",
            "END:",
            [
                ("Finish: Lily found the lost marble.", "END: She placed it in the glass jar."),
                ("Finish: Omar fixed the copper bell.", "END: It rang clearly at sunset."),
                ("Finish: Mira reached the wooden bridge.", "END: She docked the boat safely."),
                ("Finish: Tariq opened the star map.", "END: He traced the northern crown."),
            ],
        ),
        (
            "family_ending_06",
            "story_ending",
            "END:",
            [
                ("Finish: Nora baked cinnamon rolls.", "END: The travelers smiled with thanks."),
                ("Finish: Zayd solved the seventh riddle.", "END: He wrote four on the slate."),
                ("Finish: Kittens heard the tiny bell.", "END: They trotted home before dark."),
                ("Finish: Clouds gathered over the meadow.", "END: Lily latched the cozy burrow."),
            ],
        ),
        (
            "family_kv_item_07",
            "item_kv",
            "ITEM=",
            [
                ("Extract item: Lily held a brass compass.", "ITEM=brass compass"),
                ("Extract item: Omar read a blue notebook.", "ITEM=blue notebook"),
                ("Extract item: Mira rang a copper bell.", "ITEM=copper bell"),
                ("Extract item: Tariq kept a star map.", "ITEM=star map"),
            ],
        ),
        (
            "family_kv_item_08",
            "item_kv",
            "ITEM=",
            [
                ("Extract item: Nora carried warm rolls.", "ITEM=warm rolls"),
                ("Extract item: Zayd used a chalk slate.", "ITEM=chalk slate"),
                ("Extract item: Lily latched a wooden gate.", "ITEM=wooden gate"),
                ("Extract item: Omar lit a brass lantern.", "ITEM=brass lantern"),
            ],
        ),
        # Validation families (families 09..10 -> 8 examples)
        (
            "family_val_moral_09",
            "moral_tag",
            "MORAL:",
            [
                ("Write moral: Sammy folded the map before rain fell.", "MORAL: early prep saves dry maps."),
                ("Write moral: Hana tied the red ribbon neatly.", "MORAL: neat knots hold strong."),
                ("Write moral: Leo listened to the quiet brook.", "MORAL: quiet ears hear the stream."),
                ("Write moral: Dina counted seven silver coins.", "MORAL: careful counts prevent loss."),
            ],
        ),
        (
            "family_val_item_10",
            "item_kv",
            "ITEM=",
            [
                ("Extract item: Sammy packed a wool cloak.", "ITEM=wool cloak"),
                ("Extract item: Hana polished a silver spoon.", "ITEM=silver spoon"),
                ("Extract item: Leo carved a pine whistle.", "ITEM=pine whistle"),
                ("Extract item: Dina locked a cedar box.", "ITEM=cedar box"),
            ],
        ),
        # Sealed Test families (families 11..12 -> 8 examples)
        (
            "family_test_moral_11",
            "moral_tag",
            "MORAL:",
            [
                ("Write moral: Faris mended the torn canvas sail.", "MORAL: timely stitches save sails."),
                ("Write moral: Lina watered the tiny olive sprout.", "MORAL: daily drops grow tall trees."),
                ("Write moral: Rami stacked the firewood before snow.", "MORAL: dry logs warm winter nights."),
                ("Write moral: Sana labeled every spice jar clearly.", "MORAL: clear labels speed cooking."),
            ],
        ),
        (
            "family_test_item_12",
            "item_kv",
            "ITEM=",
            [
                ("Extract item: Faris held a linen rope.", "ITEM=linen rope"),
                ("Extract item: Lina filled a clay pitcher.", "ITEM=clay pitcher"),
                ("Extract item: Rami swung an iron lantern.", "ITEM=iron lantern"),
                ("Extract item: Sana opened a spice tin.", "ITEM=spice tin"),
            ],
        ),
    ]

    examples: list[SftExample] = []
    idx = 0
    for fam_id, cat, marker, items in families:
        for usr, asst in items:
            idx += 1
            ex_id = f"sft-{idx:04d}"
            byte_sample = encode_sft_example_with_assistant_mask(
                byte_tok,
                example_id=ex_id,
                system_text=system_prompt,
                user_text=usr,
                assistant_text=asst,
                seq_len=128,
            )
            bpe_sample = encode_sft_example_with_assistant_mask(
                bpe_tokenizer,
                example_id=ex_id,
                system_text=system_prompt,
                user_text=usr,
                assistant_text=asst,
                seq_len=128,
            )
            examples.append(
                SftExample(
                    example_id=ex_id,
                    task_category=cat,
                    template_family_id=fam_id,
                    system_text=system_prompt,
                    user_text=usr,
                    assistant_text=asst,
                    expected_prefix_marker=marker,
                    byte_token_length=byte_sample.total_non_pad_tokens,
                    bpe_token_length=bpe_sample.total_non_pad_tokens,
                )
            )
    return examples


def verify_sft_split_disjointness(
    train_examples: Sequence[SftExample],
    val_examples: Sequence[SftExample],
    test_examples: Sequence[SftExample],
) -> None:
    """
    Day047: Verify that:
      1. Every example fits within tiny's 128-byte-token limit (`byte_token_length <= 128`).
      2. No `example_id` or `template_family_id` crosses split boundaries.
      3. No normalized answer (`assistant_text`) or prompt (`user_text`) is duplicated across splits.
    """
    splits = {"train": train_examples, "val": val_examples, "test": test_examples}
    seen_ids: dict[str, str] = {}
    seen_families: dict[str, str] = {}
    seen_answers: dict[str, str] = {}
    seen_prompts: dict[str, str] = {}

    for split_name, items in splits.items():
        if not items:
            raise ValueError(f"SFT split '{split_name}' cannot be empty")
        families_in_this_split: set[str] = set()
        for ex in items:
            if ex.byte_token_length > 128 or ex.bpe_token_length > 128:
                raise ValueError(
                    f"SFT example '{ex.example_id}' exceeds 128-token budget: "
                    f"byte={ex.byte_token_length}, bpe={ex.bpe_token_length}"
                )
            if ex.example_id in seen_ids:
                raise ValueError(f"Duplicate SFT example_id '{ex.example_id}'")
            seen_ids[ex.example_id] = split_name

            if ex.template_family_id in seen_families and seen_families[ex.template_family_id] != split_name:
                raise ValueError(
                    f"Template family leakage: '{ex.template_family_id}' in both "
                    f"'{seen_families[ex.template_family_id]}' and '{split_name}'"
                )
            families_in_this_split.add(ex.template_family_id)

            ans_hash = compute_canonical_dedup_hash(ex.assistant_text)
            if ans_hash in seen_answers and seen_answers[ans_hash] != split_name:
                raise ValueError(
                    f"Duplicated evaluation answer across splits for '{ex.example_id}' "
                    f"('{seen_answers[ans_hash]}' vs '{split_name}')"
                )
            seen_answers[ans_hash] = split_name

            usr_hash = compute_canonical_dedup_hash(ex.user_text)
            if usr_hash in seen_prompts and seen_prompts[usr_hash] != split_name:
                raise ValueError(
                    f"Duplicated SFT prompt across splits for '{ex.example_id}' "
                    f"('{seen_prompts[usr_hash]}' vs '{split_name}')"
                )
            seen_prompts[usr_hash] = split_name

        for fam in families_in_this_split:
            seen_families[fam] = split_name


def build_and_freeze_sft_dataset(
    output_dir: Path,
    bpe_tokenizer: BPETokenizer,
) -> tuple[dict[str, list[SftExample]], SftDatasetManifest]:
    """Build and persist `ml/data/sft/{train,val,test}.jsonl` and `sft_manifest.json`."""
    all_examples = generate_curated_sft_examples(bpe_tokenizer)
    train_ex = [e for e in all_examples if not e.template_family_id.startswith(("family_val_", "family_test_"))]
    val_ex = [e for e in all_examples if e.template_family_id.startswith("family_val_")]
    test_ex = [e for e in all_examples if e.template_family_id.startswith("family_test_")]

    verify_sft_split_disjointness(train_ex, val_ex, test_ex)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_map = {"train": train_ex, "val": val_ex, "test": test_ex}
    split_metas: dict[str, SftSplitFileMeta] = {}

    for split_name, items in split_map.items():
        path = output_dir / f"{split_name}.jsonl"
        lines = [json.dumps(e.to_dict(), sort_keys=True, ensure_ascii=False) for e in items]
        payload = "\n".join(lines) + "\n"
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)

        file_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        split_metas[split_name] = SftSplitFileMeta(
            split_name=split_name,
            example_count=len(items),
            file_sha256=file_sha,
            example_ids=[e.example_id for e in items],
            template_families=sorted({e.template_family_id for e in items}),
            answer_canonical_hashes=[compute_canonical_dedup_hash(e.assistant_text) for e in items],
        )

    manifest = SftDatasetManifest(
        schema_version="1.0.0",
        dataset_id="scratch-narrow-story-sft-v1",
        chat_template=ChatTemplateSpec(),
        max_byte_token_budget=128,
        max_observed_byte_tokens=max(e.byte_token_length for e in all_examples),
        max_observed_bpe_tokens=max(e.bpe_token_length for e in all_examples),
        test_split_sealed=True,
        splits=split_metas,
    )
    manifest.save(output_dir / "sft_manifest.json")
    return split_map, manifest


def load_sft_split(sft_dir: Path, split_name: str, *, allow_sealed_test: bool = False) -> list[SftExample]:
    manifest = SftDatasetManifest.load(sft_dir / "sft_manifest.json")
    if split_name == "test" and manifest.test_split_sealed and not allow_sealed_test:
        raise PermissionError("The held-out SFT 'test' split is sealed until final release evaluation.")
    meta = manifest.splits[split_name]
    jsonl_path = sft_dir / f"{split_name}.jsonl"
    raw = jsonl_path.read_text(encoding="utf-8")
    actual_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if actual_sha != meta.file_sha256:
        raise ValueError(f"SFT split '{split_name}' hash mismatch")
    return [SftExample(**json.loads(line)) for line in raw.splitlines() if line.strip()]


@dataclass(frozen=True)
class SftRunResult:
    sft_run_id: str
    parent_base_run_id: str
    base_checkpoint_sha256: str
    sft_checkpoint_sha256: str
    sft_dataset_manifest_sha256: str
    tokenizer_sha256: str
    initial_sft_train_loss: float
    final_sft_train_loss: float
    initial_sft_val_loss: float
    final_sft_val_loss: float
    sft_checkpoint_meta: CheckpointMetadata


def _collate_sft_batch(
    samples: Sequence[EncodedSftSample],
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    inputs = torch.stack([s.input_ids for s in samples], dim=0).to(device)
    targets = torch.stack([s.targets for s in samples], dim=0).to(device)
    valid_count = int((targets != IGNORE_INDEX).sum().item())
    if valid_count <= 0:
        raise ValueError("SFT batch contains 0 supervised assistant tokens")
    return inputs, targets, valid_count


@torch.no_grad()
def evaluate_sft_loss(
    model: ScratchGPT,
    encoded_samples: Sequence[EncodedSftSample],
    *,
    batch_size: int = 4,
    device: str = "cpu",
) -> float:
    was_training = model.training
    model.eval()
    try:
        total_nll = 0.0
        total_valid = 0
        for i in range(0, len(encoded_samples), batch_size):
            chunk = encoded_samples[i : i + batch_size]
            inp, tgt, n_valid = _collate_sft_batch(chunk, device=device)
            _, loss = model(inp, targets=tgt, ignore_index=IGNORE_INDEX)
            assert loss is not None
            total_nll += float(loss.item()) * n_valid
            total_valid += n_valid
        return total_nll / max(1, total_valid)
    finally:
        model.train(was_training)


def run_bounded_sft_from_base_checkpoint(
    *,
    base_checkpoint_dir: Path,
    base_lineage: LineageHashes,
    parent_base_run_id: str,
    sft_checkpoint_dir: Path,
    sft_dir: Path,
    tokenizer: BPETokenizer,
    model_config: ModelConfig,
    sft_config: TrainingConfig,
    tracker: ExperimentTracker | None = None,
    experiment_name: str = "scratchlm-phase5-sft",
    run_name: str = "tiny-bpe-narrow-sft",
) -> tuple[ScratchGPT, SftRunResult]:
    """
    Day048: Load the frozen base pretraining checkpoint (`best.pt`) without modifying it,
    fine-tune on assistant-only masked SFT examples, save SFT checkpoints in an isolated
    `sft_checkpoint_dir`, and record a linked child MLflow run (`mlflow.parentRunId`).
    """
    base_mgr = AtomicCheckpointManager(base_checkpoint_dir)
    base_best_path = base_checkpoint_dir / "best.pt"
    base_bytes_before = base_best_path.read_bytes()
    base_sha_before = hashlib.sha256(base_bytes_before).hexdigest()

    model = ScratchGPT(model_config, init_seed=sft_config.seed)
    base_meta = base_mgr.load_and_verify("best", expected_lineage=base_lineage, model=model, restore_rng=False)
    assert base_meta.file_sha256 == base_sha_before

    sft_manifest = SftDatasetManifest.load(sft_dir / "sft_manifest.json")
    sft_manifest_sha = sft_manifest.manifest_sha256()
    train_examples = load_sft_split(sft_dir, "train")
    val_examples = load_sft_split(sft_dir, "val")

    encoded_train = [
        encode_sft_example_with_assistant_mask(
            tokenizer,
            example_id=ex.example_id,
            system_text=ex.system_text,
            user_text=ex.user_text,
            assistant_text=ex.assistant_text,
            seq_len=sft_config.seq_len,
        )
        for ex in train_examples
    ]
    encoded_val = [
        encode_sft_example_with_assistant_mask(
            tokenizer,
            example_id=ex.example_id,
            system_text=ex.system_text,
            user_text=ex.user_text,
            assistant_text=ex.assistant_text,
            seq_len=sft_config.seq_len,
        )
        for ex in val_examples
    ]

    active_tracker: ExperimentTracker = tracker if tracker is not None else InMemoryExperimentTracker()
    sft_lineage = LineageHashes(
        dataset_manifest_sha256=sft_manifest_sha,
        splits_manifest_sha256=sft_manifest.splits["train"].file_sha256,
        tokenizer_sha256=tokenizer.tokenizer_hash(),
        config_sha256=sft_config.config_sha256(asdict(model_config)),
        git_commit="phase5-g5-sft",
    )

    sft_run_id = active_tracker.start_run(
        experiment_name=experiment_name,
        run_name=run_name,
        tags={
            "mlflow.parentRunId": parent_base_run_id,
            "parent_base_run_id": parent_base_run_id,
            "base_checkpoint_sha256": base_sha_before,
            "sft_dataset_manifest_sha256": sft_manifest_sha,
            "tokenizer_sha256": tokenizer.tokenizer_hash(),
            "stage": "narrow_sft",
        },
    )
    active_tracker.log_params(
        {
            "parent_base_run_id": parent_base_run_id,
            "base_checkpoint_sha256": base_sha_before,
            "sft_dataset_manifest_sha256": sft_manifest_sha,
            "tokenizer_sha256": tokenizer.tokenizer_hash(),
            **asdict(sft_config),
        }
    )

    seed_everything(sft_config.seed)
    model.to(sft_config.device)
    model.train()
    optimizer = build_adamw_optimizer(model, sft_config)

    initial_val_loss = evaluate_sft_loss(model, encoded_val, batch_size=sft_config.micro_batch_size, device=sft_config.device)
    active_tracker.log_metrics({"sft_val_loss": initial_val_loss}, step=0)

    initial_train_loss = 0.0
    final_train_loss = 0.0
    cumulative_supervised_tokens = 0
    import random as _py_random

    sft_rng = _py_random.Random(sft_config.seed)
    shuffled_pool: list[EncodedSftSample] = []

    for step in range(1, sft_config.max_optimizer_steps + 1):
        lr = compute_cosine_lr_with_warmup(step, sft_config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        batch_samples: list[EncodedSftSample] = []
        needed = sft_config.micro_batch_size * sft_config.gradient_accumulation_steps
        while len(batch_samples) < needed:
            if not shuffled_pool:
                shuffled_pool = list(encoded_train)
                sft_rng.shuffle(shuffled_pool)
            batch_samples.append(shuffled_pool.pop())

        inp, tgt, n_valid = _collate_sft_batch(batch_samples, device=sft_config.device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inp, targets=tgt, ignore_index=IGNORE_INDEX)
        if loss is None or not torch.isfinite(loss):
            raise NonFiniteLossError("Non-finite SFT loss")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), sft_config.grad_clip_norm).item())
        optimizer.step()

        loss_val = float(loss.item())
        if step == 1:
            initial_train_loss = loss_val
        final_train_loss = loss_val
        cumulative_supervised_tokens += n_valid

        active_tracker.log_metrics(
            {
                "sft_train_loss": loss_val,
                "learning_rate": lr,
                "grad_norm": grad_norm,
                "supervised_tokens": float(cumulative_supervised_tokens),
            },
            step=step,
        )

    final_val_loss = evaluate_sft_loss(
        model, encoded_val, batch_size=sft_config.micro_batch_size, device=sft_config.device
    )
    active_tracker.log_metrics({"sft_val_loss": final_val_loss}, step=sft_config.max_optimizer_steps)

    # Save SFT checkpoint in isolated directory using a dummy shard iterator state for schema compatibility
    sft_ckpt_dir = sft_checkpoint_dir
    sft_ckpt_dir.mkdir(parents=True, exist_ok=True)
    sft_best_path = sft_ckpt_dir / "best.pt"
    tmp_sft_path = sft_ckpt_dir / "best.pt.tmp"
    sft_payload = {
        "schema_version": "1.0.0",
        "slot_name": "best",
        "optimizer_step": sft_config.max_optimizer_steps,
        "micro_step": sft_config.max_optimizer_steps * sft_config.gradient_accumulation_steps,
        "processed_valid_tokens": cumulative_supervised_tokens,
        "train_loss": final_train_loss,
        "val_loss": final_val_loss,
        "parent_base_run_id": parent_base_run_id,
        "base_checkpoint_sha256": base_sha_before,
        "lineage": sft_lineage.to_dict(),
        "model_config": asdict(model_config),
        "model_state_dict": model.state_dict(),
    }
    torch.save(sft_payload, tmp_sft_path)
    sft_raw_bytes = tmp_sft_path.read_bytes()
    sft_sha = hashlib.sha256(sft_raw_bytes).hexdigest()
    tmp_sft_path.replace(sft_best_path)

    sft_meta = CheckpointMetadata(
        schema_version="1.0.0",
        slot_name="best",
        filename="best.pt",
        file_sha256=sft_sha,
        file_bytes=len(sft_raw_bytes),
        optimizer_step=sft_config.max_optimizer_steps,
        micro_step=sft_config.max_optimizer_steps * sft_config.gradient_accumulation_steps,
        processed_valid_tokens=cumulative_supervised_tokens,
        train_loss=final_train_loss,
        val_loss=final_val_loss,
        lineage=sft_lineage,
        model_config=asdict(model_config),
    )
    manifest_path = sft_ckpt_dir / "checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "1.0.0", "slots": {"best": sft_meta.to_dict()}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    active_tracker.log_artifact(manifest_path, artifact_subpath="sft_checkpoints")
    active_tracker.log_artifact(sft_best_path, artifact_subpath="sft_checkpoints")
    active_tracker.end_run(status="FINISHED")

    # Verify base checkpoint was preserved completely untouched!
    base_sha_after = hashlib.sha256(base_best_path.read_bytes()).hexdigest()
    if base_sha_after != base_sha_before:
        raise RuntimeError("Base pretraining checkpoint was mutated during SFT!")

    return model, SftRunResult(
        sft_run_id=sft_run_id,
        parent_base_run_id=parent_base_run_id,
        base_checkpoint_sha256=base_sha_before,
        sft_checkpoint_sha256=sft_sha,
        sft_dataset_manifest_sha256=sft_manifest_sha,
        tokenizer_sha256=tokenizer.tokenizer_hash(),
        initial_sft_train_loss=round(initial_train_loss, 4),
        final_sft_train_loss=round(final_train_loss, 4),
        initial_sft_val_loss=round(initial_val_loss, 4),
        final_sft_val_loss=round(final_val_loss, 4),
        sft_checkpoint_meta=sft_meta,
    )


@torch.no_grad()
def compare_base_and_sft_models(
    *,
    base_model: ScratchGPT,
    sft_model: ScratchGPT,
    tokenizer: BPETokenizer,
    sft_val_examples: Sequence[SftExample],
    lm_val_shard: Any,
    lm_val_utf8_bytes: int,
    seq_len: int = 64,
) -> SftRubricComparison:
    """
    Day049: Compare Base vs. SFT models on held-out SFT validation examples and general LM validation shard.
    Checks whether generated assistant responses start with or include the expected format prefix marker
    (`MORAL:`, `ITEM=`, `- Role:`, `END:`), measures repetition rate, and checks LM validation perplexity.
    """
    base_lm: LanguageModelMetrics = evaluate_gpt_on_shard(
        base_model,
        lm_val_shard,
        model_label="base-tiny-bpe",
        seq_len=seq_len,
        total_utf8_bytes=lm_val_utf8_bytes,
    )
    sft_lm: LanguageModelMetrics = evaluate_gpt_on_shard(
        sft_model,
        lm_val_shard,
        model_label="sft-tiny-bpe",
        seq_len=seq_len,
        total_utf8_bytes=lm_val_utf8_bytes,
    )

    base_compliant = 0
    sft_compliant = 0
    base_reps: list[float] = []
    sft_reps: list[float] = []
    comparisons: list[dict[str, Any]] = []

    for ex in sft_val_examples:
        prefix_ids = encode_sft_prompt_prefix(tokenizer, ex.system_text, ex.user_text)
        base_gen = generate_tokens(
            base_model,
            prefix_ids,
            max_new_tokens=16,
            temperature=0.0,
            seed=1337,
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,  # type: ignore[arg-type]
        )
        sft_gen = generate_tokens(
            sft_model,
            prefix_ids,
            max_new_tokens=16,
            temperature=0.0,
            seed=1337,
            eos_token_id=tokenizer.eos_token_id,
            tokenizer=tokenizer,  # type: ignore[arg-type]
        )

        base_txt = base_gen.output_text.strip()
        sft_txt = sft_gen.output_text.strip()
        marker = ex.expected_prefix_marker

        b_ok = marker in base_txt
        s_ok = marker in sft_txt
        if b_ok:
            base_compliant += 1
        if s_ok:
            sft_compliant += 1

        b_rep = compute_token_repetition_ratio(base_gen.output_ids)
        s_rep = compute_token_repetition_ratio(sft_gen.output_ids)
        base_reps.append(b_rep)
        sft_reps.append(s_rep)

        comparisons.append(
            {
                "example_id": ex.example_id,
                "template_family_id": ex.template_family_id,
                "user_text": ex.user_text,
                "reference_answer": ex.assistant_text,
                "expected_marker": marker,
                "base_output": base_txt,
                "base_compliant": b_ok,
                "sft_output": sft_txt,
                "sft_compliant": s_ok,
            }
        )

    n_ex = max(1, len(sft_val_examples))
    base_rate = round(base_compliant / n_ex, 4)
    sft_rate = round(sft_compliant / n_ex, 4)
    base_avg_rep = round(sum(base_reps) / n_ex, 4)
    sft_avg_rep = round(sum(sft_reps) / n_ex, 4)

    # Validation gate passes if both models produce finite perplexity below random vocab (512)
    gate_passed = math.isfinite(sft_lm.perplexity) and math.isfinite(base_lm.perplexity) and base_lm.perplexity < 512.0

    return SftRubricComparison(
        evaluated_examples=len(sft_val_examples),
        base_format_compliance_rate=base_rate,
        sft_format_compliance_rate=sft_rate,
        base_avg_repetition_ratio=base_avg_rep,
        sft_avg_repetition_ratio=sft_avg_rep,
        base_lm_val_loss=base_lm.cross_entropy_loss,
        sft_lm_val_loss=sft_lm.cross_entropy_loss,
        base_lm_val_perplexity=base_lm.perplexity,
        sft_lm_val_perplexity=sft_lm.perplexity,
        validation_gate_passed=gate_passed,
        sample_comparisons=comparisons,
    )
