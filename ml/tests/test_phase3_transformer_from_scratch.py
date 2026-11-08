"""
Phase 3 (Day021-Day030, Gate G3) comprehensive unit and integration tests for the
from-scratch decoder-only Transformer (ByteTokenizer, attention, LayerNorm, PreNormTransformerBlock,
ScratchGPT, initialization, sampling, and toy-batch overfit).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from experiments.overfit_tiny_batch import run_overfit_tiny_batch, verify_full_model_causality
from experiments.tensor_basics import run_tensor_basics_check
from scratchlm.config import ModelConfig, load_model_config
from scratchlm.inference.sampling import generate_tokens, sample_next_token, stream_generate_utf8
from scratchlm.model.attention import (
    CausalMultiHeadSelfAttention,
    build_causal_mask,
    causal_scaled_dot_product_attention,
    stable_softmax,
)
from scratchlm.model.blocks import PreNormTransformerBlock, TokenAndPositionEmbedding
from scratchlm.model.gpt import ScratchGPT, masked_next_token_cross_entropy
from scratchlm.model.normalization import ScratchLayerNorm
from scratchlm.tokenizer.byte_tokenizer import (
    ASSISTANT_TOKEN_ID,
    BOS_TOKEN_ID,
    END_TURN_TOKEN_ID,
    EOS_TOKEN_ID,
    IGNORE_INDEX,
    PAD_TOKEN_ID,
    SYSTEM_TOKEN_ID,
    USER_TOKEN_ID,
    ByteTokenizer,
    build_shifted_next_token_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"


def test_day021_tensor_basics_and_numerical_finite_difference_gradient() -> None:
    report = run_tensor_basics_check(seed=1337)
    assert report.passed is True
    assert report.max_finite_diff_error < 1e-7


def test_day022_byte_tokenizer_roundtrip_arabic_emoji_and_incremental_streaming() -> None:
    tokenizer = ByteTokenizer()
    samples = [
        "",
        "   \n\t  ",
        "ASCII storytelling 123!",
        "مرحبا بالعالم — لغة عربية",  # 2-byte UTF-8 Arabic codepoints
        "Robot 🤖 rocket 🚀 stars ✨",  # 4-byte UTF-8 emojis
        "Untrusted user text containing literal <|system|> and <|assistant|> markers",
    ]
    for sample in samples:
        encoded = tokenizer.encode(sample, add_bos=True, add_eos=True)
        assert encoded[0] == BOS_TOKEN_ID
        assert encoded[-1] == EOS_TOKEN_ID
        # Ensure literal "<|system|>" in user text is encoded as raw bytes < 256, never privileged 259
        assert all(0 <= tid < 256 for tid in encoded[1:-1])
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        assert decoded == sample

        # Verify token-by-token streaming via IncrementalUtf8TokenStreamDecoder never emits U+FFFD
        stream_dec = tokenizer.create_stream_decoder()
        streamed_pieces = [stream_dec.step(tid) for tid in encoded]
        streamed_pieces.append(stream_dec.flush())
        reconstructed = "".join(streamed_pieces)
        assert reconstructed == sample
        assert "\ufffd" not in reconstructed

    # Structural chat turns use privileged role IDs (259..263)
    chat_ids = tokenizer.encode_chat_turns(
        [("system", "Be concise."), ("user", "Hi!")],
        add_generation_prompt=True,
    )
    assert SYSTEM_TOKEN_ID in chat_ids
    assert USER_TOKEN_ID in chat_ids
    assert END_TURN_TOKEN_ID in chat_ids
    assert chat_ids[-1] == ASSISTANT_TOKEN_ID


def test_day022_shifted_next_token_batch_distinguishes_pad_from_null_byte() -> None:
    # Include actual null byte 0x00 (token 0) inside a sequence
    seq1 = [BOS_TOKEN_ID, 65, 0, 66, EOS_TOKEN_ID]
    seq2 = [BOS_TOKEN_ID, 67, EOS_TOKEN_ID]

    batch = build_shifted_next_token_batch([seq1, seq2], max_seq_len=8)
    assert batch.input_ids.shape == (2, 4)
    assert batch.target_ids.shape == (2, 4)

    # seq1 has no padding: all 4 targets (including byte 0x00 at index 1) must be supervised!
    assert batch.input_ids[0].tolist() == [BOS_TOKEN_ID, 65, 0, 66]
    assert batch.target_ids[0].tolist() == [65, 0, 66, EOS_TOKEN_ID]
    assert batch.loss_mask[0].tolist() == [True, True, True, True]

    # seq2 has 2 valid targets followed by 2 padded positions (PAD_TOKEN_ID = 256 -> IGNORE_INDEX = -100)
    assert batch.input_ids[1].tolist() == [BOS_TOKEN_ID, 67, PAD_TOKEN_ID, PAD_TOKEN_ID]
    assert batch.target_ids[1].tolist() == [67, EOS_TOKEN_ID, IGNORE_INDEX, IGNORE_INDEX]
    assert batch.loss_mask[1].tolist() == [True, True, False, False]
    assert batch.valid_target_count == 6


def test_day023_hand_computable_causal_attention_and_future_invariance() -> None:
    # Hand-computable 2-step, 1-head, d_head=2 example
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)  # [1, 1, 2, 2]
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
    v = torch.tensor([[[[10.0, 20.0], [30.0, 40.0]]]], dtype=torch.float32)

    context, weights = causal_scaled_dot_product_attention(q, k, v)
    # Row 0 can only attend to key 0 -> weights[0, 0, 0] == [1.0, 0.0], context[0] == [10.0, 20.0]
    assert torch.allclose(weights[0, 0, 0], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(context[0, 0, 0], torch.tensor([10.0, 20.0]), atol=1e-6)

    # Row 1 has q1=[0,1], k0=[1,0] (dot=0), k1=[0,1] (dot=1/sqrt(2))
    s0 = 0.0
    s1 = 1.0 / math.sqrt(2.0)
    w0 = math.exp(s0) / (math.exp(s0) + math.exp(s1))
    w1 = math.exp(s1) / (math.exp(s0) + math.exp(s1))
    assert torch.allclose(weights[0, 0, 1], torch.tensor([w0, w1]), atol=1e-6)

    # Modifying future token k[:, :, 1, :] or v[:, :, 1, :] cannot alter position 0 output
    v_mut = v.clone()
    v_mut[0, 0, 1] = torch.tensor([999.0, -999.0])
    context_mut, _ = causal_scaled_dot_product_attention(q, k, v_mut)
    assert torch.equal(context[0, 0, 0], context_mut[0, 0, 0])

    # Verify stable_softmax handles extreme positive offsets without NaN/Inf
    huge_logits = torch.tensor([[1000.0, 1000.0, 999.0]], dtype=torch.float32)
    probs = stable_softmax(huge_logits, dim=-1)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.tensor([1.0]), atol=1e-6)


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8])
def test_day024_multi_head_attention_shapes_and_no_nn_multihead_attention(n_heads: int) -> None:
    d_model = 32
    mha = CausalMultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, dropout=0.0, use_bias=True)
    for mod in mha.modules():
        assert not isinstance(mod, nn.MultiheadAttention)

    x = torch.randn(2, 5, d_model)
    out, weights = mha(x, return_weights=True)
    assert out.shape == (2, 5, d_model)
    assert weights.shape == (2, n_heads, 5, 5)
    # Upper triangle above diagonal must be strictly 0.0
    upper_tri = torch.triu(weights, diagonal=1)
    assert float(upper_tri.abs().max().item()) == 0.0


def test_day025_scratch_layer_norm_matches_pytorch_reference_and_handles_constant_input() -> None:
    d_model = 16
    ln_scratch = ScratchLayerNorm(d_model, eps=1e-5)
    ln_ref = nn.LayerNorm(d_model, eps=1e-5)

    gen = torch.Generator().manual_seed(2026)
    with torch.no_grad():
        gamma = torch.randn(d_model, generator=gen)
        beta = torch.randn(d_model, generator=gen)
        ln_scratch.weight.copy_(gamma)
        ln_scratch.bias.copy_(beta)
        ln_ref.weight.copy_(gamma)
        ln_ref.bias.copy_(beta)

    x1 = torch.randn(3, 7, d_model, generator=gen, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y_scratch = ln_scratch(x1)
    y_ref = ln_ref(x2)
    assert torch.allclose(y_scratch, y_ref, atol=1e-5)

    (y_scratch.sum()).backward()
    (y_ref.sum()).backward()
    assert x1.grad is not None and x2.grad is not None
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)

    # Constant input has 0 variance; output must equal beta finitely without NaNs
    const_in = torch.full((2, 4, d_model), 7.5, dtype=torch.float32)
    const_out = ln_scratch(const_in)
    assert torch.isfinite(const_out).all()
    assert torch.allclose(const_out, beta.view(1, 1, d_model).expand_as(const_out), atol=1e-5)


def test_day026_prenorm_block_and_embedding_context_overflow_protection() -> None:
    emb = TokenAndPositionEmbedding(vocab_size=264, max_seq_len=16, d_model=32, dropout=0.0)
    valid_ids = torch.zeros((2, 16), dtype=torch.long)
    out = emb(valid_ids)
    assert out.shape == (2, 16, 32)

    with pytest.raises(ValueError, match="Context length overflow"):
        emb(torch.zeros((1, 17), dtype=torch.long))

    with pytest.raises(ValueError, match="Token ID out of range"):
        emb(torch.tensor([[264]], dtype=torch.long))

    block = PreNormTransformerBlock(d_model=32, n_heads=4, d_ff=128, dropout=0.1)
    block.eval()
    # In eval mode, dropout is disabled so two forward passes on the same input are identical
    y1 = block(out)
    y2 = block(out)
    assert torch.equal(y1, y2)


def test_day027_scratch_gpt_parameter_count_weight_tying_and_residual_scaling() -> None:
    tiny_cfg = load_model_config(ML_ROOT / "configs" / "tiny.json")
    model = ScratchGPT(tiny_cfg, init_seed=1337)

    # Exact parameter count matches closed-form formula (843,520)
    assert model.exact_num_parameters() == tiny_cfg.exact_parameter_count() == 843_520

    # Tied input and output embeddings share the exact underlying storage pointer
    assert (
        model.lm_head.weight.data_ptr()
        == model.embeddings.token_embedding.weight.data_ptr()
    )

    # Verify residual output projections have std scaled by 1 / sqrt(2 * n_layers)
    expected_res_std = 0.02 / math.sqrt(2.0 * tiny_cfg.n_layers)
    measured_res_std = float(model.blocks[0].mlp.fc_down.weight.std().item())
    assert abs(measured_res_std - expected_res_std) < 0.0015

    # Forward + backward produces finite loss and finite non-zero gradients
    x = torch.randint(0, tiny_cfg.vocab_size, (2, 12))
    y = torch.randint(0, tiny_cfg.vocab_size, (2, 12))
    logits, loss = model(x, y)
    assert logits.shape == (2, 12, tiny_cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient on {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {name}"

    # Zero supervised tokens must fail explicitly
    all_ignored = torch.full((2, 12), IGNORE_INDEX, dtype=torch.long)
    with pytest.raises(ValueError, match="zero supervised target tokens"):
        masked_next_token_cross_entropy(logits, all_ignored)


def test_day028_sampling_determinism_eos_stopping_and_context_overflow() -> None:
    tiny_cfg = load_model_config(ML_ROOT / "configs" / "tiny.json")
    model = ScratchGPT(tiny_cfg, init_seed=1337)
    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode("Hello", add_bos=True)

    # Seeded temperature + top-k + top-p sampling is 100% repeatable for identical seed
    r1 = generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=10,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        seed=999,
        tokenizer=tokenizer,
    )
    r2 = generate_tokens(
        model,
        prompt_ids,
        max_new_tokens=10,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        seed=999,
        tokenizer=tokenizer,
    )
    assert r1.output_ids == r2.output_ids
    assert r1.output_text == r2.output_text

    # Streaming UTF-8 generator produces the exact same decoded text
    streamed = "".join(
        stream_generate_utf8(
            model,
            prompt_ids,
            max_new_tokens=10,
            temperature=0.8,
            top_k=20,
            top_p=0.9,
            seed=999,
            tokenizer=tokenizer,
        )
    )
    assert streamed == r1.output_text

    # Context overflow is rejected clearly
    with pytest.raises(ValueError, match="Context overflow"):
        generate_tokens(model, prompt_ids, max_new_tokens=tiny_cfg.max_seq_len)


def test_day029_overfit_tiny_batch_beats_unigram_baseline_and_proves_causality() -> None:
    res = run_overfit_tiny_batch(steps=140, lr=3e-3, seed=1337)
    assert res.passed is True
    assert res.final_loss < res.unigram_baseline_loss
    assert res.final_loss < 0.15
    assert res.causality_max_abs_diff == 0.0
    assert "little robot counted 1, 2, 3." in res.greedy_memorized_text
