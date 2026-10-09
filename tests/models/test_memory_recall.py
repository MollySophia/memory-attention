# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""No-RoPE recall must match full KV caching while retaining no V tensors."""

import pytest
import torch

from fla.models.memory.configuration_memory import MemoryConfig
from fla.models.memory.modeling_memory import MemoryForCausalLM


def _model(recall, qk_norm=False, kv_heads=2, window=None):
    torch.manual_seed(1234)
    config = MemoryConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_heads=2,
        num_kv_heads=kv_heads,
        intermediate_size=256,
        vocab_size=128,
        rope_theta=None,
        memory_recall=recall,
        qk_norm=qk_norm,
        qkv_bias=True,
        window_size=window,
        use_gate=True,
        use_head_gate=True,
        fuse_norm=False,
    )
    model = MemoryForCausalLM(config).to('cuda', dtype=torch.bfloat16).eval()
    for layer in model.model.layers:
        layer.attn.m_norm.weight.copy_(torch.linspace(0.5, 1.5, 64, device='cuda'))
    return model


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    pytest.importorskip('flash_attn')


def _cache_bytes(cache):
    return sum(t.untyped_storage().nbytes() for state in cache for t in state['attn_state'])


@pytest.mark.parametrize('qk_norm', [False, True])
@pytest.mark.parametrize('kv_heads', [1, 2])
@pytest.mark.parametrize('folded', [False, True])
@pytest.mark.parametrize('window,padded', [(None, False), (8, True)])
@torch.inference_mode()
def test_recall_matches_full_kv(qk_norm, kv_heads, folded, window, padded):
    _cuda()
    reference = _model(False, qk_norm, kv_heads, window)
    recall = _model(True, qk_norm, kv_heads, window)
    if folded:
        reference.fold_memory_table_on_gpu()
        recall.fold_memory_table_on_gpu()
    ids = torch.randint(0, 128, (2, 16), device='cuda')
    mask = torch.ones_like(ids)
    if padded:
        mask[0, :3] = 0
    ref_cache = cache = None
    history = ids.clone()
    for step in range(17):
        if step:
            ids = torch.randint(0, 128, (2, 1), device='cuda')
            history = torch.cat((history, ids), dim=1)
            mask = torch.cat((mask, torch.ones_like(ids)), dim=1)
        expected = reference(input_ids=ids, attention_mask=mask if padded else None,
                             past_key_values=ref_cache, use_cache=True,
                             output_hidden_states=True, logits_to_keep=1)
        actual = recall(input_ids=ids, attention_mask=mask if padded else None,
                        past_key_values=cache, use_cache=True,
                        output_hidden_states=True, logits_to_keep=1)
        ref_cache, cache = expected.past_key_values, actual.past_key_values
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        for a, b in zip(actual.hidden_states, expected.hidden_states, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for state in cache:
            raw_k, token_ids = state['attn_state']
            assert raw_k.dtype == torch.bfloat16
            assert token_ids.dtype == torch.long
            assert raw_k.shape[-1] == kv_heads * 64 and token_ids.shape[-1] == 1
            torch.testing.assert_close(token_ids.squeeze(-1), history[:, -raw_k.shape[1]:])
        # BF16 keys plus int64 IDs replace BF16 keys and values. After decode
        # there are no larger prefix backing stores retained by cache views.
        if step:
            expected_ratio = 0.5 + 2 / (kv_heads * 64)
            assert _cache_bytes(cache) / _cache_bytes(ref_cache) == pytest.approx(expected_ratio)


@torch.inference_mode()
def test_recall_128_steps_and_uncached_reference():
    _cuda()
    reference, recall = _model(False), _model(True)
    ids = torch.randint(0, 128, (1, 8), device='cuda')
    history = ids.clone()
    cache = None
    for step in range(129):
        if step:
            ids = torch.randint(0, 128, (1, 1), device='cuda')
            history = torch.cat((history, ids), dim=1)
        expected = reference(input_ids=history, use_cache=False, logits_to_keep=1)
        actual = recall(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = actual.past_key_values
        # Full-prefix and one-token projection/attention kernels can round differently.
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0.02, atol=0.005)
    cached_ids = cache[0]['attn_state'][1].clone()
    ids.zero_()
    torch.testing.assert_close(cache[0]['attn_state'][1], cached_ids)


@torch.inference_mode()
def test_recall_rejects_offload_and_full_kv_cache():
    _cuda()
    reference, recall = _model(False), _model(True)
    ids = torch.randint(0, 128, (1, 4), device='cuda')
    cache = reference(input_ids=ids, use_cache=True).past_key_values
    with pytest.raises(ValueError, match='fresh cache'):
        recall(input_ids=ids[:, :1], past_key_values=cache, use_cache=True)
    with pytest.raises(ValueError, match='GPU-resident'):
        recall.enable_memory_offload()
    assert all(layer.attn.m_proj is not None for layer in recall.model.layers)


@torch.inference_mode()
def test_recall_owns_prefix_ids():
    _cuda()
    reference, recall = _model(False), _model(True)
    ids = torch.randint(0, 128, (2, 8), device='cuda')
    saved_ids = ids.clone()
    cache = recall(input_ids=ids, use_cache=True).past_key_values
    ids.zero_()
    for state in cache:
        torch.testing.assert_close(state['attn_state'][1].squeeze(-1), saved_ids)
    next_ids = torch.randint(0, 128, (2, 1), device='cuda')
    expected = reference(input_ids=torch.cat((saved_ids, next_ids), dim=1), use_cache=False, logits_to_keep=1)
    actual = recall(input_ids=next_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0.02, atol=0.005)


def test_recall_config_roundtrip():
    config = MemoryConfig(rope_theta=None, memory_recall=True)
    restored = MemoryConfig.from_dict(config.to_dict())
    assert restored.memory_recall and restored.rope_theta is None
    assert MemoryConfig().rope_theta == 10000. and not MemoryConfig().memory_recall
    with pytest.raises(ValueError, match='rope_theta=None'):
        MemoryConfig(memory_recall=True)
    with pytest.raises(ValueError, match='GPU-resident'):
        MemoryConfig(memory_recall=True, rope_theta=None, memory_offload=True)
