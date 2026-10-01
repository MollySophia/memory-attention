"""Exact append values, backing capacity, rollback and ownership regression gates."""
import torch
from fla.models.utils import FLALayer, FLACache


def append(layer, tensors, **kwargs):
    return layer.update(attn_state=tensors, cache_kwargs=dict(memory_append=True, **kwargs))['attn_state']


@torch.inference_mode()
def test_growing_rollback_and_transformed_cache():
    torch.manual_seed(19)
    layer = FLALayer()
    expected = tuple(torch.randn(3, 127, 16) for _ in range(2))
    append(layer, expected)
    for i in range(260):
        if i == 130:
            layer.state['attn_state'] = tuple(x[:, :128] for x in layer.state['attn_state'])
            expected = tuple(x[:, :128] for x in expected)
        if i == 160:
            # Simulate a batch reorder; old backing storage must not be reused.
            indices = torch.tensor([2, 0, 1])
            layer.state['attn_state'] = tuple(x[indices] for x in layer.state['attn_state'])
            expected = tuple(x[indices] for x in expected)
        new = tuple(torch.randn(3, 1, 16) for _ in range(2))
        expected = tuple(torch.cat((x, y), 1) for x, y in zip(expected, new))
        actual = append(layer, new)
        for x, y, buf in zip(actual, expected, layer._memory_kv_buffers):
            assert torch.equal(x, y)
            assert 0 <= buf.shape[1] - x.shape[1] < 128


def test_training_keeps_gradients():
    layer = FLALayer()
    x, y = torch.randn(2, 3, 4, requires_grad=True), torch.randn(2, 1, 4, requires_grad=True)
    append(layer, (x, x))
    result = append(layer, (y, y))
    result[0].sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    assert torch.equal(y.grad, torch.ones_like(y))
    assert layer._memory_kv_buffers is None


@torch.inference_mode()
def test_legacy_export_is_independent_of_rollback_append():
    cache = FLACache()
    def update(x):
        cache.update(attn_state=(x, x), cache_kwargs={'memory_append': True})
    update(torch.zeros(2, 3, 4))
    update(torch.ones(2, 1, 4))
    legacy = cache.to_legacy_cache()
    saved = legacy[0]['attn_state'][0].clone()
    cache[0]['attn_state'] = tuple(x[:, :3] for x in cache[0]['attn_state'])
    update(torch.full((2, 1, 4), 7.))
    assert torch.equal(legacy[0]['attn_state'][0], saved)


@torch.inference_mode()
def test_sliding_window_stays_on_original_path():
    layer = FLALayer()
    append(layer, (torch.zeros(2, 4, 8),)*2, window_size=4)
    actual = append(layer, (torch.ones(2, 1, 8),)*2, window_size=4)
    assert layer._memory_kv_buffers is None
    assert actual[0].shape[1] == 4
    assert torch.equal(actual[0][:, -1], torch.ones(2, 8))


@torch.inference_mode()
def test_append_reuses_owned_capacity_and_offload_releases_owner():
    layer = FLALayer()
    append(layer, (torch.zeros(2, 16, 8),)*2)
    first = append(layer, (torch.ones(2, 1, 8),)*2)
    pointers = [x.data_ptr() for x in first]
    second = append(layer, (torch.full((2, 1, 8), 2.),)*2)
    assert [x.data_ptr() for x in second] == pointers
    assert torch.equal(first[0][:, -1], torch.ones(2, 8))
    layer.offload()
    assert layer._memory_kv_buffers is None
    third = append(layer, (torch.full((2, 1, 8), 3.),)*2)
    assert [x.data_ptr() for x in third] != pointers
    assert torch.equal(third[0][:, :18], second[0])
