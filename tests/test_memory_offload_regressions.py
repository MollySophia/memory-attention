"""Regression coverage for table placement, model outputs and sweep identity."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'profile' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def model():
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    pytest.importorskip('flash_attn')
    gate = load_script('test_memory_offload')
    with torch.inference_mode():
        instance, _ = gate.build(layers=3, hidden=128, heads=2, vocab=128)
        # Non-unit affine weights expose accidental repeated normalization.
        for layer in instance.model.layers:
            layer.attn.m_norm.weight.copy_(
                torch.linspace(0.5, 1.5, layer.attn.head_dim, device='cuda', dtype=torch.bfloat16)
            )
        try:
            yield instance
        finally:
            instance.close_memory_offload()


@pytest.mark.parametrize('policy', ['bulk', 'pipeline'])
@pytest.mark.parametrize('return_dict', [True, False])
def test_offloaded_hidden_states(model, policy, return_dict):
    ids = torch.randint(0, 128, (2, 8), device='cuda')
    model.fold_memory_table_on_gpu()
    ref = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    model.close_memory_offload()
    model.config.memory_offload_policy = policy
    model.config.memory_offload_group_size = 2
    model.config.memory_offload_prefetch_depth = 1
    model.enable_memory_offload()
    out = model(input_ids=ids, use_cache=False, output_hidden_states=True, return_dict=return_dict)
    logits, hidden = (out.logits, out.hidden_states) if return_dict else out
    assert len(hidden) == len(model.model.layers) + 1
    torch.testing.assert_close(logits, ref.logits, rtol=0, atol=0)
    for actual, expected in zip(hidden, ref.hidden_states):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fold_close_restores_weights_logits_and_device(model):
    ids = torch.randint(0, 128, (2, 8), device='cuda')
    original = [layer.attn.m_proj.weight.clone() for layer in model.model.layers]
    ref = model(input_ids=ids, use_cache=False).logits.clone()
    for _ in range(2):
        model.fold_memory_table_on_gpu()
        with pytest.raises(RuntimeError, match='already folded'):
            model.fold_memory_table_on_gpu()
        with pytest.raises(RuntimeError, match='folded table'):
            model.enable_memory_offload()
        model.close_memory_offload()
        for layer, weight in zip(model.model.layers, original):
            assert not layer.attn.memory_table_folded
            assert layer.attn.m_proj.weight.device == weight.device
            assert layer.attn.m_proj.weight.dtype == weight.dtype
            torch.testing.assert_close(layer.attn.m_proj.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits, ref, rtol=0, atol=0)
        model.close_memory_offload()
        assert all(layer.attn.m_proj.weight.is_cuda for layer in model.model.layers)


def test_each_prefill_case_uses_resident_reference(model):
    gate = load_script('test_memory_offload')
    for policy, batch, length in [('bulk', 2, 8), ('pipeline', 2, 8), ('pipeline', 1, 16)]:
        ids = torch.randint(0, 128, (batch, length), device='cuda')
        # After the first iteration the model is still offloaded. The helper
        # must restore it before producing the next independent reference.
        ref = gate.resident_prefill(model, ids)
        assert model.model.memory_table is None
        assert all(not layer.attn.memory_table_folded for layer in model.model.layers)
        diff, agree = gate.run_case(policy, batch, length, model, model.config, ids, ref)
        assert diff == 0
        assert agree == 100


def test_source_fingerprint_tracks_model_edits_and_ignores_results(tmp_path):
    sweep = load_script('sweep_bmk')
    profile = tmp_path / 'profile'
    layers = tmp_path / 'fla' / 'layers'
    profile.mkdir()
    layers.mkdir(parents=True)
    bmk = profile / 'bench_fla.py'
    bmk.write_text('benchmark = 1\n')
    source = layers / 'memory_offload.py'
    source.write_text('implementation = 1\n')
    before = sweep.source_fingerprint(bmk)
    (profile / 'results').mkdir()
    (profile / 'results' / 'out.json').write_text('{}')
    assert sweep.source_fingerprint(bmk) == before
    source.write_text('implementation = 2\n')
    after = sweep.source_fingerprint(bmk)
    assert after != before
    extra = layers / 'new_operator.py'
    extra.write_text('operator = 1\n')
    assert sweep.source_fingerprint(bmk) != after
    extra.unlink()
    assert sweep.source_fingerprint(bmk) == after


def test_resume_reruns_after_dirty_model_edit(tmp_path, monkeypatch):
    sweep = load_script('sweep_bmk')
    profile = tmp_path / 'profile'
    layers = tmp_path / 'fla' / 'layers'
    profile.mkdir()
    layers.mkdir(parents=True)
    bmk = profile / 'fake_bmk.py'
    bmk.write_text('''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
for name in ('mode', 'variants', 'batch-size', 'seq-len', 'context-len',
             'warmup', 'repeats', 'rounds', 'logits-to-keep', 'json'):
    p.add_argument('--' + name)
a = p.parse_args()
counter = Path(__file__).with_suffix('.count')
counter.write_text(str(int(counter.read_text()) + 1 if counter.exists() else 1))
Path(a.json).write_text(json.dumps(dict(
    config=dict(batch_size=int(a.batch_size), seq_len=int(a.seq_len), context_len=int(a.context_len)),
    results=[dict(variant=a.variants, mode=a.mode, median_ms=1.0)])))
''')
    source = layers / 'memory_offload.py'
    source.write_text('version = 1\n')
    output = tmp_path / 'output'
    monkeypatch.setattr(sweep, 'plot_results', lambda *a: None)
    monkeypatch.setattr(sweep, 'probe_env', lambda b: dict(env=dict(
        git_commit='same_commit', git_dirty=True, source_sha256=sweep.source_fingerprint(b))))
    args = ['sweep_bmk.py', '--bmk', str(bmk), '--output', str(output),
            '--sweep', 'batch', '--batch-sizes', '1', '--fixed-length', '8',
            '--modes', 'prefill', '--variants', 'ma_gpu', '--reference', 'ma_gpu', '--resume']
    monkeypatch.setattr('sys.argv', args)
    assert sweep.main() == 0
    assert sweep.main() == 0
    assert bmk.with_suffix('.count').read_text() == '1'
    source.write_text('version = 2\n')
    with pytest.raises(SystemExit) as exc:
        sweep.main()
    assert exc.value.code == 2
    assert bmk.with_suffix('.count').read_text() == '1'
    monkeypatch.setattr('sys.argv', args + ['--allow-env-change'])
    assert sweep.main() == 0
    assert bmk.with_suffix('.count').read_text() == '2'


@pytest.mark.parametrize('padded', [False, True])
@pytest.mark.parametrize('seed', [1234, 4321])
@pytest.mark.parametrize('policy', ['bulk', 'pipeline', 'auto'])
def test_growing_cache_exact_with_slot_reuse(seed, policy, padded):
    """128 growing steps, non-unit norms, partial groups, GQA and padding."""
    if not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from fla.models.memory.configuration_memory import MemoryConfig
    from fla.models.memory.modeling_memory import MemoryForCausalLM
    torch.manual_seed(seed)
    config = MemoryConfig(hidden_size=128, num_hidden_layers=3, num_heads=4,
                          num_kv_heads=2, vocab_size=128, qk_norm=True,
                          use_gate=True, fuse_norm=False,
                          memory_offload_policy=policy,
                          memory_offload_mapped_bulk_min_tokens=1,
                          memory_offload_mapped_bulk_max_tokens=1024,
                          memory_offload_group_size=2,
                          memory_offload_bulk_max_tokens=1,
                          memory_offload_prefetch_depth=4 if policy == 'auto' else 1)
    with torch.inference_mode():
        model = MemoryForCausalLM(config).to('cuda', dtype=torch.bfloat16).eval()
        for layer in model.model.layers:
            layer.attn.m_norm.weight.copy_(torch.linspace(.5, 1.5, 32, device='cuda'))
        prefix = torch.randint(0, 128, (2, 17), device='cuda')
        tokens = torch.randint(0, 128, (128, 2, 1), device='cuda').unbind()
        mask = torch.ones_like(prefix)
        if padded:
            mask[0, :5] = 0

        def trajectory():
            records, cache = [], None
            for index, token in enumerate((prefix, *tokens)):
                current_mask = torch.cat((mask, torch.ones((2,index),device='cuda',dtype=mask.dtype)), dim=1)
                out = model(input_ids=token, attention_mask=current_mask,
                            past_key_values=cache, use_cache=True,
                            output_hidden_states=True, logits_to_keep=1)
                cache = out.past_key_values
                tensors = [out.logits, *out.hidden_states]
                # Rotary skips writes at negative (left-pad) positions. Those
                # uninitialized K cells are excluded by attention_mask forever.
                # Compare every usable KV element; unpadded cases cover all cells.
                tensors += [t[current_mask.bool()] for state in cache for t in state['attn_state']]
                records.append([t.clone() for t in tensors])
            return records

        try:
            model.fold_memory_table_on_gpu()
            reference = trajectory()
            model.close_memory_offload()
            model.enable_memory_offload()
            for _ in range(2):
                actual = trajectory()
                for expected_step, actual_step in zip(reference, actual):
                    assert len(expected_step) == len(actual_step)
                    for expected, value in zip(expected_step, actual_step):
                        assert torch.isfinite(value).all()
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        finally:
            model.close_memory_offload()


def test_auto_policy_boundary_and_shape_reuse(model):
    model.config.memory_offload_policy = 'auto'
    assert model.config.memory_offload_bulk_max_tokens == 1024
    gate = load_script('test_memory_offload')
    for length, policy in [(1024, 'bulk'), (1025, 'pipeline'), (1, 'bulk'), (1025, 'pipeline')]:
        ids = torch.randint(0, 128, (1, length), device='cuda')
        ref = gate.resident_prefill(model, ids)
        model.enable_memory_offload()
        for _ in range(2):
            actual = model(input_ids=ids, use_cache=False).logits.float()
            assert model.model.memory_offloader.policy == policy
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)


@pytest.mark.parametrize('policy', ['auto', 'pipeline'])
def test_selective_depth_boundary_capacity_and_exactness(model, policy):
    model.config.memory_offload_policy = policy
    assert model.config.memory_offload_single_slot_max_tokens == 2048
    model.config.memory_offload_prefetch_depth = 4
    gate = load_script('test_memory_offload')
    for batch, length in [(1,1024),(1,1025),(1,2048),(1,2049),(2,1025),(1,1025)]:
        ids = torch.randint(0,128,(batch,length),device='cuda')
        expected = gate.resident_prefill(model,ids)
        model.enable_memory_offload()
        for _ in range(2):
            actual = model(input_ids=ids,use_cache=False).logits.float()
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        off = model.model.memory_offloader
        if policy == 'auto' and batch * length <= 1024:
            assert off.policy == 'bulk'
        else:
            slots = 1 if policy == 'auto' and batch * length <= 2048 else min(4,len(model.model.layers))
            assert off.policy == 'pipeline' and len(off.slots) == slots
            expected_bytes = slots * batch * length * off.group * off.dim * off.weights.element_size()
            telemetry = load_script('benchmark_telemetry')
            assert telemetry.storage_bytes(slot['host'] for slot in off.slots) == expected_bytes
            assert telemetry.storage_bytes(slot['gpu'] for slot in off.slots) == expected_bytes


def test_single_slot_limit_can_be_disabled_and_validated(model):
    from fla.models.memory.configuration_memory import MemoryConfig
    with pytest.raises(ValueError):
        MemoryConfig(memory_offload_single_slot_max_tokens=-1)
    model.config.memory_offload_single_slot_max_tokens = 0
    model.config.memory_offload_policy = 'auto'
    model.enable_memory_offload()
    model.set_offload_offloader(1,1025)
    assert len(model.model.memory_offloader.slots) == min(4,len(model.model.layers))


@pytest.mark.parametrize('group', [1, 2, 3])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_shared_host_dma_lifetime_fresh_values_and_partial_groups(group):
    from fla.layers.memory_offload import MemoryTableOffloader
    telemetry = load_script('benchmark_telemetry')
    with torch.inference_mode():
        weights = torch.randn(73, 7, 64, dtype=torch.bfloat16)
        off = MemoryTableOffloader(weights, 2, 17, group_size=group,
                                   prefetch_depth=4, single_host_buffer=True)
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        capacity = 2 * 17 * group * 64 * 2
        assert telemetry.storage_bytes(s['host'] for s in off.slots) == capacity
        assert telemetry.storage_bytes(s['gpu'] for s in off.slots) == capacity * len(off.slots)
        try:
            for iteration in range(4):
                ids = torch.randint(0, 73, (2, 17))
                weights.add_(1)
                actual = []
                def consume(layer, handle):
                    value = handle.acquire()
                    actual.append(value.clone())
                    handle.release()
                # Delay DMA to expose premature overwrite of the shared host
                # buffer. Per-slot events alone cannot guard other slots.
                with torch.cuda.stream(off.copy_stream):
                    torch.cuda._sleep(1_000_000)
                with torch.cuda.stream(streams[iteration % 2]):
                    off.forward(ids, consume)
                torch.cuda.synchronize()
                expected = weights.index_select(0, ids.reshape(-1)).view(2, 17, 7, 64)
                for layer, value in enumerate(actual):
                    torch.testing.assert_close(value.cpu(), expected[:, :, layer], rtol=0, atol=0)
            def fail(layer, handle):
                handle.acquire()
                raise ValueError('consumer failure')
            with pytest.raises(ValueError, match='consumer failure'):
                off.forward(ids, fail)
            with pytest.raises(RuntimeError, match='closed or failed'):
                off.forward(ids, consume)
        finally:
            off.close()


@pytest.mark.parametrize('policy', ['auto', 'pipeline'])
def test_shared_host_policy_boundaries_and_full_model_exactness(model, policy):
    from fla.models.memory.configuration_memory import MemoryConfig
    with pytest.raises(ValueError, match='single-host'):
        MemoryConfig(memory_offload_single_host_min_tokens=16385,
                     memory_offload_single_host_max_tokens=16384)
    model.config.memory_offload_policy = policy
    model.enable_memory_offload()
    for tokens in (4095, 4096, 8192, 8193, 16384, 16385):
        model.set_offload_offloader(1, tokens)
        off = model.model.memory_offloader
        shared = policy == 'auto' and 4096 <= tokens <= 16384
        assert off.single_host_buffer == shared
        # Extending host sharing must not shrink the GPU prefetch capacity.
        assert len(off.slots) == min(4, len(model.model.layers))
        telemetry = load_script('benchmark_telemetry')
        capacity = tokens * off.group * off.dim * off.weights.element_size()
        assert telemetry.storage_bytes(s['host'] for s in off.slots) == capacity * (1 if shared else len(off.slots))
        assert telemetry.storage_bytes(s['gpu'] for s in off.slots) == capacity * len(off.slots)
    # Exercise enabled scheduling cheaply in a full model, including tail group.
    model.close_memory_offload()
    model.config.memory_offload_bulk_max_tokens = 0
    model.config.memory_offload_single_slot_max_tokens = 0
    model.config.memory_offload_single_host_min_tokens = 1
    model.config.memory_offload_single_host_max_tokens = 32
    model.config.memory_offload_group_size = 2
    gate = load_script('test_memory_offload')
    for length in (17, 23, 17):
        ids = torch.randint(0, 128, (1, length), device='cuda')
        expected = gate.resident_prefill(model, ids)
        model.enable_memory_offload()
        for _ in range(2):
            actual = model(input_ids=ids, use_cache=False).logits.float()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('batch', [1, 4, 8, 16])
@pytest.mark.parametrize('ids_device', ['cpu', 'cuda'])
def test_mapped_bulk_fresh_weights_cross_stream_and_host_lifetime(batch, ids_device):
    from fla.layers.memory_mapped_bulk import MappedBulkMemoryTableOffloader
    from fla.layers.memory_offload import PendingM
    with torch.inference_mode():
        weights=torch.randn(73,5,64,dtype=torch.bfloat16).pin_memory()
        off=MappedBulkMemoryTableOffloader(weights,batch,1,'cuda')
        streams=[torch.cuda.Stream(),torch.cuda.Stream()]
        for stream in streams:stream.wait_stream(torch.cuda.current_stream())
        try:
            for iteration in range(4):
                ids=torch.randint(0,73,(batch,1))
                expected=weights.index_select(0,ids.flatten()).clone()
                values=[]
                with torch.cuda.stream(streams[iteration%2]):
                    device_ids=ids.to(ids_device)
                    def consume(layer,value):
                        handle=value if isinstance(value,PendingM) else None
                        if handle is not None:value=handle.acquire()
                        torch.cuda._sleep(100000)
                        values.append(value.clone())
                        if handle is not None:handle.release()
                    off.forward(device_ids,consume)
                # No GPU synchronization before host mutation: forward must
                # already have completed every mapped read of CPU storage.
                weights.add_(torch.tensor(.125,dtype=weights.dtype))
                streams[iteration%2].synchronize()
                actual=torch.stack(values,dim=2).reshape(batch,5,64).cpu()
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        finally:off.close()


def test_mapped_bulk_error_cleanup_and_cpu_bounds():
    from fla.layers.memory_mapped_bulk import MappedBulkMemoryTableOffloader
    with torch.inference_mode():
        weights=torch.randn(17,3,64,dtype=torch.bfloat16).pin_memory()
        for invalid in (False,True):
            off=MappedBulkMemoryTableOffloader(weights,8,1,'cuda')
            try:
                if invalid:
                    with pytest.raises(IndexError,match='out of range'):
                        off.forward(torch.full((8,1),17),lambda *args:None)
                else:
                    def fail(*args):raise ValueError('consumer failure')
                    with pytest.raises(ValueError,match='consumer failure'):
                        off.forward(torch.zeros((8,1),device='cuda',dtype=torch.long),fail)
                assert off.broken
                with pytest.raises(RuntimeError,match='closed or failed'):
                    off.forward(torch.zeros((8,1),dtype=torch.long),lambda *args:None)
            finally:off.close()


def test_mapped_bulk_model_selection_and_pinned_table_accounting(model):
    from fla.layers.memory_mapped_bulk import MappedBulkMemoryTableOffloader
    from fla.layers.memory_offload import BulkMemoryTableOffloader
    model.enable_memory_offload()
    for batch in (1,4,8,16):
        assert isinstance(model.model._offloader_for(batch,1),MappedBulkMemoryTableOffloader)
    assert isinstance(model.model._offloader_for(17,1),BulkMemoryTableOffloader)
    mapped=model.model._offloader_for(8,1)
    assert isinstance(mapped,MappedBulkMemoryTableOffloader)
    assert mapped.host.numel()==0
    snapshot=load_script('benchmark_telemetry').memory_snapshot(model,'cuda')
    assert snapshot['cpu_table_pinned_bytes']==snapshot['cpu_table_bytes']
    assert snapshot['offload_pinned_bytes']==snapshot['cpu_table_bytes']+sum(v['host_bytes'] for v in snapshot['offloader_capacities'])


@pytest.mark.parametrize('ids_device', ['cpu', 'cuda'])
def test_offload_dispatch_resolves_current_shape_once_and_direct_fallback(model, monkeypatch, ids_device):
    ids = torch.randint(0, 128, (2, 8), device=ids_device)
    model.fold_memory_table_on_gpu()
    reference = model(input_ids=ids.cuda(), use_cache=False, output_hidden_states=True)
    model.close_memory_offload()
    model.enable_memory_offload()
    backbone = model.model
    original = backbone._offloader_for
    resolved = []

    def track(batch, length):
        resolved.append((batch, length))
        return original(batch, length)

    monkeypatch.setattr(backbone, '_offloader_for', track)
    actual = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    assert resolved == [(2, 8)]
    torch.testing.assert_close(actual.logits, reference.logits, rtol=0, atol=0)
    for got, expected in zip(actual.hidden_states, reference.hidden_states):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    one_token = ids[:, :1]
    model(input_ids=one_token, use_cache=False)
    assert resolved == [(2, 8), (2, 1)]
    embedded = backbone.embeddings(ids.cuda())
    direct = backbone._forward_offloaded(embedded, ids.cpu(), None, None, False, True, True)
    assert resolved == [(2, 8), (2, 1), (2, 8)]
    torch.testing.assert_close(direct.last_hidden_state, reference.hidden_states[-1], rtol=0, atol=0)


@pytest.mark.parametrize('use_cache', [False, True])
@pytest.mark.parametrize('collect_hidden', [False, True])
@pytest.mark.parametrize('return_dict', [False, True])
def test_offload_dispatch_preserves_hooks_replacements_and_callback_state(use_cache, collect_hidden, return_dict):
    from fla.models.memory.modeling_memory import MemoryModel

    class Block(torch.nn.Module):
        def __init__(self, increment):
            super().__init__()
            self.increment = increment

        def forward(self, hidden, **kwargs):
            assert kwargs['memory_table'] == self.increment
            assert kwargs['output_attentions'] is False
            return hidden + self.increment, kwargs['past_key_values'] + self.increment

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(1), Block(2)])
            self.norm = torch.nn.Identity()

        def _offloader_for(self, batch, length):
            assert (batch, length) == (1, 1)
            return self

        def forward(self, ids, consume):
            for index, layer in enumerate(self.layers):
                consume(index, layer.increment)

    stub = Stub()
    hidden = torch.zeros(1, 1, 4)
    ids = torch.zeros(1, 1, dtype=torch.long)
    calls = []
    for increment in (2, 4):
        stub.layers[1] = Block(increment)
        handle = stub.layers[1].register_forward_hook(lambda module, args, output: calls.append(module.increment))
        result = MemoryModel._forward_offloaded(stub, hidden, ids, None, 10, use_cache, collect_hidden, return_dict)
        output = result.last_hidden_state if return_dict else result[0]
        cache = result.past_key_values if return_dict else result[1]
        assert torch.equal(output, hidden + 1 + increment)
        assert cache == (11 + increment if use_cache else 10)
        if collect_hidden:
            states = result.hidden_states if return_dict else result[2]
            assert len(states) == 3
            for got, value in zip(states, (0, 1, 1 + increment)):
                assert torch.equal(got, hidden + value)
        elif not return_dict:
            assert len(result) == 2
        handle.remove()
    assert calls == [2, 4]


@pytest.mark.parametrize('group', [1, 2, 7])
@pytest.mark.parametrize('depth', [1, 4])
@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_mapped_first_group_fresh_reads_slot_reuse_and_host_lifetime(group, depth, shared, monkeypatch):
    from fla.layers.memory_offload import MemoryTableOffloader
    import fla.layers.memory_mapped_first as kernel
    with torch.inference_mode():
        weights = torch.randn(73, 5, 64, dtype=torch.bfloat16).pin_memory()
        off = MemoryTableOffloader(weights, 2, 17, group_size=group,
                                   prefetch_depth=depth, single_host_buffer=shared,
                                   mapped_first_group=True)
        assert off.mapped_first_group
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        launches = []
        original = kernel.mapped_first_group
        def track(*args):
            launches.append(args[-1])
            return original(*args)
        monkeypatch.setattr(kernel, 'mapped_first_group', track)
        try:
            for iteration in range(3):
                ids = torch.randint(0, 73, (2, 17))
                expected = weights.index_select(0, ids.flatten()).view(2, 17, 5, 64).clone()
                actual = []
                def consume(layer, handle):
                    value = handle.acquire()
                    torch.cuda._sleep(100000)
                    actual.append(value.clone())
                    handle.release()
                with torch.cuda.stream(off.copy_stream):
                    torch.cuda._sleep(1000000)
                with torch.cuda.stream(streams[iteration % 2]):
                    off.forward(ids, consume)
                # Returning must end mapped host reads, even if compute is live.
                weights.add_(0.125)
                streams[iteration % 2].synchronize()
                for layer, value in enumerate(actual):
                    torch.testing.assert_close(value.cpu(), expected[:, :, layer], rtol=0, atol=0)
            assert launches == [min(group, 5)] * 3
            for bad in (-1, 73):
                with pytest.raises(IndexError, match='out of range'):
                    off.forward(torch.full((2, 17), bad), consume)
            def fail(layer, handle):
                handle.acquire()
                raise ValueError('consumer failure')
            with pytest.raises(ValueError, match='consumer failure'):
                off.forward(ids, fail)
            weights.zero_()
            with pytest.raises(RuntimeError, match='closed or failed'):
                off.forward(ids, consume)
        finally:
            off.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_mapped_first_group_fallback_and_launch_failure():
    from fla.layers.memory_offload import MemoryTableOffloader
    from unittest.mock import patch
    with torch.inference_mode():
        for pinned, enabled in ((False, True), (True, False)):
            weights = torch.randn(17, 3, 64, dtype=torch.bfloat16)
            if pinned:
                weights = weights.pin_memory()
            off = MemoryTableOffloader(weights, 1, 3, mapped_first_group=enabled)
            assert not off.mapped_first_group
            off.close()
        weights = torch.randn(17, 3, 64, dtype=torch.bfloat16).pin_memory()
        off = MemoryTableOffloader(weights, 1, 3, mapped_first_group=True)
        def consume(layer, handle):
            handle.acquire()
            handle.release()
        try:
            with patch('fla.layers.memory_mapped_first.mapped_first_group', side_effect=ValueError('launch failure')):
                with pytest.raises(RuntimeError, match='prefetch failed'):
                    off.forward(torch.zeros((1, 3), dtype=torch.long), consume)
            assert off.broken
            weights.zero_()
        finally:
            off.close()


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('pin_table', [False, True])
def test_mapped_first_group_model_policy_and_growing_cache(model, enabled, pin_table):
    model.config.memory_offload_bulk_max_tokens = 0
    model.config.memory_offload_mapped_first_group = enabled
    model.config.memory_offload_mapped_bulk = pin_table
    prefix = torch.randint(0, 128, (2, 17), device='cuda')
    tokens = [torch.randint(0, 128, (2, 1), device='cuda') for _ in range(3)]
    def trajectory():
        output = None
        results = []
        for ids in [prefix, *tokens]:
            output = model(input_ids=ids, past_key_values=None if output is None else output.past_key_values,
                           use_cache=True, output_hidden_states=True)
            results.append((output.logits.clone(), tuple(x.clone() for x in output.hidden_states)))
        return results
    model.fold_memory_table_on_gpu()
    expected = trajectory()
    model.close_memory_offload()
    model.enable_memory_offload()
    actual = trajectory()
    assert model.model._offloader_for(2, 17).mapped_first_group == (enabled and pin_table)
    for (logits, hidden), (ref_logits, ref_hidden) in zip(actual, expected):
        torch.testing.assert_close(logits, ref_logits, rtol=0, atol=0)
        for got, ref in zip(hidden, ref_hidden):
            torch.testing.assert_close(got, ref, rtol=0, atol=0)
