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


@pytest.mark.parametrize('offload', [False, True])
def test_cached_prefill_last_token_scope(model, offload):
    ids = torch.randint(0, 128, (2, 8), device='cuda')
    if offload:
        model.enable_memory_offload()
    else:
        model.fold_memory_table_on_gpu()
    full = model(input_ids=ids, use_cache=True, logits_to_keep=0)
    last = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    assert last.logits.shape == (2, 1, 128)
    torch.testing.assert_close(last.logits, full.logits[:, -1:], rtol=0, atol=0)
    assert last.past_key_values.get_seq_length() == 8
    for actual, expected in zip(last.past_key_values, full.past_key_values):
        for a, b in zip(actual['attn_state'], expected['attn_state']):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
