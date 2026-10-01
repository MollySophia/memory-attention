"""Verify fixed-context decode measures model work without benchmark rollback."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_decode_excludes_rollback_and_keeps_context_fixed(monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'profile' / 'bench_fla.py'
    spec = importlib.util.spec_from_file_location('bench_fla', script)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    args = SimpleNamespace(vocab_size=16, batch_size=2, context_len=8,
                           warmup=2, repeats=3, rounds=2, logits_to_keep=1)
    clock = SimpleNamespace(now=0.0, timing=False, intervals=0, rollbacks=0, calls=0)

    def timestamp():
        clock.timing = not clock.timing
        if not clock.timing:
            clock.intervals += 1
        return clock.now

    original_rollback = bench.rollback

    def rollback(cache, length):
        assert not clock.timing, 'benchmark rollback entered the timed interval'
        clock.rollbacks += 1
        clock.now += 0.1  # Large setup cost must not appear in decode samples.
        original_rollback(cache, length)

    def model(input_ids, past_key_values, use_cache, logits_to_keep=0):
        assert use_cache
        cache = past_key_values
        if input_ids.shape[1] == args.context_len:
            cache.states = [dict(attn_state=[torch.zeros(2, args.context_len, 1)] * 2)]
            cache._seen_tokens = args.context_len
        else:
            assert input_ids.shape == (args.batch_size, 1)
            assert logits_to_keep == 1
            assert cache._seen_tokens == args.context_len
            assert all(t.shape[1] == args.context_len for t in cache.states[0]['attn_state'])
            clock.calls += 1
            clock.now += 0.002  # Includes the model's own cache update.
            cache.states[0]['attn_state'] = [
                torch.cat((t, torch.zeros(2, 1, 1)), dim=1)
                for t in cache.states[0]['attn_state']
            ]
            cache._seen_tokens += 1
        return SimpleNamespace(past_key_values=cache, logits=torch.zeros(2, 1, 16))

    monkeypatch.setattr(bench, 'Cache', SimpleNamespace(from_legacy_cache=lambda _: SimpleNamespace()))
    monkeypatch.setattr(bench, 'rollback', rollback)
    monkeypatch.setattr(bench, 'time', SimpleNamespace(perf_counter=timestamp))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda _: None)
    rounds = bench.run_decode(model, args, 'cpu')
    assert rounds["round_ms"] == pytest.approx([2.0, 2.0])
    for samples in rounds["samples_ms"]:
        assert samples == pytest.approx([2.0] * args.repeats)
    assert clock.intervals == args.rounds * args.repeats
    assert clock.calls == clock.rollbacks == args.warmup + args.rounds * args.repeats
    assert not clock.timing


@pytest.mark.parametrize("workload,keep,cache", [("inference", 1, True), ("historical", 0, False)])
def test_prefill_scope_and_raw_samples(monkeypatch, workload, keep, cache):
    script = Path(__file__).resolve().parents[1] / 'profile' / 'bench_fla.py'
    spec = importlib.util.spec_from_file_location('bench_fla_prefill', script)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    args = SimpleNamespace(vocab_size=16, batch_size=2, seq_len=8,
                           warmup=2, repeats=3, rounds=2, logits_to_keep=1,
                           prefill_workload=workload)
    calls = []
    ticks = iter([0, .001, 1, 1.002, 2, 2.003, 3, 3.004, 4, 4.005, 5, 5.006])

    def model(**kwargs):
        assert kwargs['use_cache'] is cache
        assert kwargs['logits_to_keep'] == keep
        assert kwargs['input_ids'].shape == (2, 8)
        assert 'past_key_values' not in kwargs
        calls.append(kwargs['input_ids'].clone())

    monkeypatch.setattr(torch.cuda, 'synchronize', lambda _: None)
    monkeypatch.setattr(bench, 'time', SimpleNamespace(perf_counter=lambda: next(ticks)))
    result = bench.run_prefill(model, args, 'cpu')
    assert result['round_ms'] == pytest.approx([2, 5])
    assert result['samples_ms'][0] == pytest.approx([1, 2, 3])
    assert result['samples_ms'][1] == pytest.approx([4, 5, 6])
    assert len(calls) == 8
    assert all(torch.equal(calls[0], ids) for ids in calls)


def test_generation_grows_and_restarts_cache(monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'profile' / 'bench_fla.py'
    spec = importlib.util.spec_from_file_location('bench_generation', script)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    args = SimpleNamespace(vocab_size=16, batch_size=2, context_len=8,
                           warmup=1, repeats=2, rounds=2, logits_to_keep=1,
                           generation_steps=128)
    lengths, inputs = [], []
    ticks = SimpleNamespace(now=0.)

    def model(input_ids, use_cache, logits_to_keep, past_key_values=None):
        assert use_cache and logits_to_keep == 1
        if past_key_values is None:
            assert input_ids.shape == (2, 8)
            cache = SimpleNamespace(length=8)
            inputs.append([])
        else:
            assert input_ids.shape == (2, 1)
            cache = past_key_values
            cache.length += 1
            inputs[-1].append(input_ids.clone())
        lengths.append(cache.length)
        ticks.now += .001
        return SimpleNamespace(past_key_values=cache)

    monkeypatch.setattr(torch.cuda, 'synchronize', lambda _: None)
    monkeypatch.setattr(bench, 'time', SimpleNamespace(perf_counter=lambda: ticks.now))
    result = bench.run_generation(model, args, 'cpu')
    assert lengths == list(range(8, 137)) * 5
    assert result['round_ms'] == pytest.approx([129., 129.])
    assert all(torch.equal(torch.stack(inputs[0]), torch.stack(x)) for x in inputs)


@pytest.mark.parametrize('oom', [False, True])
def test_benchmark_persists_setup_failure(monkeypatch, tmp_path, oom):
    script = Path(__file__).resolve().parents[1] / 'profile' / 'bench_fla.py'
    spec = importlib.util.spec_from_file_location('bench_failure', script)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    args = SimpleNamespace(device='cuda:0', json=tmp_path/'failure.json', stage='screening', measurement_plan_id='screen_v1_w3_n5_r1')
    monkeypatch.setattr(bench, 'parse_args', lambda: args)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(bench, 'environment_details', lambda: {})
    monkeypatch.setattr(bench, 'source_state', lambda: {})
    monkeypatch.setattr(bench, 'env_fingerprint', lambda: {})

    def fail(_):
        raise torch.cuda.OutOfMemoryError('injected OOM') if oom else RuntimeError('injected failure')

    monkeypatch.setattr(bench, 'build', fail)
    assert bench.main() == 1
    import json
    result = json.loads(args.json.read_text())
    assert result['status'] == ('oom' if oom else 'benchmark_failed')
    assert result['results'] == []
    assert result['model_config'] is None
    assert 'injected' in result['failure']['message']


def test_paper_matrix_covers_goal():
    script = Path(__file__).resolve().parents[1] / 'profile' / 'run_paper_matrix.py'
    spec = importlib.util.spec_from_file_location('paper_matrix', script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert len(list(runner.jobs())) == 8
    jobs = list(runner.jobs('full_validation'))
    assert len(jobs) == 48
    assert len({tuple(j.values()) for j in jobs}) == 48
    for variant in ('ma_offload', 'ma_gpu', 'ma_gpu_unfolded'):
        for mode in ('prefill', 'decode'):
            shapes = {(j['batch'], j['length']) for j in jobs if j['variant'] == variant and j['mode'] == mode}
            assert shapes == {(b, 2048) for b in (1,4,8,16)} | {(8,l) for l in (512,2048,4096,8192)}
        assert {j['batch'] for j in jobs if j['mode']=='generation' and j['variant']==variant} == {1,8}
