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
    assert rounds == pytest.approx([2.0, 2.0])
    assert clock.intervals == args.rounds * args.repeats
    assert clock.calls == clock.rollbacks == args.warmup + args.rounds * args.repeats
    assert not clock.timing
