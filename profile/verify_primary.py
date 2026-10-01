"""Exact folded-resident/offload gate at the GOAL.md primary shape.

Keeps reference tensors on CPU so reference retention cannot cause GPU OOM.
This is correctness work, never a latency measurement.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from bench_fla import build
from benchmark_telemetry import source_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--decode-steps', type=int, default=3)
    args = parser.parse_args()
    config = SimpleNamespace(seed=1234, hidden_size=2048, num_layers=24,
                             num_heads=32, num_kv_heads=32, vocab_size=32000,
                             hidden_ratio=4, intermediate_size=5632, group_size=1,
                             prefetch_depth=4, policy='auto', device='cuda:0')
    model = build(config)
    assert sum(p.numel() for p in model.parameters()) == 2836499968
    torch.manual_seed(1235)
    prefix = torch.randint(0, 32000, (8, 2048), device='cuda')
    tokens = torch.randint(0, 32000, (args.decode_steps, 8, 1), device='cuda').unbind(0)
    reference = []
    comparisons = []

    def tensors(output):
        yield 'logits', output.logits
        for index, tensor in enumerate(output.hidden_states):
            yield f'hidden_{index}', tensor
        for index, state in enumerate(output.past_key_values):
            for name, tensor in zip(('k', 'v'), state['attn_state']):
                yield f'{name}_{index}', tensor

    try:
        for placement in ('ma_gpu', 'ma_offload'):
            model.close_memory_offload()
            if placement == 'ma_gpu':
                model.fold_memory_table_on_gpu()
            else:
                model.enable_memory_offload()
            cache = None
            for step, ids in enumerate((prefix, *tokens)):
                out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                            logits_to_keep=1, output_hidden_states=True)
                cache = out.past_key_values
                assert cache.get_seq_length() == 2048 + step
                if placement == 'ma_gpu':
                    reference.append({name: value.cpu().clone() for name, value in tensors(out)})
                else:
                    for name, value in tensors(out):
                        actual = value.cpu()
                        expected = reference[step].pop(name)
                        assert torch.isfinite(actual).all(), (step, name, 'nonfinite')
                        assert torch.equal(actual, expected), (step, name, 'not bit exact')
                        comparisons.append(dict(step=step, tensor=name, shape=list(actual.shape),
                                                exact=True, finite=True, max_abs_error=0))
                    reference[step].clear()
                print(f'{placement} step={step} cache_length={cache.get_seq_length()} OK', flush=True)
                del out
            del cache
            torch.cuda.empty_cache()
        payload = dict(status='passed', source=source_state(), config=model.config.to_dict(),
                       batch=8, prefix_length=2048, decode_steps=args.decode_steps,
                       comparison='bit exact all logits, hidden states and KV entries', comparisons=comparisons)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str)+'\n')
    finally:
        model.close_memory_offload()


if __name__ == '__main__':
    with torch.inference_mode():
        main()
