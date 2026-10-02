"""CPU lookup diagnostic at the actual table dimensions; never headline timing.

Measures identical index_select operations with strided versus contiguous rows.
Packing is setup work; both variants copy fresh token rows on every invocation.
Run only when model performance measurements have stopped.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source_root.resolve() / 'profile'))
import torch
from benchmark_telemetry import environment_details, source_state

torch.manual_seed(1234)
records = []
with torch.inference_mode():
    weights = torch.randn(32000, 24, 2048, dtype=torch.bfloat16)
    for tokens in (2048, 16384):
        ids = torch.randint(0, 32000, (tokens,))
        for group in (1, 2):
            # Representative layer group; source rows retain full-table stride.
            source = weights[:, 12:12 + group]
            packed = source.contiguous()
            output = torch.empty((tokens, group, 2048), dtype=weights.dtype, pin_memory=True)
            original = torch.index_select(source, 0, ids)
            torch.index_select(packed, 0, ids, out=output)
            assert torch.equal(original, output)
            samples = {'strided': [], 'contiguous': []}
            for iteration in range(40):
                order = [('strided', source), ('contiguous', packed)]
                if iteration % 2:
                    order.reverse()
                for label, table in order:
                    start = time.perf_counter_ns()
                    torch.index_select(table, 0, ids, out=output)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    if iteration >= 10:
                        samples[label].append(elapsed)
            records.append(dict(tokens=tokens, group=group, source_stride=list(source.stride()),
                                packed_stride=list(packed.stride()), samples_ms=samples,
                                mean_ms={k: statistics.mean(v) for k, v in samples.items()},
                                median_ms={k: statistics.median(v) for k, v in samples.items()},
                                exact=True))
            del packed, output, original
result = dict(campaign_id='offload-gap-001', diagnostic='CPU-only lookup layout microbenchmark',
              command=sys.argv, source=source_state(), environment=environment_details(),
              weights_shape=[32000, 24, 2048], dtype='bfloat16', records=records,
              limitations='Same-process diagnostic; synthetic random table, 10 warmups and 30 alternating samples per layout. Does not measure overlap or end-to-end gains. Packing time excluded, extra host storage required. No model optimization verdict.')
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
for record in records:
    print(record['tokens'], record['group'], record['mean_ms'], flush=True)
