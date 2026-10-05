"""CPU host-buffer working-set diagnostic; no model or overlap speedup claim."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path[:0] = [str(a.source_root), str(a.source_root / 'profile')]
import torch
from benchmark_telemetry import environment_details, source_state

with torch.inference_mode():
    torch.manual_seed(1234)
    table = torch.randn(32000, 24, 2048, dtype=torch.bfloat16).pin_memory()
    results = []
    for tokens in (4096, 8192, 16384):
        buffers = [torch.empty(tokens, 1, 2048, dtype=table.dtype, pin_memory=True)
                   for _ in range(4)]
        for buf in buffers:
            buf.zero_()
        samples = {depth: [] for depth in (1, 2, 4)}
        for trial in range(6):
            ids = torch.randint(0, 32000, (tokens,))
            order = (1, 2, 4)
            order = order[trial % 3:] + order[:trial % 3]
            for depth in order:
                start = time.perf_counter_ns()
                for layer in range(24):
                    torch.index_select(table[:, layer:layer + 1], 0, ids,
                                       out=buffers[layer % depth])
                elapsed = (time.perf_counter_ns() - start) / 1e6
                # Validate every surviving slot outside timing, with fresh IDs.
                for layer in range(24 - depth, 24):
                    expected = table[:, layer:layer + 1].index_select(0, ids)
                    torch.testing.assert_close(buffers[layer % depth], expected,
                                               rtol=0, atol=0)
                if trial >= 2:
                    samples[depth].append(elapsed)
        results.append(dict(tokens=tokens, samples_ms=samples,
                            mean_ms={k: sum(v) / len(v) for k, v in samples.items()},
                            active_host_bytes={d: d * buffers[0].numel() * 2
                                               for d in samples}))
    payload = dict(campaign_id='offload-gap-001', purpose=__doc__,
                   source=source_state(), environment=environment_details(),
                   command=sys.argv,
                   helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   warmup_trajectories=2, measured_trajectories=4,
                   layers_per_trajectory=24, seed=1234, results=results,
                   exactness='Fresh-ID surviving slots checked bit-exact after every trajectory',
                   limitations='No H2D or model compute. Isolates active output working set; cannot establish overlap, end-to-end gain or retention.')
    a.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(results))
