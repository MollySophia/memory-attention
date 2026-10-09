"""Render a PR supplement from the audited direct-bmk comparison."""
import json, sys
from pathlib import Path
D=Path(__file__).resolve().parent
s=json.loads((D/'summary.json').read_text())
print('### Original standalone bmk offload comparison\n')
print('Main already contains CPU offload in the standalone `profile/bmk.py` benchmark. This PR integrates offload into `MemoryForCausalLM` and improves its scheduling; it does not introduce the repository’s first offload implementation.\n')
print('The table below directly reruns that **unmodified original bmk** at `66bd7c6`, with matching model dimensions, BF16, last-token logits, 16 CPU threads and 10 warmups / 10 repeats × 3 rounds. Original policies remain prefill pipeline / decode bulk. Each latency is the mean of three separate process means; PR values come from the fresh study above, not a new paired run. Negative change means the PR is faster.\n')
print('| Workload | Batch | Length/context | Original bmk offload, ms | PR offload, ms | PR change |')
print('| --- | ---: | ---: | ---: | ---: | ---: |')
for r in s['rows']:
 a=f"{r['bmk_ms']:.3f}" if r['bmk_status']=='completed' else 'OOM'
 b=f"{r['pr_ms']:.3f}" if r['pr_status']=='completed' else 'OOM'
 c=f"{r['pr_vs_bmk_pct']:+.2f}%" if 'pr_vs_bmk_pct' in r else '—'
 print(f"| {r['mode']} | {r['batch']} | {r['length']} | {a} | {b} | {c} |")
print('\nThis compares complete implementations, **not isolated offloader performance**. Original bmk uses a preallocated static KV cache and has both CPU/GPU IDs ready before timing; the formal model uses a dynamic KV cache and stages IDs as needed. Synthetic weight/input initialization and timing-loop details also differ. The original bmk has no generation mode. In particular, the formal model’s larger-batch/long-context decode is materially slower than original bmk even though its offload is close to its own GPU control. Small changes are descriptive, not demonstrated significance.\n')
base=f'https://github.com/MollySophia/memory-attention/blob/{sys.argv[1]}/profile/results/pr-offload-bmk-001'
print(f'[Full comparison with between-process variation]({base}/REPORT.md) · [Raw commands and run index]({base}/manifest.json) · [CSV]({base}/summary.csv)\n')
