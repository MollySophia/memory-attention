"""Describe two separate benchmark studies without treating them as paired."""
import csv, hashlib, json, statistics
from pathlib import Path
D=Path(__file__).resolve().parent
p=json.loads((D/'manifest.json').read_text())
assert p['status']=='completed'
ref=json.loads((D.parent/'pr-offload-001/summary.json').read_text())
rows=[]
for old in ref['rows']:
 if old['mode']=='generation':continue
 jobs=[j for j in p['jobs'] if (j['mode'],j['batch'],j['length'])==(old['mode'],old['batch'],old['length'])]
 means=[]
 for j in jobs:
  if j['status']=='completed':
   raw=json.loads((D/(j['name']+'.json')).read_text());cfg=raw['config'];r=raw['results'][0]
   assert cfg['cpu_threads']==16 and cfg['norm_eps']==1e-6 and cfg['logits_to_keep']==1
   assert r['total_parameters']==2836498432
   means.append(statistics.mean(r['round_ms']))
 new=old['placements']['head_offload']
 row=dict(mode=old['mode'],batch=old['batch'],length=old['length'],bmk_status='completed' if means else 'oom',bmk_block_means_ms=means,pr_status=new['status'])
 if means:
  assert len(means)==3
  row.update(bmk_ms=statistics.mean(means),bmk_sd_ms=statistics.stdev(means))
 if new['status']=='completed':row.update(pr_ms=new['mean_ms'],pr_sd_ms=statistics.stdev(new['block_means_ms']))
 if means and new['status']=='completed':row.update(pr_minus_bmk_ms=row['pr_ms']-row['bmk_ms'],pr_vs_bmk_pct=(row['pr_ms']/row['bmk_ms']-1)*100)
 rows.append(row)
summary=dict(status='audited',note='Original bmk and prior fresh PR benchmark are separate studies, not paired. Mean of three process means; SD across those three means. This is an end-to-end implementation comparison, not isolated offloader speedup.',rows=rows)
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (D/'summary.csv').open('w') as f:
 keys=['mode','batch','length','bmk_status','bmk_ms','bmk_sd_ms','pr_status','pr_ms','pr_sd_ms','pr_minus_bmk_ms','pr_vs_bmk_pct']
 w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
lines=['# Original main benchmark offload vs PR offload','','Direct execution of unmodified `main/profile/bmk.py` at `66bd7c6326a6a7badf01e7ef5a14ff967fb8000c`; PR reference is the completed `pr-offload-001` study at `9925991f02502409c0bc838ee0d8c6e70ba878bc`. No old offloader was transplanted into the PR model.','','## Reading the table','','Milliseconds, lower is better. Negative PR change means PR is faster. Each value is the mean of three independent process means; ± is the sample SD of those process means, not a confidence interval. The two studies were collected separately and are not paired. Small differences should not be read as confirmed improvements.','','| Mode | Batch | Length/context | Original bmk offload (ms ± SD) | PR offload (ms ± SD) | PR change |','|---|---:|---:|---:|---:|---:|']
for r in rows:
 a=f"{r['bmk_ms']:.3f} ± {r['bmk_sd_ms']:.3f}" if r['bmk_status']=='completed' else 'OOM'
 b=f"{r['pr_ms']:.3f} ± {r['pr_sd_ms']:.3f}" if r['pr_status']=='completed' else 'OOM'
 c=f"{r['pr_vs_bmk_pct']:+.2f}%" if 'pr_vs_bmk_pct' in r else '—'
 lines.append(f"| {r['mode']} | {r['batch']} | {r['length']} | {a} | {b} | {c} |")
lines+=['','## Aligned settings','','RTX 5090, BF16, 24 layers, hidden size 2048, 32 Q/KV heads, intermediate size 5632, vocabulary 32000, untied embedding/head, RMSNorm epsilon 1e-6, RoPE theta 10000, no QK normalization or gate, fused SwiGLU, last-token logits, seed 1234, 16 CPU threads, group size 1, prefetch depth 4. Per process: 10 warmups, 3 rounds of 10 repeats, synchronized wall-clock latency. Original bmk policy remains prefill pipeline / decode bulk. All 18 prefill/decode shapes are planned; identical OOM repeats are skipped.','','## Preserved differences','','- Original bmk runs its own `AttentionStack`, preallocates a static KV cache and overwrites slots. PR uses `MemoryForCausalLM` and its dynamic KV cache. Prefill in PR includes fresh cache creation; decode includes the model’s cache update.','- Original bmk has both CPU and GPU IDs available before timing and cycles a 16-input pool. PR starts with GPU IDs and includes any needed staging; its prefill repeats one input.','- Parameter accounting differs by 1,536 folded memory-normalization affine values (24 × 64): original bmk reports 2,836,498,432; PR reports 2,836,499,968. This is not a hidden-size/layer-count mismatch.', '- Model dimensions match, but synthetic weight and input initialization differ. Outputs are not asserted equal across these two independent model implementations.','- Original bmk times each whole round with synchronization after every call; PR records each call separately. Both include embedding, transformer blocks, final normalization and last-token logits.','- Original bmk has no generation mode. The earlier PR generation results have no original-bmk counterpart.','- Original bmk reports parameter/staging sizes, not a comparable measured GPU allocation peak. No peak-memory comparison is inferred here.','- Results describe the complete implementations and must not be attributed solely to offload optimization.','','## Evidence','','`manifest.json` records exact commands, frozen-source hashes, job states and hardware. Raw JSON/TXT files preserve each process. `summary.json` and `summary.csv` contain the numerical comparison. PR raw data remain in `../pr-offload-001/`.']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n')
files=[f for f in D.iterdir() if f.suffix in ('.json','.txt','.py','.md','.csv') and f.name!='integrity-audit.json']
(D/'integrity-audit.json').write_text(json.dumps(dict(status='passed',counts={s:sum(j['status']==s for j in p['jobs']) for s in ('completed','oom','not_run')},elapsed_minutes=(p['finished_unix']-p['started_unix'])/60,sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}),indent=2)+'\n')
print('\n'.join(lines[:28]))
