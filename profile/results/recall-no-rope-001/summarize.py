"""Audit paired no-RoPE recall timings and cache/peak-memory measurements."""
import csv, hashlib, json, math, statistics
from pathlib import Path
D=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
p=read(D/'manifest.json');assert p['status']=='completed'
for name, expected in p['harness_sha256'].items():
 assert hashlib.sha256((D/name).read_bytes()).hexdigest()==expected,name
raws={j['name']:read(D/(j['name']+'.json')) for j in p['jobs'] if j['status']=='completed'}
envs=[]
for j in p['jobs']:
 if j['status']!='completed':continue
 raw=raws[j['name']];assert raw['source']['git_commit']['stdout'].strip()==p['source_commit']
 assert raw['model_config']['rope_theta'] is None and raw['model_config']['memory_recall']==j['recall']
 if j['kind']!='performance':
  for point in raw['checkpoints']:
   tensors=[point['logits']]+point.get('hidden_states',[])+[t for pair in point.get('kv_states',[]) for t in pair]
   assert all(t['finite'] for t in tensors)
  continue
 cfg=raw['config'];r=raw['results'][0]
 for k,v in dict(mode=j['mode'],batch_size=j['batch'],seq_len=j['length'],context_len=j['length'],variants=['ma_gpu_unfolded'],hidden_size=2048,num_layers=24,num_heads=32,num_kv_heads=32,intermediate_size=5632,vocab_size=32000,seed=1234,logits_to_keep=1,generation_steps=128,rounds=3,memory_recall=j['recall']).items():assert cfg[k]==v,(j['name'],k)
 assert cfg['warmup']==(2 if j['mode']=='generation' else 10) and cfg['repeats']==(5 if j['mode']=='generation' else 10)
 assert len(r['samples_ms'])==3 and all(len(x)==cfg['repeats'] for x in r['samples_ms'])
 vals=[x for a in r['samples_ms'] for x in a];assert all(math.isfinite(x) and x>0 for x in vals)
 assert abs(statistics.mean(vals)-r['mean_ms'])<1e-8
 e=raw['environment_before'];v=raw['env'];envs.append([v[k] for k in ('torch','torch_cuda','flash_attn','gpu','python')]+[e[k] for k in ('cpu_affinity','torch_threads','torch_interop_threads','thread_environment')])
assert all(x==envs[0] for x in envs)
def paired(vals):
 avg=statistics.mean(vals);half=4.302652729911275*statistics.stdev(vals)/math.sqrt(3)
 return dict(mean=avg,lower_95=avg-half,upper_95=avg+half,block_values=vals)
rows=[]
keys=list(dict.fromkeys((j['mode'],j['batch'],j['length']) for j in p['jobs'] if j['kind']=='performance'))
for mode,b,l in keys:
 row=dict(mode=mode,batch=b,length=l,placements={},placement_order_by_block=[[('recall' if j['recall'] else 'full_kv') for j in p['jobs'] if j['kind']=='performance' and (j['mode'],j['batch'],j['length'],j['block'])==(mode,b,l,block)] for block in (1,2,3)])
 for recall,label in [(False,'full_kv'),(True,'recall')]:
  jobs=[j for j in p['jobs'] if j['kind']=='performance' and (j['mode'],j['batch'],j['length'],j['recall'])==(mode,b,l,recall)]
  rr=[raws[j['name']]['results'][0] for j in jobs if j['status']=='completed']
  if rr:
   assert len(rr)==3
   means=[r['mean_ms'] for r in rr]
   row['placements'][label]=dict(status='completed',mean_ms=statistics.mean(means),sd_ms=statistics.stdev(means),block_means_ms=means,cache_gib=statistics.mean(r['memory_after']['kv_cache_storage_bytes']/2**30 for r in rr),peak_gpu_gib=statistics.mean(r['memory_after']['gpu_peak_allocated_bytes']/2**30 for r in rr),allocated_gpu_gib=statistics.mean(r['memory_after']['gpu_allocated_bytes']/2**30 for r in rr))
  else:
   assert any(j['status']=='oom' for j in jobs)
   row['placements'][label]=dict(status='oom')
 a,c=row['placements']['full_kv'],row['placements']['recall']
 if a['status']==c['status']=='completed':
  row['recall_minus_full_kv_ms']=paired([y-x for x,y in zip(a['block_means_ms'],c['block_means_ms'],strict=True)])
  row['recall_vs_full_kv_pct']=paired([(y/x-1)*100 for x,y in zip(a['block_means_ms'],c['block_means_ms'],strict=True)])
  row['cache_saved_pct']=(1-c['cache_gib']/a['cache_gib'])*100
  assert abs(c['cache_gib']/a['cache_gib']-(0.5+2/2048))<1e-6
 rows.append(row)
summary=dict(status='audited',source_commit=p['source_commit'],environment_match=True,environment=envs[0],correctness=read(D/'correctness-audit.json'),statistics='Paired process-block means, n=3, two-sided 95% Student-t df=2. Per-shape placement order is fixed across blocks; order bias is not eliminated.',rows=rows)
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
flat=[]
for r in rows:
 f={k:r[k] for k in ('mode','batch','length')}
 for label,v in r['placements'].items():
  for k in ('status','mean_ms','sd_ms','cache_gib','peak_gpu_gib','allocated_gpu_gib'):f[label+'_'+k]=v.get(k)
 for k in ('recall_minus_full_kv_ms','recall_vs_full_kv_pct'):
  for stat in ('mean','lower_95','upper_95'):f[k+'_'+stat]=r.get(k,{}).get(stat)
 flat.append(f)
with (D/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
lines=['# No-RoPE recall versus full KV caching','','Both paths run frozen source `'+p['source_commit']+'`, with identical seeded weights. Only `memory_recall` changes. RoPE is disabled in both. Tables stay on GPU and are not normalization-folded. This compares the first recall implementation against the corresponding no-RoPE model, not against the prior RoPE/offload study.','','## Latency','','Milliseconds; lower is better. Positive change means recall is slower. Means use three independent process blocks. Paired two-sided 95% Student-t intervals use those three block means (df=2), not individual samples as independent replicates. Neighboring workloads alternate placement order and block2 reverses the workload list. Audit shows each individual shape retains the same placement order across blocks; order bias is therefore not eliminated. Intervals describe block-to-block variation, not that systematic bias. Intervals crossing zero are inconclusive.','','| Mode | Batch | Length/context | Full KV, ms | Recall, ms | Recall change | Recall − full KV, ms [95% CI] |','|---|---:|---:|---:|---:|---:|---|']
for r in rows:
 a,c=r['placements']['full_kv'],r['placements']['recall']
 av=f"{a['mean_ms']:.3f}" if a['status']=='completed' else 'OOM'
 cv=f"{c['mean_ms']:.3f}" if c['status']=='completed' else 'OOM'
 if 'recall_minus_full_kv_ms' in r:
  delta=r['recall_minus_full_kv_ms'];pct=f"{r['recall_vs_full_kv_pct']['mean']:+.2f}%";gap=f"{delta['mean']:+.3f} [{delta['lower_95']:+.3f}, {delta['upper_95']:+.3f}]"
 else:pct=gap='—'
 lines.append(f"| {r['mode']} | {r['batch']} | {r['length']} | {av} | {cv} | {pct} | {gap} |")
lines+=['','## Memory','','GiB (2^30 bytes). Cache counts retained tensor storage, including token IDs. Peak GPU is measured allocated memory during the timed workload, not reserved memory or nvidia-smi usage. Persistent cache savings do not imply the same reduction in peak GPU usage.','','| Mode | Batch | Length/context | Full KV cache | Recall cache | Full KV GPU peak | Recall GPU peak |','|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
 a,c=r['placements']['full_kv'],r['placements']['recall']
 vals=[f'{v[k]:.3f}' if v['status']=='completed' else 'OOM' for v,k in [(a,'cache_gib'),(c,'cache_gib'),(a,'peak_gpu_gib'),(c,'peak_gpu_gib')]]
 lines.append(f"| {r['mode']} | {r['batch']} | {r['length']} | "+' | '.join(vals)+' |')
lines+=['','## Scope and validation','','RTX 5090; 24 layers, hidden 2048, 32 Q/KV heads, intermediate 5632, vocabulary 32000, 2.8365B BF16; QK norm and gates off; seed1234. Performance prefill/decode: 10 warmups, 10 samples × 3 rounds per process. Generation: 2 warmups, 5 trajectories × 3 rounds; prefix2048 +128 predetermined growing-cache decode steps, no sampling. Each process constructs fresh model weights. Construction and setup are excluded. Inputs start on GPU.','','Prefill includes cache creation and last-token logits. Decode measures one fixed-context step with rollback/token selection excluded; the model’s cache update, historical table lookup, RMSNorm and V reconstruction remain timed. Generation uses explicit model calls, not the currently incompatible Transformers generate wrapper.','','Full-model correctness passed at batch1/8: bit-exact logits at all129 prefix/decode steps; bit-exact hidden states and cached raw keys at prefix/steps1,2,128; recall stores int64 IDs in place of V. 69 smaller recall/offload regression tests passed before measurement. No trained-checkpoint language quality claim.','','Recall retains raw K and per-layer token IDs, then materializes historical M and V each step. At key width2048 and BF16, persistent cache drops from4D to2D+8 bytes/token/layer: 49.9023% saved. QK norm is disabled here; enabling it would also require renormalizing recalled keys. No CPU offload or fused tiled reconstruction is used.','','OOMs are preserved; only identical repeats are skipped. See manifest.json for every command/status, correctness-audit.json, harness-audit.json for unchanged timing functions, summary.json/CSV for numerical results, and B*.json/TXT for raw measurements.']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n')
files=[f for f in D.iterdir() if f.suffix in ('.py','.json','.txt','.md','.csv') and f.name!='integrity-audit.json']
(D/'integrity-audit.json').write_text(json.dumps(dict(status='passed',elapsed_minutes=(p['finished_unix']-p['started_unix'])/60,counts={s:sum(j['status']==s for j in p['jobs']) for s in ('completed','oom','not_run')},sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}),indent=2)+'\n')
print('\n'.join(lines[:35]))
