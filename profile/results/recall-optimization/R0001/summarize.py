"""Audit balanced fresh controls and report fused recall versus both baselines."""
import csv,hashlib,json,math,statistics
from pathlib import Path
D=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
p=read(D/'manifest.json');assert p['status']=='completed'
for name,digest in p['harness_sha256'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest
raws={j['name']:read(D/(j['name']+'.json')) for j in p['jobs'] if j['status']=='completed'}
envs=[]
for j in p['jobs']:
 if j['status']!='completed':continue
 raw=raws[j['name']];src=p['sources'][j['placement']]
 assert raw['source']['git_commit']['stdout'].strip()==src['commit'] and not raw['source']['git_status']['stdout'].strip()
 assert raw['model_config']['rope_theta'] is None and raw['model_config']['memory_recall']==src['recall']
 if j['kind']=='correctness':
  for point in raw['checkpoints']:
   tensors=[point['logits']]+point.get('hidden_states',[])+[t for pair in point.get('kv_states',[]) for t in pair]
   assert all(t['finite'] for t in tensors)
  continue
 cfg=raw['config'];r=raw['results'][0]
 for k,v in dict(mode=j['mode'],batch_size=j['batch'],seq_len=j['length'],context_len=j['length'],variants=['ma_gpu_unfolded'],hidden_size=2048,num_layers=24,num_heads=32,num_kv_heads=32,intermediate_size=5632,vocab_size=32000,seed=1234,logits_to_keep=1,generation_steps=128,rounds=3,memory_recall=src['recall']).items():assert cfg[k]==v,(j['name'],k)
 assert cfg['warmup']==(2 if j['mode']=='generation' else 10) and cfg['repeats']==(5 if j['mode']=='generation' else 10)
 assert len(r['samples_ms'])==3 and all(len(x)==cfg['repeats'] for x in r['samples_ms'])
 vals=[x for a in r['samples_ms'] for x in a];assert all(math.isfinite(x) and x>0 for x in vals)
 assert abs(statistics.mean(vals)-r['mean_ms'])<1e-8
 e=raw['environment_before'];v=raw['env'];envs.append([v[k] for k in ('torch','torch_cuda','flash_attn','gpu','python')]+[e[k] for k in ('cpu_affinity','torch_threads','torch_interop_threads','thread_environment')])
assert all(x==envs[0] for x in envs)
def stats(vals):
 avg=statistics.mean(vals);half=4.302652729911275*statistics.stdev(vals)/math.sqrt(3)
 return dict(mean=avg,lower_95=avg-half,upper_95=avg+half,block_values=vals)
rows=[]
keys=list(dict.fromkeys((j['mode'],j['batch'],j['length']) for j in p['jobs'] if j['kind']=='performance'))
for mode,b,l in keys:
 orders=[[j['placement'] for j in p['jobs'] if j['kind']=='performance' and (j['mode'],j['batch'],j['length'],j['block'])==(mode,b,l,block)] for block in (1,2,3)]
 for label in p['sources']:assert sorted(order.index(label) for order in orders)==[0,1,2]
 row=dict(mode=mode,batch=b,length=l,placements={},order_by_block=orders)
 for label in p['sources']:
  jobs=[j for j in p['jobs'] if j['kind']=='performance' and (j['mode'],j['batch'],j['length'],j['placement'])==(mode,b,l,label)]
  rr=[raws[j['name']]['results'][0] for j in jobs if j['status']=='completed']
  if rr:
   assert len(rr)==3
   means=[r['mean_ms'] for r in rr]
   row['placements'][label]=dict(status='completed',mean_ms=statistics.mean(means),sd_ms=statistics.stdev(means),block_means_ms=means,cache_gib=statistics.mean(r['memory_after']['kv_cache_storage_bytes']/2**30 for r in rr),peak_gpu_gib=statistics.mean(r['memory_after']['gpu_peak_allocated_bytes']/2**30 for r in rr))
  else:
   assert any(j['status']=='oom' for j in jobs);row['placements'][label]=dict(status='oom')
 for target,base in [('initial_recall','full_kv'),('fused_recall','initial_recall'),('fused_recall','full_kv')]:
  a,c=row['placements'][base],row['placements'][target]
  if a['status']==c['status']=='completed':
   row[target+'_minus_'+base+'_ms']=stats([y-x for x,y in zip(a['block_means_ms'],c['block_means_ms'],strict=True)])
   row[target+'_vs_'+base+'_pct']=stats([(y/x-1)*100 for x,y in zip(a['block_means_ms'],c['block_means_ms'],strict=True)])
   expected_ratio=1 if base=='initial_recall' else 0.5+2/2048
   assert abs(c['cache_gib']/a['cache_gib']-expected_ratio)<1e-6
 rows.append(row)
summary=dict(status='audited',sources=p['sources'],environment_match=True,environment=envs[0],correctness=read(D/'correctness-audit.json'),statistics=p['statistics'],balanced_order=True,rows=rows)
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
flat=[]
for r in rows:
 f={k:r[k] for k in ('mode','batch','length')}
 for label,v in r['placements'].items():
  for k in ('status','mean_ms','sd_ms','cache_gib','peak_gpu_gib'):f[label+'_'+k]=v.get(k)
 for base in ('initial_recall','full_kv'):
  for unit in ('ms','pct'):
   k='fused_recall_'+('minus' if unit=='ms' else 'vs')+'_'+base+'_'+unit
   for stat in ('mean','lower_95','upper_95'):f[k+'_'+stat]=r.get(k,{}).get(stat)
 flat.append(f)
with (D/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
lines=['# R0001: fused memory recall','','Initial baseline: `'+p['sources']['initial_recall']['commit']+'`. Candidate: `'+p['sources']['fused_recall']['commit']+'`. Candidate fuses historical GPU embedding lookup, per-head RMSNorm and K+M into one Triton kernel during inference decode. It preserves BF16/FP16 normalization rounding, stores no historical V cache and creates no full-size M intermediates. Prefill, training and incompatible layouts/hooks use the existing path. No table normalization folding, CPU offload or RoPE is used.','','## Fresh three-way latency comparison','','Milliseconds, lower is better. Negative change means faster. All three paths were freshly measured in the same campaign, with matching weights and no-RoPE configuration. Three independent process blocks rotate the three placements through every ordering position for every workload. Paired two-sided 95% Student-t intervals (df2) are in summary.json/CSV. They describe these measurements, not a universal bound.','','| Mode | Batch | Length/context | Full KV | Initial recall | Fused recall | Fused vs initial | Fused vs full KV |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 vals=[f"{r['placements'][label]['mean_ms']:.3f}" if r['placements'][label]['status']=='completed' else 'OOM' for label in p['sources']]
 vals += [f"{r['fused_recall_vs_'+base+'_pct']['mean']:+.2f}%" if 'fused_recall_vs_'+base+'_pct' in r else '—' for base in ('initial_recall','full_kv')]
 lines.append(f"| {r['mode']} | {r['batch']} | {r['length']} | "+' | '.join(vals)+' |')
lines+=['','## Peak allocated GPU memory','','GiB. These peaks exclude other processes, CUDA context, reserved-but-unused allocator storage and setup. Both recall modes retain the same raw K + int64 IDs, with 49.9023% less persistent cache than full KV when both run. They materialize V transiently.','','| Mode | Batch | Length/context | Full KV peak | Initial recall peak | Fused recall peak |','|---|---:|---:|---:|---:|---:|']
for r in rows:
 vals=[f"{r['placements'][label]['peak_gpu_gib']:.3f}" if r['placements'][label]['status']=='completed' else 'OOM' for label in p['sources']]
 lines.append(f"| {r['mode']} | {r['batch']} | {r['length']} | "+' | '.join(vals)+' |')
lines+=['','## Protocol and evidence','','RTX5090, 24 layers, hidden2048, 32 Q/KV heads, intermediate5632, vocab32000, 2.8365B BF16, QK norm/gates disabled, seed1234. No RoPE, GPU-resident unfolded memory tables throughout. Prefill/decode:10 warmups,10 samples ×3 rounds. Generation:2 warmups,5 trajectories ×3 rounds; prefix2048 +128 growing-cache decode steps, predetermined tokens, no sampling. Explicit calls bypass the existing Transformers generate compatibility issue. Prefill constructs cache and last-token logits; decode uses fixed context with rollback/token selection outside timing. Model cache updates and V reconstruction remain timed. No optimized static KV cache.','','82 CUDA regression cases pass, including exact reconstruction for BF16/FP16 and head dimensions40/64/128, folded/unfolded tables and hook fallback. Six full-model correctness processes validate exact logits at129 steps for batch1/8 and all hidden/raw-K states at prefix/steps1,2,128; initial/fused recall fingerprints match exactly, including IDs. Training and unsupported paths keep PyTorch autograd. No trained-checkpoint quality claim.','','An unrelated GPU process holds about6994 MiB, versus about654 MiB during the original baseline collection. It was not stopped. We remeasured all controls; do not subtract old timings as though they were paired, or interpret a new OOM as a capacity regression without accounting for external allocation. Identical OOM repeats are skipped, never recorded as zero.','','Artifacts: manifest.json (all commands/statuses), correctness-audit.json, harness-audit.json, regression-tests.txt, summary.json/CSV and raw B*.json/TXT. Original baseline and microbench diagnostics are in the parent directory. Microbench stage latencies are diagnostic and not additive to model latency.']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n')
files=[f for f in D.iterdir() if f.suffix in ('.py','.json','.txt','.md','.csv') and f.name!='integrity-audit.json']
(D/'integrity-audit.json').write_text(json.dumps(dict(status='passed',elapsed_minutes=(p['finished_unix']-p['started_unix'])/60,counts={s:sum(j['status']==s for j in p['jobs']) for s in ('completed','oom','not_run')},balanced_order=True,sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files}),indent=2)+'\n')
print('\n'.join(lines[:34]))
