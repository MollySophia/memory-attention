"""Audit and summarize frozen main/PR comparisons from paired process blocks."""
import csv,json,math,statistics,sys
from pathlib import Path
from scipy.stats import t
from run import D,read,save,audit,compare_correctness,WORKLOADS,PLACEMENTS,SOURCES
p=read(D/'manifest.json');assert p['status']=='completed';compare_correctness()
data={};environments=set();oom=[]
for j in p['jobs']:
 if j['kind']!='performance':continue
 if j['status']=='completed':
  raw=read(D/(j['name']+'.json'));audit(j,raw)
  r=raw['results'][0];data[j['mode'],j['batch'],j['length'],j['block'],j['source'],j['variant']]=r
  env=raw['env'];e=raw['environment_before'];environments.add((env['torch'],env['torch_cuda'],env['flash_attn'],env['gpu'],env['python'],tuple(e['cpu_affinity']),e['torch_threads'],e['torch_interop_threads'],json.dumps(e['thread_environment'],sort_keys=True),e['gpu_telemetry']['stdout'].splitlines()[1].split(', ')[1]))
 elif j['status']=='oom':
  raw=read(D/(j['name']+'.json'));assert raw['status']=='oom';assert raw['source']['git_commit']['stdout'].strip()==SOURCES[j['source']][1];oom.append(j['name'])
 else:assert j['status']=='not_run' and j['reason']=='Prior identical cell OOM'
assert len(environments)==1
rows=[]
def ci(values):
 n=len(values);mean=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(n);half=float(t.ppf(.975,n-1))*se
 return dict(mean=mean,lower_95=mean-half,upper_95=mean+half,values=values,n=n)
labels=['main_gpu','head_gpu','head_offload']
for mode,b,l in WORKLOADS:
 row=dict(mode=mode,batch=b,length=l,original_matrix=b not in (32,64),placements={})
 for label,(s,v) in zip(labels,PLACEMENTS):
  parts=[data[mode,b,l,k,s,v] for k in range(1,4) if (mode,b,l,k,s,v) in data]
  if len(parts)!=3:
   assert not parts,'Incomplete successful cells require review'
   row['placements'][label]=dict(status='oom',mean_ms=None);continue
  row['placements'][label]=dict(status='completed',mean_ms=statistics.mean(x['mean_ms'] for x in parts),block_means_ms=[x['mean_ms'] for x in parts],gpu_peak_allocated_gib=statistics.mean(x['memory_after']['gpu_peak_allocated_bytes']/2**30 for x in parts),host_rss_gib=statistics.mean(x['memory_after']['host_rss_bytes']/2**30 for x in parts),host_pinned_gib=statistics.mean(x['memory_after']['offload_pinned_bytes']/2**30 for x in parts))
 for ref,label in [('main_gpu','main'),('head_gpu','folded_gpu')]:
  if row['placements'][ref]['status']=='completed' and row['placements']['head_offload']['status']=='completed':
   baseline=row['placements'][ref]['block_means_ms'];off=row['placements']['head_offload']['block_means_ms']
   row['offload_minus_'+label+'_ms']=ci([o-g for g,o in zip(baseline,off)])
   row['offload_vs_'+label+'_pct']=ci([(o/g-1)*100 for g,o in zip(baseline,off)])
 if row['placements']['main_gpu']['status']==row['placements']['head_gpu']['status']=='completed':
  main=row['placements']['main_gpu']['block_means_ms'];gpu=row['placements']['head_gpu']['block_means_ms']
  row['folded_gpu_minus_main_ms']=ci([g-m for m,g in zip(main,gpu)])
 if 'offload_minus_folded_gpu_ms' in row:
  gpu=row['placements']['head_gpu']['block_means_ms'];off=row['placements']['head_offload']['block_means_ms']
  residual=[o-g-max(.1,.01*g) for g,o in zip(gpu,off)];v=ci(residual);v['upper_simultaneous_95']=v['mean']+float(t.ppf(1-.05/16,2))*statistics.stdev(residual)/math.sqrt(3);row['original_matrix_tolerance_diagnostic']=v if row['original_matrix'] else None
 rows.append(row)
out=dict(sources=p['sources'],status='audited',note='All data freshly collected on frozen source commits. Paired 95% Student-t intervals use n=3 independent process blocks, df=2, not 90 independent samples. GPU control folds normalization; main leaves normalization per token. Supplemental OOM does not become zero latency. Tolerance calculation is diagnostic, not a final goal verdict.',environment_match=True,environment_fingerprint=list(next(iter(environments))),correctness=read(D/'correctness-audit.json'),oom=oom,rows=rows)
save(D/'summary.json',out)
flat=[]
for r in rows:
 f={k:r[k] for k in ('mode','batch','length','original_matrix')}
 for label in labels:
  for key,val in r['placements'][label].items():
   if key!='block_means_ms':f[label+'_'+key]=val
 for key in ('offload_minus_main_ms','offload_minus_folded_gpu_ms','offload_vs_main_pct','offload_vs_folded_gpu_pct'):
  for stat in ('mean','lower_95','upper_95'):f[key+'_'+stat]=r.get(key,{}).get(stat)
 flat.append(f)
keys=list(dict.fromkeys(k for r in flat for k in r))
with (D/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(flat)
lines=['# Fresh upstream / PR performance comparison','',f"Upstream main: `{SOURCES['main'][1]}`. PR head: `{SOURCES['head'][1]}`.",'', 'RTX 5090, 24 layers, 2.8365B BF16, vocab 32000, hidden 2048, 32 Q/KV heads, intermediate 5632. Fixed random weights (seed1234) and inputs; no language-quality claim. Inputs start on GPU. Last-token logits and real KV cache. Decode is one fixed-context step; generation includes prefix2048 and all128 growing-cache decode steps, excluding sampling. Setup, folding and allocation excluded.','', 'Three independent blocks rotate the three placement orders. Prefill/decode: 10 warmups, 10 samples ×3 rounds/process. Generation: 2 warmups, 5 trajectories ×3 rounds/process. Intervals use paired process means (t, df=2). Timed functions are AST-identical to the prior harness; only fresh-model/source compatibility and untimed accounting were adapted.','', '## Latency','', '| Mode | Batch | Context | Main GPU ms | PR folded GPU ms | PR offload ms | Offload−main ms | Offload−folded ms (95% CI) |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
for r in rows:
 vals=[f"{r['placements'][label]['mean_ms']:.3f}" if r['placements'][label]['status']=='completed' else 'OOM' for label in labels]
 before=f"{r['offload_minus_main_ms']['mean']:+.3f}" if 'offload_minus_main_ms' in r else 'unavailable'
 if 'offload_minus_folded_gpu_ms' in r:
  g=r['offload_minus_folded_gpu_ms'];gap=f"{g['mean']:+.3f} [{g['lower_95']:+.3f}, {g['upper_95']:+.3f}]"
 else:gap='unavailable' 
 lines.append(f"| {r['mode']} | {r['batch']} | {r['length']} | {' | '.join(vals)} | {before} | {gap} |")
lines+=['','## Peak allocated GPU memory','','| Mode | Batch | Context | Main GiB | PR folded GiB | PR offload GiB | Offload pinned host GiB |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in rows:
 vals=[f"{r['placements'][label]['gpu_peak_allocated_gib']:.3f}" if r['placements'][label]['status']=='completed' else 'OOM' for label in labels]
 off=r['placements']['head_offload'];pin=f"{off['host_pinned_gib']:.3f}" if off['status']=='completed' else 'OOM'
 lines.append(f"| {r['mode']} | {r['batch']} | {r['length']} | {' | '.join(vals)} | {pin} |")
lines+=['','Host storage also includes raw restoration weights and process overhead. GPU allocated peaks are not nvidia-smi usage. Every raw JSON includes before/after environment and memory telemetry. Another ~654 MiB GPU process was present at planning time and is retained in telemetry.','', '## Audit and interpretation','', 'Full-model main resident, PR folded resident and PR offload outputs match bit-for-bit at batch1 and8, prefix2048, seed1234: logits at all129 steps; all25 hidden states and24 layer KV tensors at prefix and steps1,2,128. Small-model regression suite separately covers49 cases.','', 'The main-to-offload comparison includes normalization folding as well as table placement. The folded-GPU comparison isolates the incremental offload path more closely. Negative differences retain their sign; intervals crossing zero are inconclusive, not equivalence. No workload is silently omitted and no OOM timing is zero.','', '[Plan and execution](manifest.json) · [Audited statistics](summary.json) · [CSV](summary.csv) · [Correctness audit](correctness-audit.json) · [Harness compatibility audit](harness-audit.json)','', 'Reproduce: use `/home/molly/miniconda3/envs/fla-bench/bin/python summarize.py` in this directory. Per-process exact commands, source roots, source SHAs and harness hashes are in the manifest.']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n');print('Audited rows:',len(rows),'OOM processes:',len(oom))
