"""Audit raw results and summarize paired process differences, including OOM."""
import sys,json,csv,statistics,math
from pathlib import Path
from scipy.stats import t
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parents[1]/'continuation-03'))
import driver
p=json.loads((D/'manifest.json').read_text());assert p['status'].startswith('completed')
data={};envs=set()
for j in p['jobs']:
 if j['status']=='oom':
  raw=json.loads((D/(j['name']+'.json')).read_text());assert raw['status']=='oom';assert raw['source']['source_sha256']==p['source']['source_sha256'];assert raw['source']['git_commit']['stdout'].strip()==p['source']['commit']
 if j['status']=='completed':
  env,r=driver.audit_job(j,D/(j['name']+'.json'));envs.add(env);data[j['mode'],j['batch'],j['block'],j['variant']]=r
assert len(envs)==1
rows=[]
for mode in ('prefill','decode'):
 for b in (32,64):
  row=dict(mode=mode,batch=b,length=2048)
  values={v:[data[mode,b,k,v]['mean_ms'] for k in range(1,5) if (mode,b,k,v) in data] for v in ('ma_gpu','ma_offload')}
  for v in values:
   row[v+'_status']='completed' if len(values[v])==4 else 'oom' if any(j['mode']==mode and j['batch']==b and j['variant']==v and j['status']=='oom' for j in p['jobs']) else 'incomplete'
   row[v+'_ms']=statistics.mean(values[v]) if values[v] else None
   mem=[data[mode,b,k,v]['memory_after'] for k in range(1,5) if (mode,b,k,v) in data]
   row[v+'_memory']=mem
   row[v+'_peak_allocated_gib']=statistics.mean(x['gpu_peak_allocated_bytes']/2**30 for x in mem) if mem else None
  gaps=[data[mode,b,k,'ma_offload']['mean_ms']-data[mode,b,k,'ma_gpu']['mean_ms'] for k in range(1,5) if all((mode,b,k,v) in data for v in values)]
  row['paired_blocks']=len(gaps);row['paired_gaps_ms']=gaps
  if len(gaps)==4:
   mean=statistics.mean(gaps);half=float(t.ppf(.975,3))*statistics.stdev(gaps)/2
   row.update(gap_ms=mean,gap_lower_95_ms=mean-half,gap_upper_95_ms=mean+half,overhead_pct=statistics.mean([(o/g-1)*100 for o,g in zip(values['ma_offload'],values['ma_gpu'])]))
  rows.append(row)
(D/'summary.json').write_text(json.dumps(dict(source=p['source'],environment_match=True,note='Four independent paired process blocks; two-sided Student-t 95% intervals, df=3. Supplemental measurements, not final matrix acceptance. OOM values are missing, never zero.',rows=rows),indent=2)+'\n')
lines=['# A0030 大 batch 补测','', '固定 A0030；24 层、2.8365B BF16，RTX 5090，长度/context 2048。Prefill 包含 KV 构建与 last-token logits；decode 为已构建 KV 后的单步。使用随机权重与固定输入，不评估输出质量。','', '每点四组独立进程配对，GPU/offload 顺序交替。每进程 10 warmup、10 samples × 3 rounds。差值区间为配对进程均值的双侧 95% Student-t 区间（df=3）；这不是最终 16 场景验收。','', '| 场景 | batch | GPU ms | offload ms | 差值 ms（95% CI） | 相对开销 |','| --- | ---: | ---: | ---: | --- | ---: |']
for r in rows:
 def val(v):return f"{r[v+'_ms']:.3f}" if r[v+'_ms'] is not None else r[v+'_status'].upper()
 gap=f"{r['gap_ms']:+.3f} [{r['gap_lower_95_ms']:+.3f}, {r['gap_upper_95_ms']:+.3f}]" if 'gap_ms' in r else '无法配对'
 pct=f"{r['overhead_pct']:+.2f}%" if 'overhead_pct' in r else '—'
 lines.append(f"| {r['mode']} | {r['batch']} | {val('ma_gpu')} | {val('ma_offload')} | {gap} | {pct} |")
lines+=['','## 显存峰值','','GPU allocated 峰值的进程均值（GiB）；不等同于 nvidia-smi 或 reserved 显存。','','| 场景 | batch | GPU GiB | offload GiB |','| --- | ---: | ---: | ---: |']
for r in rows:
 def memval(v):return f"{r[v+'_peak_allocated_gib']:.3f}" if r[v+'_peak_allocated_gib'] is not None else r[v+'_status'].upper()
 lines.append(f"| {r['mode']} | {r['batch']} | {memval('ma_gpu')} | {memval('ma_offload')} |")
lines+=['','OOM 保留原始错误；同一配置的后续重复跳过。未改变长度、精度或模型以规避 OOM。当前设备另有约 654 MiB 常驻进程，其状态记录于各次原始 telemetry。','', '[计划与执行记录](manifest.json) · [完整统计与内存数据](summary.json)','', '复现：使用 fla-bench Python 执行本目录 `summarize.py`；测量命令逐条保存在 manifest 中。']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
