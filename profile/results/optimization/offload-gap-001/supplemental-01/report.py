"""Regenerate survey tables/figures from audited raw screen and confirmation data."""
import csv,json,math,statistics
from pathlib import Path
import study as s
D=s.D

def export(rows,path):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def main():
 assert s.read(D/'workflow-controller.json')['status']=='measurements_complete_pending_report'
 import confirm
 confirm.analyze()
 screen=s.read(D/'screen-summary.json')['rows'];confirmation=s.read(D/'confirmation-summary.json');family=confirmation['family']
 out=D/'final';out.mkdir(exist_ok=True)
 sr=[]
 for r in screen:
  row={k:r[k] for k in ('attempt','baseline','mode','batch','length','screen_promising')}
  row.update(offload_reduction_ms=r['statistics']['offload_reduction_ms']['mean'],gap_reduction_ms=r['statistics']['gap_reduction_ms']['mean'],source=r.get('source_manifest',json.dumps(r.get('source_results'))))
  if r.get('latencies'):
   row.update(r['latencies'][0]);row['offload_reduction_percent']=100*row['offload_reduction_ms']/row['baseline_offload_ms']
  sr.append(row)
 cr=[]
 for f in family:
  row={k:f[k] for k in ('attempt','comparator','baseline','mode','batch','length','status','p_for_multiplicity','holm_adjusted_p','confirmed_local_gain')}
  if f.get('result'):
   r=f['result'];row.update(nominal_dual_gain=r['nominal_dual_gain'],resolved_resident_slowdown=r['resolved_resident_slowdown'],memory_savings_preserved=r['memory_savings_preserved'],min_gpu_savings_mib=min(r['gpu_savings_bytes'])/2**20)
   for metric,stat in r['statistics'].items():
    for field in ('mean','lower_95','upper_95'):row[metric+'_'+field]=stat[field]
   row['source_manifest']=r['source_manifest']
  cr.append(row)
 export(sr,out/'screen.csv');export(cr,out/'confirmation.csv')
 measurements=[]
 for manifest in sorted(D.glob('A*/*/manifest.json')):
  data=s.read(manifest)
  if data['status']!='completed':continue
  for job in data['jobs']:
   raw=manifest.parent/(job['name']+'.json');payload=s.read(raw);r=payload['results'][0]
   row={k:job[k] for k in ('attempt','implementation','candidate_sha','mode','batch','length','variant','block')}
   row.update(comparison_attempt=data['attempt'],stage=manifest.parent.name,latency_ms=r['mean_ms'],tokens_per_second=r['tokens_per_second'],samples_ms=json.dumps(r['samples_ms']),source_result=str(raw))
   flat=[x for rnd in r['samples_ms'] for x in rnd];row.update(sample_sd_ms=statistics.stdev(flat),sample_min_ms=min(flat),sample_max_ms=max(flat),sample_p50_ms=r['sample_p50_ms'],sample_p95_ms=r['sample_p95_ms'],median_ms=r['median_ms'])
   for k,v in r['memory_after'].items():row[k]=json.dumps(v) if isinstance(v,(list,dict)) else v
   measurements.append(row)
 export(measurements,out/'measurements.csv')
 s.save(out/'source.json',dict(screen=screen,confirmation=confirmation))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 keys=sorted({(r['mode'],r['batch'],r['length']) for r in screen})
 labels=[f'{m}\nb{b} / {l}' for m,b,l in keys];attempts=s.ATTEMPTS
 for metric,title in [('offload_reduction_ms','Offload latency reduction (ms)'),('gap_reduction_ms','Offload–resident gap reduction (ms)')]:
  values=np.full((len(attempts),len(keys)),np.nan)
  for r in screen:values[attempts.index(r['attempt']),keys.index((r['mode'],r['batch'],r['length']))]=r['statistics'][metric]['mean']
  finite=values[np.isfinite(values)];limit=max(.01,float(np.percentile(abs(finite),85)))
  fig,ax=plt.subplots(figsize=(max(13,len(keys)*1.3),11));im=ax.imshow(values,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
  ax.set_xticks(range(len(keys)),labels,fontsize=8);ax.set_yticks(range(len(attempts)),attempts);ax.set_title(title+' — descriptive screening only',pad=14)
  for i in range(len(attempts)):
   for j in range(len(keys)):
    if np.isfinite(values[i,j]):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=7,color='white' if abs(values[i,j])>limit*.65 else 'black')
    else:ax.text(j,i,'—',ha='center',va='center',color='gray')
  fig.colorbar(im,ax=ax,label='Positive = faster / smaller gap; color saturates, labels show full values')
  fig.text(.02,.02,'2.8365B BF16, RTX 5090. Original parent comparison; fresh matched resident/offload screens except historical b16.\nPrefill: last-token logits + KV; decode: fixed real KV; generation: prefix2048 +128 predetermined steps.\nRandom weights; no quality claim. Unmeasured/inactive cells shown as —. Screens do not establish gains.',fontsize=9)
  fig.subplots_adjust(left=.08,right=.9,top=.93,bottom=.14)
  for ext in ('png','svg','pdf'):fig.savefig(out/(metric+'.'+ext),dpi=170)
  plt.close(fig)
 # All measured confirmation intervals, split by attempt for readable complete coverage.
 for a in attempts:
  rows=[f for f in family if f['attempt']==a and f['status']=='measured']
  if not rows:continue
  fig,axs=plt.subplots(1,2,figsize=(14,max(4,len(rows)*.37+2)),sharey=True)
  ys=list(range(len(rows)));yl=[f"{f['comparator']} {f['result']['mode']} b{f['batch']}/{f['length']}" for f in rows]
  for ax,metric in zip(axs,('offload_reduction_ms','gap_reduction_ms')):
   for y,f in zip(ys,rows):
    stat=f['result']['statistics'][metric];color='tab:green' if f['confirmed_local_gain'] else 'tab:blue'
    ax.errorbar(stat['mean'],y,xerr=[[stat['mean']-stat['lower_95']],[stat['upper_95']-stat['mean']]],fmt='o',capsize=3,color=color)
   ax.axvline(0,color='gray',lw=1);ax.set_xlabel(metric.replace('_',' '));ax.grid(axis='x',alpha=.25)
  axs[0].set_yticks(ys,yl,fontsize=8);axs[0].invert_yaxis();fig.suptitle(a+' independent confirmations; positive = improvement')
  fig.text(.02,.015,'2.8365B BF16 / RTX 5090; 3 fresh balanced process blocks, two-sided 95% paired t intervals (df=2).\nGreen: joint latency/gap Holm gate and memory/resident checks pass; local performance evidence, not automatic acceptance.\nScreens excluded; all measured selected points retained. Prefix/last-token-logit/KV and generation scope match the campaign.',fontsize=8)
  fig.subplots_adjust(left=.24,right=.98,top=.88,bottom=.2,wspace=.15)
  for ext in ('png','svg','pdf'):fig.savefig(out/(a+'-confirmation.'+ext),dpi=160)
  plt.close(fig)
 lines=['# Supplemental shape survey results','',f"Evaluated all18 nonretained frozen attempts at118 newly selected workload points (472 fresh screening processes, including the predeclared12-process A0011 bulk-coverage addendum), plus eligible historical batch16 screens. {len(family)} potential independent confirmation comparisons were fixed before confirmation; {sum(f['status']=='measured' for f in family)} were measured. {sum(f['confirmed_local_gain'] for f in family)} comparisons pass the local Holm-adjusted dual latency/gap gate. Historical verdicts and retained A0016 code remain unchanged.",'','Existing accepted-source evidence already shows a secondary-shape benefit: A0016 versus A0000 at b8/512 prefill reduces offload latency26.9542 ms [25.7087,28.1996] and gap26.9798 ms [25.6126,28.3469] in three retained regression pairs ([source](../../A0016/full-regression-analysis.json)). This is cumulative existing evidence, attributable to the A0001 cutoff path; it is not a new supplemental discovery. Earlier conversational claims of no b8 prefill gain apply only to the primary length2048.','','The purpose is to test whether primary-shape screening missed workload-specific improvements. This survey does not prove that every unmeasured shape is equivalent. See [predeclared scope](../PLAN.md), [screen table](screen.csv), [complete confirmation table with intervals](confirmation.csv), and [raw structured source](source.json), [all process latency/throughput/memory rows](measurements.csv), and [explicit coverage map](../coverage.json).','','| Attempt | Workload | Comparator | Offload reduction ms [95% CI] | Gap reduction ms [95% CI] | Holm p | Local gate |','|---|---|---|---|---|---:|---|']
 for f in family:
  if f['status']!='measured':continue
  r=f['result'];st=r['statistics'];fmt=lambda k:f"{st[k]['mean']:.4f} [{st[k]['lower_95']:.4f}, {st[k]['upper_95']:.4f}]"
  lines.append(f"| {f['attempt']} | {f['mode']} b{f['batch']}/{f['length']} | {f['baseline']} ({f['comparator']}) | {fmt('offload_reduction_ms')} | {fmt('gap_reduction_ms')} | {f['holm_adjusted_p']:.4g} | {'pass' if f['confirmed_local_gain'] else 'not established'} |")
 lines+=['','All screens and every unfavorable confirmation remain available. Current-comparison slots skipped by the predeclared parent gate retain p=1 in the fixed multiplicity family; they do not shrink the correction denominator. Passing a parent comparison alone does not show superiority to current A0016. Passing a local comparison does not resolve regressions elsewhere or authorize retention. In particular, A0014’s prior primary regression remains part of its evidence.','','Intervals use only three independent balanced process differences; within-process rounds are not independent replicates. The t model assumes sufficiently symmetric differences, not verified with n=3. Holm controls multiplicity conditional on valid p-values, but does not repair distribution or measurement assumptions. Inconclusive results are not equivalence claims. Seeded random weights support performance and equivalence experiments, not language quality.','','Prefill/decode screen3 warmups/5 samples. Formal prefill/decode10 warmups/3×10 samples. Generation2 warmups/3×5 trajectories at prefix2048+128 predetermined steps, excluding sampling. Every source is frozen and both placements measured; GPU/host buffers, pinned tables, inverse maps and cached KV are audited from raw telemetry. All GPU jobs ran sequentially.','','## Figures','','[Screen offload reduction](offload_reduction_ms.png), [screen gap reduction](gap_reduction_ms.png). Colors are symmetric and saturated for readability; numeric labels retain full signed values. Blank cells are unmeasured, never zero.']
 for a in attempts:
  if (out/(a+'-confirmation.png')).exists():lines.append(f'- [{a} all confirmation intervals]({a}-confirmation.png)')
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print('Report and figures generated',out)
if __name__=='__main__':main()
