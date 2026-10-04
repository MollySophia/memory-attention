"""Freeze independent confirmation after a complete screen; no automatic retention."""
import argparse,json
import driver as d
from scipy.stats import t
import statistics,math
PRIMARY={('prefill',1,2048),('decode',1,2048),('prefill',8,2048),('decode',8,2048)}

def key(row):return tuple(row[k] for k in ('mode','batch','length'))

def nominations(rows):
 selected={};assert {key(r) for r in rows}==set(d.WORKLOADS)
 for r in rows:
  st=r['statistics'];lat=r['latencies'][0];reasons=[]
  if r['screen_promising']:reasons.append('positive offload and gap screen with GPU savings')
  off_tol=max(.1,.01*lat['baseline_offload_ms']);gpu_tol=max(.05,.01*lat['baseline_gpu_ms'])
  if st['offload_reduction_ms']['mean'] < -off_tol:reasons.append('offload regression trigger')
  if st['gap_reduction_ms']['mean'] < -off_tol:reasons.append('gap regression trigger')
  if st['gpu_reduction_ms']['mean'] < -gpu_tol:reasons.append('resident slowdown trigger')
  if not r['memory_savings_preserved']:reasons.append('memory savings trigger')
  if key(r) in PRIMARY:reasons.append('mandatory original primary control')
  if reasons:selected[key(r)]=reasons
 return selected

def holm(values):
 out=[None]*len(values);prev=0
 for rank,(index,p) in enumerate(sorted(enumerate(values),key=lambda x:x[1])):
  prev=max(prev,min(1.,p*(len(values)-rank)));out[index]=prev
 return out

def prepare(a):
 screen=d.summarize(d.C/a/'R01-complete-screen');selected=nominations(screen['rows'])
 directory=d.C/a/'R02-parent-confirmation';directory.mkdir(exist_ok=False)
 shapes=[w for w in d.WORKLOADS if w in selected]
 p=d.plan(a,shapes,d.record(a)['parent_attempt_id'],True,directory)
 p['nomination_reasons']=[dict(mode=w[0],batch=w[1],length=w[2],reasons=selected[w]) for w in shapes]
 p['fixed_gain_family']=len(shapes);p['stopping_rule']='Exactly3 balanced independent blocks; no optional extension. Screens excluded. No automatic retention.'
 p['regression_rule']='Resolved slowdown: paired two-sided95% interval of offload or gap reduction entirely below0; resident guard likewise; preserve adverse directions and memory losses.'
 d.save(directory/'manifest.json',p);print(json.dumps(p['planned_work']),flush=True)
 return p

def analyze(a):
 directory=d.C/a/'R02-parent-confirmation';summary=d.summarize(directory);p=d.read(directory/'manifest.json');rows=summary['rows']
 assert len(rows)==p['fixed_gain_family']
 adj=holm([r['joint_one_sided_p'] for r in rows])
 for r,pv in zip(rows,adj):
  st=r['statistics'];r['holm_adjusted_p']=pv
  r['confirmed_local_gain']=pv<.05 and r['nominal_dual_gain'] and not r['resolved_resident_slowdown'] and r['memory_savings_preserved']
  r['resolved_offload_regression']=st['offload_reduction_ms']['upper_95']<0
  r['resolved_gap_regression']=st['gap_reduction_ms']['upper_95']<0
  # Exploratory current offload-vs-resident bounds, not the final all16-workload audit.
  gaps=[l['candidate_offload_ms']-l['candidate_gpu_ms'] for l in r['latencies']]
  residuals=[l['candidate_offload_ms']-l['candidate_gpu_ms']-max(.1,.01*l['candidate_gpu_ms']) for l in r['latencies']]
  r['candidate_gap_ms']=d.interval(gaps)
  r['candidate_target_residual_ms']=d.interval(residuals)
  r['candidate_target_residual_upper_one_sided_95_bonferroni16']=statistics.mean(residuals)+float(t.ppf(1-.05/16,2))*statistics.stdev(residuals)/math.sqrt(3)
 result=dict(status='completed',attempt=a,baseline=p['baseline'],fixed_gain_family=len(rows),rows=rows,confirmed_local_gains=[key(r) for r in rows if r['confirmed_local_gain']],resolved_regressions=[key(r) for r in rows if r['resolved_offload_regression'] or r['resolved_gap_regression'] or r['resolved_resident_slowdown'] or not r['memory_savings_preserved']],accepted=False,note='No automatic acceptance. Complete correctness/full validation required; target residual bounds cover selected points only, not final goal completion.')
 d.save(directory/'analysis.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze']);p.add_argument('--attempt',required=True);args=p.parse_args()
 if args.action=='prepare':prepare(args.attempt)
 elif args.action=='run':d.run(d.C/args.attempt/'R02-parent-confirmation');analyze(args.attempt)
 else:analyze(args.attempt)
