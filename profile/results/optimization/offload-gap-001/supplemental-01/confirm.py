"""Predeclared independent confirmation and multiplicity accounting for the survey.
Screen data are used for nomination only. Never extend a confirmation plan.
"""
import argparse,json,os,subprocess,time
from pathlib import Path
import study as s
D=s.D;C=s.C

def holm(pvalues):
 """Adjusted p values over the entire fixed family, including unrun p=1 slots."""
 indexed=sorted(enumerate(pvalues),key=lambda v:v[1]);result=[None]*len(indexed);previous=0
 for rank,(index,p) in enumerate(indexed):
  previous=max(previous,min(1.,(len(indexed)-rank)*p));result[index]=previous
 return result

def historical(a):
 # Existing b16 screening can nominate a local effect; never contributes to inference.
 modes=['decode'] if a=='A0013' else ['prefill','decode'] if a=='A0011' else [] if a in ('A0002','A0005') else ['prefill']
 parent=s.record(a)['parent_attempt_id'];entries=[]
 for mode in modes:
  rr={};refs=[]
  for impl,aid in [('baseline',parent),('candidate',a)]:
   manifest=s.read(C/aid/'R01/manifest.json')
   for variant in ('ma_offload','ma_gpu'):
    j=next(x for x in manifest['jobs'] if (x['mode'],x['batch'],x['length'],x['variant'])==(mode,16,2048,variant)).copy()
    j.update(attempt=aid,candidate_sha=s.record(aid)['candidate_sha'],source_sha256=s.source_hash(s.root(aid)))
    path=C/aid/'R01'/(j['name']+'.json');s.audit_job(j,path);rr[impl,variant]=s.read(path)['results'][0];refs.append(str(path))
  bo,bg,co,cg=[rr[i,v]['mean_ms'] for i,v in [('baseline','ma_offload'),('baseline','ma_gpu'),('candidate','ma_offload'),('candidate','ma_gpu')]]
  savings=rr['candidate','ma_gpu']['memory_after']['gpu_peak_allocated_bytes']-rr['candidate','ma_offload']['memory_after']['gpu_peak_allocated_bytes']
  entries.append(dict(attempt=a,baseline=parent,mode=mode,batch=16,length=2048,screen_promising=bo>co and bo-bg>co-cg and savings>0,statistics=dict(offload_reduction_ms=dict(mean=bo-co),gap_reduction_ms=dict(mean=(bo-bg)-(co-cg))),source_results=refs,nomination_source='historical_screen_only'))
 return entries

def prepare():
 assert not (D/'confirmation-plan.json').exists()
 assert s.read(D/'screen-controller.json')['status']=='completed'
 nominees=[];allpoints=[];comparisons=[]
 for a in s.ATTEMPTS:
  new=s.summarize(D/a/'screen')['rows'];old=historical(a)
  s.save(D/a/'historical-b16-review.json',dict(rows=old))
  points=new+old;allpoints+=points;selected=[r for r in points if r['screen_promising']]
  nominees+=selected
  if not selected:continue
  workloads=[(r['mode'],r['batch'],r['length']) for r in selected];parent=s.record(a)['parent_attempt_id']
  for comparator,baseline in [('parent',parent)]+([('current','A0016')] if parent!='A0016' else []):
   directory=D/a/('confirmation-'+comparator);directory.mkdir(exist_ok=False)
   p=s.plan(a,workloads,baseline,True,directory)
   p['launch_condition']='all nominated workloads' if comparator=='parent' else 'Only parent workloads passing nominal dual 95% gain, no resolved resident slowdown and positive GPU savings; other slots remain unrun with p=1 in fixed family.'
   s.save(directory/'initial-plan.json',p)
   if comparator=='parent':s.save(directory/'manifest.json',p)
   for r in selected:comparisons.append(dict(attempt=a,comparator=comparator,baseline=baseline,mode=r['mode'],batch=r['batch'],length=r['length'],status='planned',p_for_multiplicity=1.0))
 plan=dict(status='frozen_before_independent_confirmation',created_unix=time.time(),nominees=nominees,family=comparisons,family_size=len(comparisons),parent_jobs=12*len(nominees),maximum_current_jobs=12*sum(r['comparator']=='current' for r in comparisons),screen_points_reviewed=len(allpoints),rule='Exactly three new independent blocks; screens excluded. Fixed potential parent/current family; unrun current slots p=1. Holm-adjusted conjunction p plus both two-sided95% lower bounds positive, memory savings and no resident slowdown. Local evidence only; not acceptance.')
 s.save(D/'confirmation-plan.json',plan)
 s.save(D/'screen-summary.json',dict(rows=allpoints,nominees=len(nominees),raw_new_results=460))
 print(json.dumps({k:v for k,v in plan.items() if k not in ('nominees','family')}),flush=True)

def passes(row):return row['nominal_dual_gain'] and row['memory_savings_preserved'] and not row['resolved_resident_slowdown']

def run():
 assert not (D/'confirmation-controller.json').exists()
 plan=s.read(D/'confirmation-plan.json');state=dict(status='running',pid=os.getpid(),started_unix=time.time(),completed=[])
 s.save(D/'confirmation-controller.json',state)
 for a in s.ATTEMPTS:
  parentdir=D/a/'confirmation-parent'
  if not parentdir.exists():continue
  state['active']=str(parentdir);s.save(D/'confirmation-controller.json',state)
  s.run(parentdir);summary=s.summarize(parentdir);state['completed'].append(str(parentdir));s.save(D/'confirmation-controller.json',state)
  current=D/a/'confirmation-current'
  if not current.exists():continue
  eligible={(r['mode'],r['batch'],r['length']) for r in summary['rows'] if passes(r)}
  if not eligible:
   s.save(current/'not-run.json',dict(status='not_run_parent_gate',reason='No nominated parent workload passed the fixed local gate; all potential current hypotheses retain p=1 in the family.'));continue
  p=s.read(current/'initial-plan.json');p['jobs']=[j for j in p['jobs'] if (j['mode'],j['batch'],j['length']) in eligible]
  s.validate_balanced_order(p['jobs']);p['planned_work']=dict(jobs=len(p['jobs']),estimated_seconds=sum(j['estimated_seconds'] for j in p['jobs']))
  s.save(current/'manifest.json',p);print('CURRENT PLAN',a,json.dumps(p['planned_work']),flush=True)
  state['active']=str(current);s.save(D/'confirmation-controller.json',state);s.run(current);s.summarize(current);state['completed'].append(str(current));s.save(D/'confirmation-controller.json',state)
 state.update(status='completed',finished_unix=time.time(),active=None);s.save(D/'confirmation-controller.json',state);analyze()

def analyze():
 plan=s.read(D/'confirmation-plan.json');family=plan['family'];measured={}
 for p in D.glob('A*/confirmation-*/summary.json'):
  comparator=p.parent.name.removeprefix('confirmation-')
  for r in s.summarize(p.parent)['rows']:measured[r['attempt'],comparator,r['mode'],r['batch'],r['length']]=r
 for f in family:
  key=tuple(f[k] for k in ('attempt','comparator','mode','batch','length'))
  if key in measured:
   r=measured[key];f.update(status='measured',p_for_multiplicity=r['joint_one_sided_p'],result=r)
  else:
   assert f['comparator']=='current',('Missing required parent evidence',key)
   f.update(status='not_run_parent_gate',p_for_multiplicity=1.0)
 for f,adj in zip(family,holm([f['p_for_multiplicity'] for f in family])):
  f['holm_adjusted_p']=adj;f['confirmed_local_gain']=f['status']=='measured' and passes(f['result']) and adj<0.05
 result=dict(status='completed',family_size=len(family),family=family,confirmed_local_comparisons=sum(f['confirmed_local_gain'] for f in family),accepted_new_attempts=0,retained_source='A0016',note='Local evidence only. Three-block t assumptions and power limits remain. Current-only or parent-only successes do not establish an integrated, regression-free improvement.')
 indexed={(f['attempt'],f['comparator'],f['mode'],f['batch'],f['length']):f for f in family}
 useful=[]
 for f in family:
  if f['comparator']!='parent' or not f['confirmed_local_gain']:continue
  current=f if f['baseline']=='A0016' else indexed[f['attempt'],'current',f['mode'],f['batch'],f['length']]
  if current['confirmed_local_gain']:useful.append(dict(attempt=f['attempt'],mode=f['mode'],batch=f['batch'],length=f['length'],parent_comparison=f,current_comparison=current))
 result['local_gains_vs_parent_and_current']=useful
 s.save(D/'confirmation-summary.json',result);print('FINAL',result['confirmed_local_comparisons'],'local comparisons;',len(useful),'workloads pass both parent/current gates',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze']);a=p.parse_args();globals()[a.action]()
