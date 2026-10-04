"""Complete fresh parent-paired coverage without repeating already confirmed points.
No automatic acceptance. Full-model fingerprints and folding references follow.
"""
import argparse,json,subprocess,sys,time,os
from pathlib import Path
import driver as d
import confirm

def prepare(a):
 result=confirm.analyze(a)
 assert result['confirmed_local_gains'] and not result['resolved_regressions'],'Confirmation requires review before validation'
 seen={confirm.key(r) for r in result['rows']}
 remaining=[w for w in d.WORKLOADS if w not in seen]
 directory=d.C/a/'R03-remaining-confirmation';directory.mkdir(exist_ok=False)
 p=d.plan(a,remaining,d.record(a)['parent_attempt_id'],True,directory)
 p['purpose']='Complete all16 independent parent-comparison workloads; retain R02 evidence, no resampling selected points'
 d.save(directory/'manifest.json',p)
 # One formal-sampling folded/unfolded reference per workload per implementation.
 fold_dir=d.C/a/'R04-unfolded-reference';fold_dir.mkdir(exist_ok=False)
 f=d.plan(a,d.WORKLOADS,d.record(a)['parent_attempt_id'],False,fold_dir)
 f['jobs']=[j for j in f['jobs'] if j['variant']=='ma_gpu']
 for j in f['jobs']:
  j['name']=j['name'].replace('ma_gpu','ma_gpu_unfolded');j['variant']='ma_gpu_unfolded'
  command=d.cmd(j['attempt'],(j['mode'],j['batch'],j['length']),fold_dir/(j['name']+'.json'),True)
  command[command.index('PLACEHOLDER')]=j['variant'];j['command']=command
  j['estimated_seconds']=d.estimate(j['attempt'],(j['mode'],j['batch'],j['length']),'ma_gpu',True)
 f['stage']='unfolded_reference';f['purpose']='Folding ablation only; not independent improvement evidence'
 f['planned_work']=dict(jobs=len(f['jobs']),estimated_seconds=sum(j['estimated_seconds'] for j in f['jobs']))
 d.save(fold_dir/'manifest.json',f)
 state=dict(status='planned',attempt=a,remaining=p['planned_work'],unfolded=f['planned_work'],correctness_pairs=[dict(batch=b,length=l,seed=s) for b,l in ((1,2048),(8,2048),(8,512)) for s in (1234,4321)],correctness_reference='A0000',prior_confirmation='R02-parent-confirmation/manifest.json',accepted=False)
 d.save(d.C/a/'validation-plan.json',state);print(json.dumps(state),flush=True)

def paired_audit(a):
 rows=[]
 for stage in ('R02-parent-confirmation','R03-remaining-confirmation'):
  manifest=d.read(d.C/a/stage/'manifest.json');assert manifest['status']=='completed'
  if manifest['jobs']:rows+=d.summarize(d.C/a/stage)['rows']
 assert len(rows)==16 and {confirm.key(r) for r in rows}==set(d.WORKLOADS)
 # Full-matrix descriptive/guard evidence. Original selected-family gain decisions stay fixed.
 adverse=[]
 for r in rows:
  st=r['statistics']
  r['resolved_offload_regression']=st['offload_reduction_ms']['upper_95']<0
  r['resolved_gap_regression']=st['gap_reduction_ms']['upper_95']<0
  if r['resolved_offload_regression'] or r['resolved_gap_regression'] or r['resolved_resident_slowdown'] or not r['memory_savings_preserved']:adverse.append(confirm.key(r))
 result=dict(status='audited',attempt=a,rows=rows,resolved_regressions=adverse,accepted=False,note='All16 parent-paired points; screening excluded. Original R02 multiplicity gate remains authoritative for selected local gains. Complete full-model correctness and folding/memory audit before retention.')
 d.save(d.C/a/'full-parent-analysis.json',result);return result

def run(a):
 p=d.read(d.C/a/'validation-plan.json');assert p['status']=='planned'
 statepath=d.C/a/'validation-controller.json';assert not statepath.exists()
 state=dict(status='running',controller_pid=os.getpid(),started_unix=time.time(),jobs=[],accepted=False);d.save(statepath,state)
 def child(name,cmd):
  j=dict(name=name,command=cmd,status='running',started_unix=time.time());state['jobs'].append(j);state['active']=name
  with (d.C/a/(name+'-controller.txt')).open('w') as log:
   process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);j['pid']=process.pid;d.save(statepath,state);j['returncode']=process.wait()
  j.update(status='completed' if j['returncode']==0 else 'failed',finished_unix=time.time());d.save(statepath,state)
  assert not j['returncode'],name
 try:
  d.run(d.C/a/'R03-remaining-confirmation');analysis=paired_audit(a)
  if analysis['resolved_regressions']:
   state.update(status='regression_needs_review',resolved_regressions=analysis['resolved_regressions']);d.save(statepath,state);return
  d.run(d.C/a/'R04-unfolded-reference')
  outputs=d.C/a/'R05-full-correctness';outputs.mkdir(exist_ok=False)
  for scope in p['correctness_pairs']:
   b,l,s=[scope[k] for k in ('batch','length','seed')];paths={}
   for impl,aid,variant in [('baseline','A0000','ma_gpu'),('candidate',a,'ma_offload')]:
    name=f'correctness-b{b}-l{l}-s{s}-{impl}';paths[impl]=outputs/(name+'.json')
    child(name,[sys.executable,str(d.D/'full_correctness.py'),'--source-root',str(d.root(aid)),'--variant',variant,'--batch-size',str(b),'--prefix-length',str(l),'--seed',str(s),'--output',str(paths[impl])])
   child(f'compare-b{b}-l{l}-s{s}',[sys.executable,str(d.C/'compare_correctness.py'),str(paths['baseline']),str(paths['candidate']),'--output',str(outputs/f'comparison-b{b}-l{l}-s{s}.json')])
  state.update(status='validation_complete_needs_audit',finished_unix=time.time(),active=None);d.save(statepath,state)
 except BaseException as e:
  state.update(status='stopped_needs_review',error=repr(e));d.save(statepath,state);raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze']);p.add_argument('--attempt',required=True);args=p.parse_args()
 if args.action=='prepare':prepare(args.attempt)
 elif args.action=='run':run(args.attempt)
 else:paired_audit(args.attempt)
