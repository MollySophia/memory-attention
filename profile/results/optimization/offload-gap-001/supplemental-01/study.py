"""Frozen-source shape survey; independent confirmation is a separate phase.
No historical verdict or execution source is modified. No automatic acceptance.
"""
import argparse,csv,hashlib,json,math,os,statistics,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent; C=D.parent
sys.path.insert(0,str(C))
from run_paired import source_hash,save,implementation_order,validate_balanced_order
from audit_continuation import validate_offload_memory
from analyze_paired import interval
PREFILLS=[('prefill',4,2048),('prefill',8,512),('prefill',8,4096),('prefill',8,8192)]
DECODES=[('decode',4,2048),('decode',8,512),('decode',8,4096),('decode',8,8192)]
GEN=[('generation',1,2048),('generation',8,2048)]
ATTEMPTS=[f'A{i:04d}' for i in range(2,21) if i!=16]
def read(p):return json.loads(Path(p).read_text())
def root(a):return Path('/home/molly/workspace-memory-attn')/('offload-gap-001-'+a)
def record(a):return read(C/a/'record.json')
def shapes(a):
 if a in ('A0002','A0005'):return DECODES+GEN
 if a=='A0013':return DECODES[1:]+GEN[1:]
 base=PREFILLS.copy()
 if a in ('A0009','A0015','A0017'):base.remove(('prefill',8,512))
 # A high-context decode control also checks indirect effects of cached pipeline storage.
 base+=[('decode',8,8192)]
 return base+(GEN[1:] if a in ('A0009','A0015','A0017') else GEN)
def signature(a):
 r=root(a);sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
 assert sha==record(a)['candidate_sha']
 assert not subprocess.check_output(['git','status','--porcelain','--','fla',*map(str,Path('profile').glob('*.py'))],cwd=r)
 return dict(attempt=a,root=str(r),commit=sha,source_sha256=source_hash(r))
def cmd(a,shape,path,formal):
 mode,b,l=shape;stage='generation_validation' if mode=='generation' else 'confirmation' if formal else 'screening'
 w,n,r=(2,5,3) if mode=='generation' else (10,10,3) if formal else (3,5,1)
 return [sys.executable,str(root(a)/'profile/bench_fla.py'),'--mode',mode,'--variants','PLACEHOLDER','--batch-size',str(b),'--seq-len',str(l),'--context-len',str(l),'--num-layers','24','--hidden-size','2048','--num-heads','32','--num-kv-heads','32','--intermediate-size','5632','--vocab-size','32000','--seed','1234','--stage',stage,'--warmup',str(w),'--repeats',str(n),'--rounds',str(r),'--logits-to-keep','1','--prefill-workload','inference','--generation-steps','128','--policy','auto','--group-size','1','--prefetch-depth','4','--json',str(path)]
def estimate(a,shape,variant,formal):
 m,b,l=shape
 screen=read(C/a/'R01/manifest.json')
 ref=[j for j in screen['jobs'] if j['variant']==variant and j['mode']==('prefill' if m=='generation' else m)]
 j=min(ref,key=lambda j:abs(j['batch']-b));row=read(C/a/'R01'/(j['name']+'.json'))['results'][0]
 setup=max(0,j['finished_unix']-j['started_unix']-8*row['mean_ms']/1000)
 # Bound anomalously paused historical setup with the campaign's observed ~21 s cost.
 setup=min(setup,30)
 latency=row['mean_ms']*(b/j['batch'])*(l/2048 if m!='decode' else max(1,l/2048))
 if m=='generation':
  dj=next(x for x in screen['jobs'] if x['variant']==variant and x['mode']=='decode' and x['batch']==b)
  latency+=128*read(C/a/'R01'/(dj['name']+'.json'))['results'][0]['mean_ms']
 return setup+latency/1000*(17 if m=='generation' else 40 if formal else 8)
def plan(a,workloads,baseline,formal,directory):
 sig={i:signature(x) for i,x in [('baseline',baseline),('candidate',a)]};jobs=[]
 for block in range(3 if formal else 1):
  sequence=list(enumerate(workloads));sequence=sequence[::-1] if block%2 else sequence
  for wi,shape in sequence:
   variants=['ma_offload','ma_gpu'];variants=variants[::-1] if (wi+block)%2 else variants
   for v in variants:
    for impl in implementation_order(block,wi,('ma_offload','ma_gpu').index(v)):
     aid=sig[impl]['attempt'];m,b,l=shape;name=f'B{block+1}-J{len(jobs)+1:03d}-{impl}-{m}-{v}-b{b}-l{l}'
     command=cmd(aid,shape,directory/(name+'.json'),formal);command[command.index('PLACEHOLDER')]=v
     jobs.append(dict(name=name,block=block+1,mode=m,batch=b,length=l,variant=v,implementation=impl,attempt=aid,candidate_sha=sig[impl]['commit'],source_sha256=sig[impl]['source_sha256'],cwd=sig[impl]['root'],command=command,estimated_seconds=estimate(aid,shape,v,formal),status='pending'))
 if formal:validate_balanced_order(jobs)
 return dict(campaign_id='offload-gap-001',study='supplemental-01',attempt=a,baseline=baseline,stage='independent_confirmation' if formal else 'supplemental_screen',formal=formal,status='planned',source_signatures=sig,jobs=jobs,planned_work=dict(jobs=len(jobs),estimated_seconds=sum(j['estimated_seconds'] for j in jobs)),accepted=False)
def audit_job(j,path):
 p=read(path);assert p['status']=='completed';assert p['source']['source_sha256']==j['source_sha256'];assert p['source']['git_commit']['stdout'].strip()==j['candidate_sha']
 cfg=p['config'];row=p['results'][0]
 for k,v in dict(mode=j['mode'],batch_size=j['batch'],seq_len=j['length'],context_len=j['length'],variants=[j['variant']],seed=1234,logits_to_keep=1,hidden_size=2048,num_layers=24,num_heads=32,num_kv_heads=32,intermediate_size=5632,vocab_size=32000,policy='auto',prefetch_depth=4,group_size=1,generation_steps=128).items():assert cfg[k]==v,(k,cfg[k],v)
 for key in ('warmup','repeats','rounds'):assert cfg[key]==int(j['command'][j['command'].index('--'+key)+1])
 assert len(row['samples_ms'])==cfg['rounds'] and all(len(x)==cfg['repeats'] for x in row['samples_ms'])
 assert all(math.isfinite(x) and x>0 for rnd in row['samples_ms'] for x in rnd)
 expected=('generation_v1_w2_n5_r3' if j['mode']=='generation' else 'screen_v1_w3_n5_r1' if cfg['warmup']==3 else 'formal_v1_w10_n10_r3')
 assert p['measurement_plan_id']==expected
 assert row['output_scope']=='cached_logits'
 if j['variant']=='ma_offload':validate_offload_memory(j['attempt'],row['memory_after'])
 env=p['environment_before'];e=p['env']
 return (e['torch'],e['torch_cuda'],e['flash_attn'],e['gpu'],e['python'],env['gpu_telemetry']['stdout'].splitlines()[1].split(', ')[1],tuple(env['cpu_affinity']),env['torch_threads'],env['torch_interop_threads'],json.dumps(env['thread_environment'],sort_keys=True)),row

def run(directory):
 path=directory/'manifest.json';p=read(path);assert p['status']=='planned'
 for s in p['source_signatures'].values():assert signature(s['attempt'])==s
 p.update(status='running',controller_pid=os.getpid(),started_unix=time.time());save(path,p)
 try:
  for j in p['jobs']:
   assert j['status']=='pending';print('START',p['attempt'],j['name'],flush=True);j['started_unix']=time.time()
   with (directory/(j['name']+'.txt')).open('w') as log:
    child=subprocess.Popen(j['command'],cwd=j['cwd'],stdout=log,stderr=subprocess.STDOUT);j.update(pid=child.pid,status='running');save(path,p);j['returncode']=child.wait()
   result=directory/(j['name']+'.json');j['finished_unix']=time.time()
   j['status']=read(result).get('status','failed') if result.exists() else 'failed'
   if j['returncode']:j['status']='failed'
   save(path,p)
   if j['status']!='completed':raise RuntimeError('Failed job preserved: '+str(result))
   audit_job(j,result)
  p.update(status='completed',finished_unix=time.time());save(path,p)
 except BaseException as error:
  p.update(status='stopped_needs_review',error=repr(error));save(path,p);raise

def summarize(directory):
 p=read(directory/'manifest.json');assert p['status']=='completed';envs=set();data={};rows=[]
 for j in p['jobs']:
  env,r=audit_job(j,directory/(j['name']+'.json'));envs.add(env);data[j['mode'],j['batch'],j['length'],j['block'],j['implementation'],j['variant']]=r
 assert len(envs)==1,'Environment mismatch in paired results'
 from scipy.stats import t
 for shape in sorted({k[:3] for k in data}):
  values={k:[] for k in ('offload_reduction_ms','gpu_reduction_ms','gap_reduction_ms','offload_speedup')};memory=[];means=[]
  for block in range(1,4 if p['formal'] else 2):
   rr=[data[*shape,block,i,v] for i,v in [('baseline','ma_offload'),('baseline','ma_gpu'),('candidate','ma_offload'),('candidate','ma_gpu')]]
   bo,bg,co,cg=[r['mean_ms'] for r in rr];means.append(dict(baseline_offload_ms=bo,baseline_gpu_ms=bg,candidate_offload_ms=co,candidate_gpu_ms=cg))
   for k,x in zip(values,(bo-co,bg-cg,(bo-bg)-(co-cg),bo/co)):values[k].append(x)
   memory.append(rr[3]['memory_after']['gpu_peak_allocated_bytes']-rr[2]['memory_after']['gpu_peak_allocated_bytes'])
  stat={k:interval(v) if p['formal'] else dict(mean=v[0],values=v) for k,v in values.items()}
  row=dict(attempt=p['attempt'],baseline=p['baseline'],mode=shape[0],batch=shape[1],length=shape[2],statistics=stat,latencies=means,gpu_savings_bytes=memory,memory_savings_preserved=min(memory)>0,source_manifest=str(directory/'manifest.json'),accepted=False)
  row['screen_promising']=stat['offload_reduction_ms']['mean']>0 and stat['gap_reduction_ms']['mean']>0 and min(memory)>0
  if p['formal']:
   def pv(xs):
    sd=statistics.stdev(xs);mu=statistics.mean(xs)
    return float(t.sf(mu/(sd/(3**.5)),2)) if sd else (0.0 if mu>0 else 1.0)
   row['joint_one_sided_p']=max(pv(values['offload_reduction_ms']),pv(values['gap_reduction_ms']))
   row['nominal_dual_gain']=all(stat[k]['lower_95']>0 for k in ('offload_reduction_ms','gap_reduction_ms'))
   row['resolved_resident_slowdown']=stat['gpu_reduction_ms']['upper_95']<0
  rows.append(row)
 result=dict(status='audited',formal=p['formal'],rows=rows,checked_raw_results=len(p['jobs']),source_manifest=str(directory/'manifest.json'))
 save(directory/'summary.json',result);return result

def prepare():
 assert not (D/'plan.json').exists()
 descriptions=[]
 for a in ATTEMPTS:
  r=record(a);work=shapes(a);directory=D/a/'screen';directory.mkdir(parents=True,exist_ok=False)
  p=plan(a,work,r['parent_attempt_id'],False,directory);save(directory/'manifest.json',p)
  descriptions.append(dict(attempt=a,parent=r['parent_attempt_id'],source=r['candidate_sha'],label=r['label'],workloads=work,planned_work=p['planned_work']))
 save(D/'plan.json',dict(status='frozen_before_measurement',created_unix=time.time(),attempts=descriptions,jobs=sum(x['planned_work']['jobs'] for x in descriptions),estimated_seconds=sum(x['planned_work']['estimated_seconds'] for x in descriptions),selection='All new screen points with positive mean offload and gap reductions and positive GPU savings, plus applicable historical batch16 screen points meeting the same sign rule. No effect-size cutoff, top-K selection or repeated screening.',confirmation='Exactly three fresh balanced blocks per selected workload against historical parent; additionally against A0016 if different. Screen observations excluded from inference. No extension until favorable.',multiple_testing='Holm family-wise correction at 0.05 across all parent and current comparisons; p=max(one-sided paired t latency p, gap p), n=3. Also require both nominal two-sided 95% lower bounds positive, no resolved resident slowdown, positive GPU savings. Only local performance evidence, never automatic acceptance.',scope='18 nonretained frozen implementations; pipeline prefill uncovered shapes, high-context decode control, applicable generation; tiny bulk uncovered decode and generation. Historical verdicts and final implementation unchanged. Not an exhaustive search outside the campaign matrix.'))
 print(read(D/'plan.json')['jobs'],read(D/'plan.json')['estimated_seconds'])

def screen_all():
 assert not (D/'screen-controller.json').exists()
 state=dict(status='running',pid=os.getpid(),started_unix=time.time(),finished=[]);save(D/'screen-controller.json',state)
 for a in ATTEMPTS:
  state['active_attempt']=a;save(D/'screen-controller.json',state);run(D/a/'screen');summarize(D/a/'screen');state['finished'].append(a);save(D/'screen-controller.json',state)
 state.update(status='completed',finished_unix=time.time(),active_attempt=None);save(D/'screen-controller.json',state)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','screen','summarize']);p.add_argument('--directory',type=Path);a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='screen':screen_all()
 else:summarize(a.directory)
