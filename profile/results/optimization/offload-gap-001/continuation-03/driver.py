"""Complete-matrix continuation driver. Freeze plans before timings; never restart implicitly."""
import argparse,csv,hashlib,json,math,os,statistics,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent; C=D.parent
sys.path.insert(0,str(C))
from run_paired import source_hash,save,implementation_order
from audit_continuation import validate_offload_memory
from analyze_paired import interval as legacy_interval
SHAPES=[(1,2048),(4,2048),(8,2048),(16,2048),(8,512),(8,4096),(8,8192)]
WORKLOADS=[(m,b,l) for b,l in SHAPES for m in ('prefill','decode')]+[('generation',b,2048) for b in (1,8)]
def read(p):return json.loads(Path(p).read_text())
def root(a):return Path('/home/molly/workspace-memory-attn')/('offload-gap-001-'+a)
def record(a):return read(C/a/'record.json')
def confirmation_blocks(a):
 n=record(a).get('confirmation_blocks',3)
 assert type(n) is int and n>=3,'Predeclare at least three independent blocks'
 return n

def planned_blocks(p):
 n=p.get('independent_blocks',3 if p['formal'] else 1)
 assert type(n) is int and n>=(3 if p['formal'] else 1)
 if p['formal']:assert n==confirmation_blocks(p['attempt']),'Frozen plan/record block count mismatch'
 return n

def validate_balanced_order(jobs,blocks=3):
 assert type(blocks) is int and blocks>=3
 if not jobs:return True
 assert {j['block'] for j in jobs}==set(range(1,blocks+1))
 keys={(j['batch'],j['length'],j['mode']) for j in jobs}
 for key in keys:
  placements=[];orders={v:[] for v in ('ma_offload','ma_gpu')}
  for block in range(1,blocks+1):
   selected=[j for j in jobs if (j['batch'],j['length'],j['mode'])==key and j['block']==block]
   assert len(selected)==4,'Each block requires four unique matched cells'
   variants=list(dict.fromkeys(j['variant'] for j in selected))
   assert set(variants)==set(orders)
   placements.append(variants)
   for v in orders:
    order=[j['implementation'] for j in selected if j['variant']==v]
    assert len(order)==2 and set(order)=={'baseline','candidate'}
    orders[v].append(order)
  for sequence in [placements,*orders.values()]:
   assert all(current==list(reversed(previous)) for previous,current in zip(sequence,sequence[1:])),(key,sequence)
 return True

def interval(values):
 if len(values)==3:return legacy_interval(values)
 from scipy.stats import t
 n=len(values);assert n>=3
 mean=statistics.mean(values);half=float(t.ppf(.975,n-1))*statistics.stdev(values)/(n**.5)
 return dict(values=values,mean=mean,lower_95=mean-half,upper_95=mean+half,
             method=f'paired process-block mean +/- t(df={n-1},0.975)*SE',n=n)

def signature(a):
 r=root(a);sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip()
 assert sha==record(a)['candidate_sha']
 assert not subprocess.check_output(['git','status','--porcelain','--','fla',*map(str,Path('profile').glob('*.py'))],cwd=r)
 return dict(attempt=a,root=str(r),commit=sha,source_sha256=source_hash(r))
def cmd(a,shape,path,formal):
 mode,b,l=shape;stage='generation_validation' if mode=='generation' else 'confirmation' if formal else 'screening'
 w,n,r=((2,5,3) if formal else (2,3,1)) if mode=='generation' else (10,10,3) if formal else (3,5,1)
 return [sys.executable,str(root(a)/'profile/bench_fla.py'),'--mode',mode,'--variants','PLACEHOLDER','--batch-size',str(b),'--seq-len',str(l),'--context-len',str(l),'--num-layers','24','--hidden-size','2048','--num-heads','32','--num-kv-heads','32','--intermediate-size','5632','--vocab-size','32000','--seed','1234','--stage',stage,'--warmup',str(w),'--repeats',str(n),'--rounds',str(r),'--logits-to-keep','1','--prefill-workload','inference','--generation-steps','128','--policy','auto','--group-size','1','--prefetch-depth','4','--json',str(path)]
def estimate(a,shape,variant,formal):
 m,b,l=shape
 screen=read(C/'A0016/R01/manifest.json')
 ref=[j for j in screen['jobs'] if j['variant']==variant and j['mode']==('prefill' if m=='generation' else m)]
 j=min(ref,key=lambda j:abs(j['batch']-b));row=read(C/'A0016/R01'/(j['name']+'.json'))['results'][0]
 setup=max(0,j['finished_unix']-j['started_unix']-8*row['mean_ms']/1000)
 # Bound anomalously paused historical setup with the campaign's observed ~21 s cost.
 setup=min(setup,30)
 latency=row['mean_ms']*(b/j['batch'])*(l/2048 if m!='decode' else max(1,l/2048))
 if m=='generation':
  dj=next(x for x in screen['jobs'] if x['variant']==variant and x['mode']=='decode' and x['batch']==b)
  latency+=128*read(C/'A0016/R01'/(dj['name']+'.json'))['results'][0]['mean_ms']
 return setup+latency/1000*((17 if formal else 5) if m=='generation' else 40 if formal else 8)
def plan(a,workloads,baseline,formal,directory):
 sig={i:signature(x) for i,x in [('baseline',baseline),('candidate',a)]};jobs=[]
 blocks=confirmation_blocks(a) if formal else 1
 for block in range(blocks):
  sequence=list(enumerate(workloads));sequence=sequence[::-1] if block%2 else sequence
  for wi,shape in sequence:
   variants=['ma_offload','ma_gpu'];variants=variants[::-1] if (wi+block)%2 else variants
   for v in variants:
    for impl in implementation_order(block,wi,('ma_offload','ma_gpu').index(v)):
     aid=sig[impl]['attempt'];m,b,l=shape;name=f'B{block+1}-J{len(jobs)+1:03d}-{impl}-{m}-{v}-b{b}-l{l}'
     command=cmd(aid,shape,directory/(name+'.json'),formal);command[command.index('PLACEHOLDER')]=v
     jobs.append(dict(name=name,block=block+1,mode=m,batch=b,length=l,variant=v,implementation=impl,attempt=aid,candidate_sha=sig[impl]['commit'],source_sha256=sig[impl]['source_sha256'],cwd=sig[impl]['root'],command=command,estimated_seconds=estimate(aid,shape,v,formal),status='pending'))
 if formal:validate_balanced_order(jobs,blocks)
 return dict(campaign_id='offload-gap-001',study='continuation-03',attempt=a,baseline=baseline,stage='independent_confirmation' if formal else 'complete_matrix_screen',formal=formal,independent_blocks=blocks,status='planned',source_signatures=sig,jobs=jobs,planned_work=dict(jobs=len(jobs),estimated_seconds=sum(j['estimated_seconds'] for j in jobs)),accepted=False)
def audit_job(j,path):
 p=read(path);assert p['status']=='completed';assert p['source']['source_sha256']==j['source_sha256'];assert p['source']['git_commit']['stdout'].strip()==j['candidate_sha']
 cfg=p['config'];row=p['results'][0]
 for k,v in dict(mode=j['mode'],batch_size=j['batch'],seq_len=j['length'],context_len=j['length'],variants=[j['variant']],seed=1234,logits_to_keep=1,hidden_size=2048,num_layers=24,num_heads=32,num_kv_heads=32,intermediate_size=5632,vocab_size=32000,policy='auto',prefetch_depth=4,group_size=1,generation_steps=128).items():assert cfg[k]==v,(k,cfg[k],v)
 for key in ('warmup','repeats','rounds'):assert cfg[key]==int(j['command'][j['command'].index('--'+key)+1])
 assert len(row['samples_ms'])==cfg['rounds'] and all(len(x)==cfg['repeats'] for x in row['samples_ms'])
 assert all(math.isfinite(x) and x>0 for rnd in row['samples_ms'] for x in rnd)
 expected=(('generation_v1_w2_n5_r3' if cfg['rounds']==3 else 'generation_v1_w2_n3_r1_custom') if j['mode']=='generation' else 'screen_v1_w3_n5_r1' if cfg['warmup']==3 else 'formal_v1_w10_n10_r3')
 assert p['measurement_plan_id']==expected
 assert row['output_scope']=='cached_logits'
 recorded_root=Path(j['cwd']).resolve() if j.get('cwd') else Path(j['command'][1]).resolve().parents[1]
 assert Path(p['env']['model_module']).resolve()==recorded_root/'fla/models/memory/modeling_memory.py'
 for key,value in dict(hidden_size=2048,num_hidden_layers=24,num_heads=32,num_kv_heads=32,intermediate_size=5632,vocab_size=32000,qk_norm=False,use_gate=False,fuse_norm=False,tie_word_embeddings=False).items():assert p['model_config'][key]==value
 if j['variant']=='ma_offload':
  memory=row['memory_after'];model_cfg=p['model_config']
  expected_cfg=record(j['attempt']).get('expected_offload_config',{})
  for field,value in expected_cfg.items():assert model_cfg[field]==value
  mapped=model_cfg.get('memory_offload_mapped_bulk',False)
  # Continuation policies are configuration-driven; historical A0013 fixed8..16
  # audit is not valid for an expanded mapped range. Count every allocation.
  assert memory['cpu_table_bytes']==3000*2**20
  table_pinned=memory.get('cpu_table_pinned_bytes',0)
  assert table_pinned==(3000*2**20 if mapped else 0)
  assert memory['offload_gpu_buffer_bytes']==sum(c['gpu_bytes'] for c in memory['offloader_capacities'])
  assert memory['offload_pinned_bytes']==table_pinned+sum(c['host_bytes'] for c in memory['offloader_capacities'])
  for cap in memory['offloader_capacities']:
   tokens=cap['batch']*cap['length'];depth=1 if tokens<=model_cfg.get('memory_offload_single_slot_max_tokens',0) else 4
   assert cap['policy']==('bulk' if tokens<=model_cfg['memory_offload_bulk_max_tokens'] else 'pipeline')
   expected=tokens*2048*2*(24 if cap['policy']=='bulk' else depth)
   mapped_shape=mapped and cap['policy']=='bulk' and model_cfg['memory_offload_mapped_bulk_min_tokens']<=tokens<=model_cfg['memory_offload_mapped_bulk_max_tokens']
   shared_host=(cap['policy']=='pipeline' and
                model_cfg.get('memory_offload_single_host_min_tokens',1) <= tokens <=
                model_cfg.get('memory_offload_single_host_max_tokens',0))
   expected_host=0 if mapped_shape else tokens*2048*2 if shared_host else expected
   assert cap['gpu_bytes']==expected and cap['host_bytes']==expected_host
 env=p['environment_before'];e=p['env']
 return (e['torch'],e['torch_cuda'],e['flash_attn'],e['gpu'],e['python'],env['gpu_telemetry']['stdout'].splitlines()[1].split(', ')[1],tuple(env['cpu_affinity']),env['torch_threads'],env['torch_interop_threads'],json.dumps(env['thread_environment'],sort_keys=True)),row

def run(directory,resume=False):
 path=directory/'manifest.json';p=read(path)
 if resume:
  assert p['status']=='stopped_needs_review'
  assert all(j['status'] in ('completed','pending') for j in p['jobs'])
  for j in p['jobs']:
   if j['status']=='completed':
    assert j['returncode']==0
    audit_job(j,directory/(j['name']+'.json'))
  p.setdefault('recovery_history',[]).append(dict(previous_controller_pid=p.get('controller_pid'),previous_error=p.get('error'),resumed_unix=time.time(),reason='Explicit audited recovery; retain completed raw jobs and execute pending only'))
  p.pop('error',None)
 else:assert p['status']=='planned'
 # Parent-comparison plans carry `formal`; final target plans have their own
 # per-workload pair validator and no baseline/candidate four-cell blocks.
 if p.get('formal',False):validate_balanced_order(p['jobs'],planned_blocks(p))
 for s in p['source_signatures'].values():assert signature(s['attempt'])==s
 p.update(status='running',controller_pid=os.getpid(),started_unix=time.time());save(path,p)
 try:
  for j in p['jobs']:
   if resume and j['status']=='completed':continue
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
 blocks=planned_blocks(p)
 if p['formal']:validate_balanced_order(p['jobs'],blocks)
 for j in p['jobs']:
  env,r=audit_job(j,directory/(j['name']+'.json'));envs.add(env);data[j['mode'],j['batch'],j['length'],j['block'],j['implementation'],j['variant']]=r
 assert len(envs)==1,'Environment mismatch in paired results'
 from scipy.stats import t
 for shape in sorted({k[:3] for k in data}):
  values={k:[] for k in ('offload_reduction_ms','gpu_reduction_ms','gap_reduction_ms','offload_speedup')};memory=[];means=[]
  for block in range(1,blocks+1):
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
    return float(t.sf(mu/(sd/(len(xs)**.5)),len(xs)-1)) if sd else (0.0 if mu>0 else 1.0)
   row['joint_one_sided_p']=max(pv(values['offload_reduction_ms']),pv(values['gap_reduction_ms']))
   row['nominal_dual_gain']=all(stat[k]['lower_95']>0 for k in ('offload_reduction_ms','gap_reduction_ms'))
   row['resolved_resident_slowdown']=stat['gpu_reduction_ms']['upper_95']<0
  rows.append(row)
 result=dict(status='audited',formal=p['formal'],rows=rows,checked_raw_results=len(p['jobs']),source_manifest=str(directory/'manifest.json'))
 save(directory/'summary.json',result);return result


def prepare(a):
 rec=record(a);directory=C/a/'R01-complete-screen';directory.mkdir(exist_ok=False)
 p=plan(a,WORKLOADS,rec['parent_attempt_id'],False,directory)
 assert len(p['jobs'])==64
 save(directory/'manifest.json',p);print(json.dumps(p['planned_work']))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','resume','summarize']);parser.add_argument('--attempt',required=True);args=parser.parse_args()
 directory=C/args.attempt/'R01-complete-screen'
 if args.action=='prepare':prepare(args.attempt)
 elif args.action in ('run','resume'):run(directory,resume=args.action=='resume');summarize(directory)
 else:summarize(directory)
