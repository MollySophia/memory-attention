"""Frozen, serial upstream/main vs PR-head comparison. No implicit retries."""
import hashlib,json,os,subprocess,sys,time,csv,statistics
from pathlib import Path
D=Path(__file__).resolve().parent
PYTHON=sys.executable
SOURCES={'main':('/home/molly/workspace-memory-attn/pr-offload-main','66bd7c6326a6a7badf01e7ef5a14ff967fb8000c'), 'head':('/home/molly/workspace-memory-attn/pr-offload-head','9925991f02502409c0bc838ee0d8c6e70ba878bc')}
PLACEMENTS=[('main','ma_gpu_unfolded'),('head','ma_gpu'),('head','ma_offload')]
WORKLOADS=[(m,b,l) for b,l in [(1,2048),(4,2048),(8,2048),(16,2048),(8,512),(8,4096),(8,8192)] for m in ('prefill','decode')]+[('generation',1,2048),('generation',8,2048)]+[(m,b,2048) for b in (32,64) for m in ('prefill','decode')]
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
def verify_sources():
 for root,sha in SOURCES.values():
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==sha
  assert not subprocess.check_output(['git','status','--porcelain'],cwd=root)
def checksum():return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(D.glob('*.py')) if p.name in ('bench_fla.py','benchmark_telemetry.py','sampling_plan.py','correctness.py')}
def plan():
 assert not (D/'manifest.json').exists();verify_sources();jobs=[]
 for batch in (1,8):
  for source,variant in PLACEMENTS:
   name=f'correctness-{source}-{variant}-b{batch}'
   jobs.append(dict(name=name,kind='correctness',source=source,variant=variant,batch=batch,status='pending',command=[PYTHON,str(D/'correctness.py'),'--source-root',SOURCES[source][0],'--variant',variant,'--batch-size',str(batch),'--seed','1234','--output',str(D/(name+'.json'))]))
 for block in range(3):
  workloads=WORKLOADS if block%2==0 else list(reversed(WORKLOADS))
  for wi,(mode,batch,length) in enumerate(workloads):
   # Each placement occupies each ordering position once at every workload.
   order=PLACEMENTS[block:]+PLACEMENTS[:block]
   for source,variant in order:
    name=f'B{block+1}-{source}-{variant}-{mode}-b{batch}-l{length}'
    stage='generation_validation' if mode=='generation' else 'confirmation'
    warmup,repeats=(2,5) if mode=='generation' else (10,10)
    command=[PYTHON,str(D/'bench_fla.py'),'--mode',mode,'--variants',variant,'--batch-size',str(batch),'--seq-len',str(length),'--context-len',str(length),'--seed','1234','--num-layers','24','--hidden-size','2048','--num-heads','32','--num-kv-heads','32','--intermediate-size','5632','--vocab-size','32000','--logits-to-keep','1','--generation-steps','128','--stage',stage,'--warmup',str(warmup),'--repeats',str(repeats),'--rounds','3','--json',str(D/(name+'.json'))]
    jobs.append(dict(name=name,kind='performance',source=source,variant=variant,block=block+1,mode=mode,batch=batch,length=length,status='pending',command=command))
 old=read(D.parent/'optimization/offload-gap-001/A0030/gap-diagnostics/gaps.json')['rows']
 estimates={ (r['mode'],r['batch'],r['length']):r['offload_ms'] for r in old }
 seconds=6*60
 for j in jobs[6:]:
  key=(j['mode'],j['batch'],j['length']);latency=estimates.get(key,estimates.get((j['mode'],16,2048),1000)*j['batch']/16)
  seconds+=20+latency*(17 if j['mode']=='generation' else 40)/1000
 p=dict(study='pr-offload-001',status='planned',sources=SOURCES,harness_sha256=checksum(),blocks=3,seed=1234,planned_performance_processes=180,planned_correctness_processes=6,estimated_seconds=seconds,estimation='Historical latency plus 20s/process setup and 60s/correctness; OOM cells may shorten execution.',scope='All original 16 workloads plus batch32/64 prefill/decode at 2048. Main native GPU vs head folded GPU vs head CPU offload.',statistics='Three paired process-block means; two-sided 95% Student-t intervals df=2. Per-workload comparisons, no blanket equivalence claim.',failure_policy='Preserve each OOM; skip identical cell repeats only. Stop on other failures or correctness mismatch. No early stopping for unfavorable latency.',jobs=jobs)
 save(D/'manifest.json',p);print('Planned',len(jobs),'processes; estimated minutes',round(seconds/60,1))
def audit(j,raw):
 assert raw['status']=='completed'
 assert raw['source']['git_commit']['stdout'].strip()==SOURCES[j['source']][1]
 assert Path(raw['env']['model_module']).resolve()==Path(SOURCES[j['source']][0])/'fla/models/memory/modeling_memory.py'
 if j['kind']=='performance':
  cfg=raw['config'];r=raw['results'][0]
  for key,value in dict(mode=j['mode'],batch_size=j['batch'],seq_len=j['length'],context_len=j['length'],variants=[j['variant']],seed=1234,logits_to_keep=1,hidden_size=2048,num_layers=24,num_heads=32,num_kv_heads=32,intermediate_size=5632,vocab_size=32000,generation_steps=128).items():assert cfg[key]==value
  assert cfg['rounds']==3 and len(r['samples_ms'])==3
  assert all(len(x)==cfg['repeats'] for x in r['samples_ms'])
  assert abs(statistics.mean([x for rnd in r['samples_ms'] for x in rnd])-r['mean_ms'])<1e-8
  assert r['output_scope']=='cached_logits'
  assert r['memory_after']['cpu_table_bytes']==(3000*2**20 if j['variant']=='ma_offload' else 0)
  assert raw['model_config']['num_hidden_layers']==24
 else:
  assert len(raw['checkpoints'])==129
  assert all(x['logits']['finite'] for x in raw['checkpoints'])
def compare_correctness():
 for b in (1,8):
  rows=[read(D/f'correctness-{source}-{variant}-b{b}.json') for source,variant in PLACEMENTS]
  for row in rows[1:]:assert row['checkpoints']==rows[0]['checkpoints'],f'Full-model correctness mismatch at batch {b}'
 save(D/'correctness-audit.json',dict(status='passed',batches=[1,8],seed=1234,steps=129,comparison='Main resident vs head folded resident vs head offload; exact logits every step and exact all hidden/KV at prefix/steps1,2,128'))
def run():
 p=read(D/'manifest.json');assert p['status'] in ('planned','audited_resume');verify_sources();assert p['harness_sha256']==checksum()
 p.update(status='running',controller_pid=os.getpid());save(D/'manifest.json',p);oom=set()
 for j in p['jobs']:
  key=(j.get('mode'),j['batch'],j.get('length'),j['source'],j['variant'])
  if j['status']=='completed':audit(j,read(D/(j['name']+'.json')));continue
  if j['status']=='oom':oom.add(key);continue
  if key in oom:j.update(status='not_run',reason='Prior identical cell OOM');save(D/'manifest.json',p);continue
  assert j['status']=='pending'
  if j['kind']=='performance' and not (D/'correctness-audit.json').exists():compare_correctness()
  print('START',j['name'],flush=True);j.update(status='running',started_unix=time.time());env=os.environ.copy();env['MA_SOURCE_ROOT']=SOURCES[j['source']][0]
  with (D/(j['name']+'.txt')).open('w') as f:
   child=subprocess.Popen(j['command'],cwd=SOURCES[j['source']][0],env=env,stdout=f,stderr=subprocess.STDOUT);j['pid']=child.pid;save(D/'manifest.json',p);code=child.wait()
  j.update(returncode=code,finished_unix=time.time());path=D/(j['name']+'.json');raw=read(path) if path.exists() else {};j['status']=raw.get('status','failed')
  if j['status']=='oom' and j['kind']=='performance':oom.add(key)
  elif code or j['status']!='completed':p['status']='stopped_needs_review';save(D/'manifest.json',p);raise RuntimeError(j['name'])
  else:audit(j,raw)
  save(D/'manifest.json',p);print('END',j['name'],j['status'],flush=True)
 p.update(status='completed',finished_unix=time.time());save(D/'manifest.json',p)
if __name__=='__main__':globals()[sys.argv[1]]()
