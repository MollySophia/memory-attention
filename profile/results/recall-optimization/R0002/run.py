"""R0002: three-way full matrix; each placement occupies each order position once."""
import ast,hashlib,json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent
BASE=('/home/molly/workspace-memory-attn/recall-bench-head','e5b52be0be5484a6f8d0c4391b158cb71e561759')
NEW=('/home/molly/workspace-memory-attn/recall-runtime-head','b7c33a6ef3679871d88871e528a35f7131ee2597')
PLACEMENTS={'full_kv':dict(root=BASE[0],commit=BASE[1],recall=False),'initial_recall':dict(root=BASE[0],commit=BASE[1],recall=True),'fused_recall':dict(root=NEW[0],commit=NEW[1],recall=True)}
def read(f):return json.loads(f.read_text())
def save(f,v):f.write_text(json.dumps(v,indent=2)+'\n')
for root,sha in (BASE,NEW):
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==sha
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=root)
assert not (D/'manifest.json').exists()
jobs=[]
for b in (1,8):
 for label,source in PLACEMENTS.items():
  name=f'correctness-{label}-b{b}'
  cmd=[sys.executable,str(D/'correctness.py'),'--source-root',source['root'],'--variant','ma_gpu_unfolded','--batch-size',str(b),'--seed','1234','--output',str(D/(name+'.json'))]
  if source['recall']:cmd+=['--memory-recall']
  jobs.append(dict(name=name,kind='correctness',placement=label,batch=b,command=cmd,status='pending'))
workloads=[(mode,b,l) for b,l in [(1,2048),(4,2048),(8,2048),(16,2048),(8,512),(8,4096),(8,8192),(32,2048),(64,2048)] for mode in ('prefill','decode')]+[('generation',1,2048),('generation',8,2048)]
labels=list(PLACEMENTS)
for block in range(3):
 for mode,b,l in (workloads if block%2==0 else list(reversed(workloads))):
  for label in labels[block:]+labels[:block]:
   name=f'B{block+1}-{label}-{mode}-b{b}-l{l}'
   cmd=[sys.executable,str(D/'bench_fla.py'),'--mode',mode,'--variants','ma_gpu_unfolded','--batch-size',str(b),'--seq-len',str(l),'--context-len',str(l),'--num-layers','24','--hidden-size','2048','--num-heads','32','--num-kv-heads','32','--intermediate-size','5632','--vocab-size','32000','--logits-to-keep','1','--generation-steps','128','--seed','1234','--stage','generation_validation' if mode=='generation' else 'confirmation','--warmup','2' if mode=='generation' else '10','--repeats','5' if mode=='generation' else '10','--rounds','3','--json',str(D/(name+'.json'))]
   if PLACEMENTS[label]['recall']:cmd+=['--memory-recall']
   jobs.append(dict(name=name,kind='performance',block=block+1,mode=mode,batch=b,length=l,placement=label,command=cmd,status='pending'))
p=dict(status='planned',started_unix=time.time(),sources=PLACEMENTS,blocks=3,regressions='84 passed',scope='20 workloads, no RoPE, unfolded GPU tables, same seeded 2.8365B BF16 model. Current controls rerun; do not use previous environment timings as paired controls.',statistics='Three paired process-block means; two-sided Student-t 95% CI df2. Rotate the 3 placements by block; each placement occupies each position once for EVERY workload.',failure_policy='Keep OOMs; skip identical OOM repeats only. Stop on other failures or correctness mismatch. No stopping for slow timings.',environment_note='Unrelated process holds about 6994 MiB on GPU, vs about654 MiB in original baseline; never terminate it. Capacity results belong to current environment.',harness_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in D.glob('*.py')},jobs=jobs)
save(D/'manifest.json',p)
old=ast.parse((D.parents[1]/'recall-no-rope-001/bench_fla.py').read_text());new=ast.parse((D/'bench_fla.py').read_text())
for name in ('build','rollback','run_prefill','run_decode','run_generation','measure'):
 a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(a)==ast.dump(b)
save(D/'harness-audit.json',dict(status='passed',note='Build, rollback and all timing routines AST-identical to initial recall study.'))
p['status']='running';save(D/'manifest.json',p);oom=set()
for j in jobs:
 key=(j.get('mode'),j['batch'],j.get('length'),j['placement']);source=PLACEMENTS[j['placement']]
 if key in oom:j.update(status='not_run',reason='Prior identical cell OOM');save(D/'manifest.json',p);continue
 if j['kind']=='performance' and not (D/'correctness-audit.json').exists():
  for b in (1,8):
   ref=read(D/f'correctness-full_kv-b{b}.json')
   for label in ('initial_recall','fused_recall'):
    rec=read(D/f'correctness-{label}-b{b}.json');assert len(ref['checkpoints'])==len(rec['checkpoints'])==129
    for a,c in zip(ref['checkpoints'],rec['checkpoints'],strict=True):
     assert a['logits']==c['logits'] and c['logits']['finite']
     if 'hidden_states' in a:
      assert a['hidden_states']==c['hidden_states']
      for kv,ki in zip(a['kv_states'],c['kv_states'],strict=True):assert kv[0]==ki[0] and ki[1]['dtype']=='torch.int64'
   assert read(D/f'correctness-initial_recall-b{b}.json')['checkpoints']==read(D/f'correctness-fused_recall-b{b}.json')['checkpoints']
  save(D/'correctness-audit.json',dict(status='passed',batches=[1,8],steps=129,comparison='Exact logits at129 steps, all hidden and raw keys at prefix/1/2/128. Initial/fused recall fingerprints identical, including cached IDs.'))
 j.update(status='running',started_unix=time.time());save(D/'manifest.json',p);print('START',j['name'],flush=True)
 env=os.environ.copy();env['MA_SOURCE_ROOT']=source['root']
 with (D/(j['name']+'.txt')).open('w') as log:proc=subprocess.run(j['command'],cwd=source['root'],env=env,stdout=log,stderr=subprocess.STDOUT)
 j.update(returncode=proc.returncode,finished_unix=time.time());f=D/(j['name']+'.json');raw=read(f) if f.exists() else {};j['status']=raw.get('status','failed')
 if j['status']=='oom' and j['kind']=='performance':oom.add(key)
 elif proc.returncode or j['status']!='completed':p['status']='stopped';save(D/'manifest.json',p);raise RuntimeError(j['name'])
 else:
  assert raw['model_config']['rope_theta'] is None and raw['model_config']['memory_recall']==source['recall']
  assert raw['source']['git_commit']['stdout'].strip()==source['commit']
 save(D/'manifest.json',p);print('END',j['name'],j['status'],flush=True)
p.update(status='completed',finished_unix=time.time());save(D/'manifest.json',p)
