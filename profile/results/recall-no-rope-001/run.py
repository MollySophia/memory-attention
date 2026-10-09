"""Frozen full-matrix comparison of full KV vs no-RoPE recall; no implicit retries."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
D=Path(__file__).resolve().parent
ROOT=Path('/home/molly/workspace-memory-attn/recall-bench-head')
SHA=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert SHA.startswith('e5b52be0')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert not (D/'manifest.json').exists()
jobs=[]
for b in (1,8):
 for recall in (False,True):
  label='recall' if recall else 'full_kv';name=f'correctness-{label}-b{b}'
  cmd=[sys.executable,str(D/'correctness.py'),'--source-root',str(ROOT),'--variant','ma_gpu_unfolded','--batch-size',str(b),'--seed','1234','--output',str(D/(name+'.json'))]
  if recall:cmd+=['--memory-recall']
  jobs.append(dict(name=name,kind='correctness',recall=recall,batch=b,command=cmd,status='pending'))
workloads=[(mode,b,l) for b,l in [(1,2048),(4,2048),(8,2048),(16,2048),(8,512),(8,4096),(8,8192),(32,2048),(64,2048)] for mode in ('prefill','decode')]+[('generation',1,2048),('generation',8,2048)]
for block in range(1,4):
 for wi,(mode,b,l) in enumerate(workloads if block%2 else list(reversed(workloads))):
  for recall in ((False,True) if (block+wi)%2 else (True,False)):
   label='recall' if recall else 'full_kv';name=f'B{block}-{label}-{mode}-b{b}-l{l}'
   warmup,repeats=(2,5) if mode=='generation' else (10,10)
   cmd=[sys.executable,str(D/'bench_fla.py'),'--mode',mode,'--variants','ma_gpu_unfolded','--batch-size',str(b),'--seq-len',str(l),'--context-len',str(l),'--num-layers','24','--hidden-size','2048','--num-heads','32','--num-kv-heads','32','--intermediate-size','5632','--vocab-size','32000','--logits-to-keep','1','--generation-steps','128','--seed','1234','--stage','generation_validation' if mode=='generation' else 'confirmation','--warmup',str(warmup),'--repeats',str(repeats),'--rounds','3','--json',str(D/(name+'.json'))]
   if recall:cmd+=['--memory-recall']
   jobs.append(dict(name=name,kind='performance',block=block,mode=mode,batch=b,length=l,recall=recall,command=cmd,status='pending'))
p=dict(status='planned',source_root=str(ROOT),source_commit=SHA,started_unix=time.time(),blocks=3,model='24 layers, 2048 hidden, 32 heads, 5632 intermediate, vocab 32000, 2.8365B BF16; QK norm false; RoPE disabled in BOTH modes; unfolded GPU tables; seed1234',comparison='Only memory_recall changes; same weights and timing routines. 20 workloads: original matrix plus batch32/64 and explicit 128-step generation.',statistics='Mean of three independent process means; paired two-sided 95% Student-t df=2. Order alternates by workload and block; no blanket equivalence claim.',failure_policy='Retain all OOMs; skip only identical OOM repeats; stop on any other failure or correctness mismatch.',harness_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in D.glob('*.py')},jobs=jobs)
save(D/'manifest.json',p)
# Only configuration and CLI changes are permitted in the established harness.
import ast
old=ast.parse((D.parent/'pr-offload-001/bench_fla.py').read_text());new=ast.parse((D/'bench_fla.py').read_text())
for name in ('rollback','run_prefill','run_decode','run_generation','measure'):
 a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
 b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(a)==ast.dump(b),name
save(D/'harness-audit.json',dict(status='passed',unchanged_timing_functions=['rollback','run_prefill','run_decode','run_generation','measure']))
p['status']='running';save(D/'manifest.json',p);oom=set()
for j in jobs:
 key=(j.get('mode'),j['batch'],j.get('length'),j['recall'])
 if key in oom:j.update(status='not_run',reason='Prior identical cell OOM');save(D/'manifest.json',p);continue
 if j['kind']=='performance' and not (D/'correctness-audit.json').exists():
  for b in (1,8):
   ref=read(D/f'correctness-full_kv-b{b}.json');rec=read(D/f'correctness-recall-b{b}.json')
   assert len(ref['checkpoints'])==len(rec['checkpoints'])==129
   for a,c in zip(ref['checkpoints'],rec['checkpoints'],strict=True):
    assert a['logits']==c['logits'] and c['logits']['finite']
    if 'hidden_states' in a:
     assert a['hidden_states']==c['hidden_states']
     for kv,ki in zip(a['kv_states'],c['kv_states'],strict=True):
      assert kv[0]==ki[0] and ki[1]['dtype']=='torch.int64' and ki[1]['shape'][-1]==1
  save(D/'correctness-audit.json',dict(status='passed',batches=[1,8],steps=129,comparison='Exact logits at every step, exact all hidden states and raw keys at prefix/steps1,2,128. Recall stores int64 token IDs instead of V.'))
 j.update(status='running',started_unix=time.time());save(D/'manifest.json',p);print('START',j['name'],flush=True)
 env=os.environ.copy();env['MA_SOURCE_ROOT']=str(ROOT)
 with (D/(j['name']+'.txt')).open('w') as log:proc=subprocess.run(j['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 j.update(returncode=proc.returncode,finished_unix=time.time());path=D/(j['name']+'.json');raw=read(path) if path.exists() else {};j['status']=raw.get('status','failed')
 if j['status']=='oom' and j['kind']=='performance':oom.add(key)
 elif proc.returncode or j['status']!='completed':p['status']='stopped';save(D/'manifest.json',p);raise RuntimeError(j['name'])
 else:
  assert raw['model_config']['rope_theta'] is None and raw['model_config']['memory_recall']==j['recall']
  assert raw['source']['git_commit']['stdout'].strip()==SHA
 save(D/'manifest.json',p);print('END',j['name'],j['status'],flush=True)
p.update(status='completed',finished_unix=time.time());save(D/'manifest.json',p)
