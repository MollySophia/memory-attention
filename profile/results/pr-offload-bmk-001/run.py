"""Run frozen, unmodified main bmk.py. Compare with existing fresh PR evidence."""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
D=Path(__file__).resolve().parent
ROOT=Path('/home/molly/workspace-memory-attn/pr-offload-main')
SHA='66bd7c6326a6a7badf01e7ef5a14ff967fb8000c'
def save(v): (D/'manifest.json').write_text(json.dumps(v,indent=2)+'\n')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==SHA
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT)
assert not (D/'manifest.json').exists()
shapes=[(b,2048) for b in (1,4,8,16)]+[(8,l) for l in (512,4096,8192)]+[(b,2048) for b in (32,64)]
jobs=[]
for block in range(1,4):
 for b,l in (shapes if block%2 else list(reversed(shapes))):
  for mode in ('prefill','decode'):
   name=f'B{block}-{mode}-b{b}-l{l}'
   cmd=[sys.executable,str(ROOT/'profile/bmk.py'),'--variants','ma_offload','--mode',mode,'--batch-size',str(b),'--seq-len',str(l),'--context-len',str(l),'--num-layers','24','--hidden-size','2048','--num-heads','32','--num-kv-heads','32','--intermediate-size','5632','--vocab-size','32000','--norm-eps','1e-6','--logits-to-keep','1','--cpu-threads','16','--seed','1234','--group-size','1','--prefetch-depth','4','--decode-offload','bulk','--warmup','10','--repeats','10','--rounds','3','--timing','latency','--json',str(D/(name+'.json'))]
   jobs.append(dict(name=name,block=block,batch=b,length=l,mode=mode,command=cmd,status='pending'))
p=dict(status='running',started_unix=time.time(),source_root=str(ROOT),source_commit=SHA,source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'profile/bmk.py',ROOT/'profile/ma_profile.py')},reference='../pr-offload-001',scope='Direct original bmk offload vs existing fresh PR offload; separate studies, not paired timing. Original static KV and preavailable CPU IDs preserved. No generation mode in original.',blocks=3,jobs=jobs)
p['environment']={k:subprocess.check_output(cmd,text=True) for k,cmd in {'gpu':['nvidia-smi'],'cpu':['lscpu']}.items()}
p['cpu_affinity']=sorted(os.sched_getaffinity(0));save(p)
oom=set()
for j in jobs:
 key=(j['mode'],j['batch'],j['length'])
 if key in oom:
  j.update(status='not_run',reason='Identical cell previously OOM');save(p);continue
 j.update(status='running',started_unix=time.time());save(p);print('START',j['name'],flush=True)
 with (D/(j['name']+'.txt')).open('w') as log:
  proc=subprocess.run(j['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 j.update(returncode=proc.returncode,finished_unix=time.time())
 if proc.returncode:
  log=(D/(j['name']+'.txt')).read_text()
  if 'OutOfMemoryError' in log or 'CUDA out of memory' in log:j['status']='oom';oom.add(key)
  else:j['status']='failed';p['status']='stopped';save(p);raise RuntimeError(j['name'])
 else:
  raw=json.loads((D/(j['name']+'.json')).read_text());r=raw['results'][0]
  assert len(raw['results'])==1 and r['mode']==j['mode'] and r['variant']=='ma_offload' and len(r['round_ms'])==3
  j['status']='completed'
 save(p);print('END',j['name'],j['status'],flush=True)
p.update(status='completed',finished_unix=time.time());save(p)
