"""A0030 supplemental large-batch paired measurements; no optimization changes."""
import sys,json,time,subprocess,statistics
from pathlib import Path
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parents[1]/'continuation-03'))
import driver
from scipy.stats import t

def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def plan():
 assert not (D/'manifest.json').exists()
 sig=driver.signature('A0030');jobs=[]
 for block in range(4):
  shapes=[(m,b,2048) for b in (32,64) for m in ('prefill','decode')]
  if block%2:shapes.reverse()
  for m,b,l in shapes:
   for v in (['ma_gpu','ma_offload'] if block%2==0 else ['ma_offload','ma_gpu']):
    name=f'B{block+1}-{m}-{v}-b{b}'
    cmd=driver.cmd('A0030',(m,b,l),D/(name+'.json'),True);cmd[cmd.index('PLACEHOLDER')]=v
    jobs.append(dict(name=name,block=block+1,mode=m,batch=b,length=l,variant=v,cwd=sig['root'],candidate_sha=sig['commit'],source_sha256=sig['source_sha256'],command=cmd,status='pending'))
 save(D/'manifest.json',dict(study='A0030-large-batch-01',status='planned',source=sig,blocks=4,scope='Supplemental batch32/64 prefill/decode at context2048; not original matrix or final acceptance',planned_jobs=32,estimated_seconds=sum(driver.estimate('A0030',(j['mode'],j['batch'],2048),j['variant'],True) for j in jobs),failure_policy='Preserve failures; after OOM skip repeats of identical cell, continue other cells. Stop on other failures.',jobs=jobs))
def run():
 p=json.loads((D/'manifest.json').read_text());assert p['status']=='planned';assert driver.signature('A0030')==p['source'];p['status']='running';save(D/'manifest.json',p);oom=set()
 for j in p['jobs']:
  key=(j['mode'],j['batch'],j['variant'])
  if key in oom:j.update(status='not_run',reason='Identical cell already OOM');save(D/'manifest.json',p);continue
  print('START',j['name'],flush=True);j.update(status='running',started_unix=time.time());save(D/'manifest.json',p)
  with (D/(j['name']+'.txt')).open('w') as f:
   proc=subprocess.Popen(j['command'],cwd=j['cwd'],stdout=f,stderr=subprocess.STDOUT);j['pid']=proc.pid;save(D/'manifest.json',p);code=proc.wait()
  j.update(returncode=code,finished_unix=time.time());raw=D/(j['name']+'.json');v=json.loads(raw.read_text()) if raw.exists() else {};status=v.get('status','failed')
  if code or status!='completed':
   log=(D/(j['name']+'.txt')).read_text()
   if 'out of memory' in (log+str(v.get('failure'))).lower():j['status']='oom';oom.add(key)
   else:j['status']='failed';p['status']='stopped_needs_review';save(D/'manifest.json',p);raise RuntimeError(j['name'])
  else:driver.audit_job(j,raw);j['status']='completed'
  save(D/'manifest.json',p);print('END',j['name'],j['status'],flush=True)
 p['status']='completed_with_oom' if oom else 'completed';save(D/'manifest.json',p)
if __name__=='__main__':globals()[sys.argv[1]]()
