"""Bounded regression audit: reuse one measured pair, add two alternating pairs.
Original raw rows retain their stage, scope and counts; no outliers discarded.
"""
import argparse,importlib.metadata,json,os,platform,subprocess,sys,time
from pathlib import Path
from run_paired import source_hash,save,validate_balanced_order
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,required=True);p.add_argument('--select',nargs='+',required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-only',action='store_true')
a=p.parse_args();original=json.loads(a.manifest.read_text());assert original['status']=='completed'
selected=[(mode,int(batch),int(length)) for mode,batch,length in (x.split(':') for x in a.select)]
output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
expected=dict(torch=importlib.metadata.version('torch'),flash_attn=importlib.metadata.version('flash_attn'),python=platform.python_version(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES',''))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip().split(', ')
assert len(gpu)==2;expected['gpu']=gpu[0]
jobs=[];counter=0;references={}
for key in selected:
 refs=[j for j in original['jobs'] if (j['mode'],j['batch'],j['length'])==key and j['variant'] in ('ma_offload','ma_gpu')]
 assert len(refs)==4
 references[key]=refs
 for j in refs:
  assert j['status']=='completed'
  path=(a.manifest.parent/(j['name']+'.json')).resolve();d=json.loads(path.read_text())
  assert d['status']=='completed' and d['protocol_version']=='offload_gap_v1'
  assert d['source']['source_sha256']==source_hash(Path(j['cwd']))
  assert d['source']['git_commit']['stdout'].strip()==j['candidate_sha']
  for k,v in expected.items():assert d['env'][k]==v
  env=d['environment_before'];assert env['gpu_telemetry']['stdout'].splitlines()[1].split(', ')[:2]==gpu
  assert env['cpu_affinity']==sorted(os.sched_getaffinity(0))
  assert env['thread_environment']=={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}
  counter+=1
  jobs.append(dict(**{k:j[k] for k in ('mode','batch','length','variant','implementation','candidate_sha','cwd')},name=f'B1-J{counter:03d}-reused',block=1,status='reused',source_result=str(path),estimated_seconds=0))
for block in (2,3):
 for key in (list(reversed(selected)) if block==2 else selected):
  refs=references[key];variants=list(dict.fromkeys(j['variant'] for j in refs))
  if block==2:variants.reverse()
  for variant in variants:
   pair=[j for j in refs if j['variant']==variant]
   if block==2:pair.reverse()
   for j in pair:
    counter+=1;name=f'B{block}-J{counter:03d}-{j["implementation"]}-{j["mode"]}-{variant}-b{j["batch"]}-l{j["length"]}'
    cmd=j['command'].copy();cmd[cmd.index('--json')+1]=str(output/(name+'.json'))
    jobs.append(dict(**{k:j[k] for k in ('mode','batch','length','variant','implementation','candidate_sha','cwd')},name=name,block=block,status='pending',command=cmd,estimated_seconds=j['finished_unix']-j['started_unix']))
validate_balanced_order(jobs)
m=dict(campaign_id='offload-gap-001',protocol_version='offload_gap_v1',measurement_plan_id='regression_reuse1_add2_balanced_v1',stage='regression_check',controller_pid=os.getpid(),source_manifest=str(a.manifest.resolve()),jobs=jobs,status='planned',planned_work=dict(new_processes=sum(j['status']=='pending' for j in jobs),reused_processes=sum(j['status']=='reused' for j in jobs),estimated_seconds=sum(j['estimated_seconds'] for j in jobs)),rule='Exactly two additional independent pairs per placement; original pair retained. No automatic acceptance. No extension until favorable.')
path=output/'manifest.json';save(path,m);print(json.dumps(m['planned_work']),flush=True)
if a.plan_only:raise SystemExit(0)
m['status']='running';save(path,m)
for j in jobs:
 if j['status']=='reused':continue
 print('START '+j['name'],flush=True);j['started_unix']=time.time()
 with (output/(j['name']+'.txt')).open('w') as log:
  child=subprocess.Popen(j['command'],cwd=j['cwd'],stdout=log,stderr=subprocess.STDOUT)
  j.update(pid=child.pid,status='running');save(path,m);j['returncode']=child.wait()
 result=output/(j['name']+'.json')
 j['status']=json.loads(result.read_text())['status'] if result.exists() else 'benchmark_failed'
 if j['returncode'] and j['status']=='completed':j['status']='benchmark_failed'
 j['finished_unix']=time.time();save(path,m);print('END '+j['name']+' '+j['status'],flush=True)
m['status']='completed';save(path,m)
