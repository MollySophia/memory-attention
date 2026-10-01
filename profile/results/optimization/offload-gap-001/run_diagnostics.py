"""Sequential primary diagnostics; timing controller must be paused/finished."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
DEST=Path(sys.argv[1]).resolve()
MATRIX=Path(sys.argv[2]).resolve()
DEST.mkdir(parents=True,exist_ok=False)
plan=[]
for batch in (1,8):
    for mode in ('prefill','decode'):
        for variant,backend in (('ma_offload','pytorch'),('ma_gpu','pytorch'),('ma_offload','offload')):
            name=f'b{batch}-{mode}-{variant}-{backend}'
            command=[sys.executable,str(ROOT/'profile/profile_paper.py'),'--batch-size',str(batch),'--mode',mode,'--variant',variant,'--backend',backend,'--matrix',str(MATRIX),'--output',str(DEST/name)]
            plan.append(dict(name=name,command=command,status='pending'))
manifest=dict(campaign_id='offload-gap-001',controller_pid=os.getpid(),status='running',jobs=plan)
p=DEST/'manifest.json'
def save():
    p.write_text(json.dumps(manifest,indent=2)+'\n')
save()
for job in plan:
    job['started_unix']=time.time()
    with (DEST/(job['name']+'.txt')).open('w') as log:
        process=subprocess.Popen(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        job.update(pid=process.pid,status='running');save()
        job['returncode']=process.wait()
    job.update(status='completed' if job['returncode']==0 else 'benchmark_failed',finished_unix=time.time());save()
    print(job['name'],job['status'],flush=True)
manifest['status']='completed';save()
