"""Run explicitly predeclared diagnostics sequentially after timing has stopped."""
import json,os,subprocess,sys,time
from pathlib import Path
path=Path(sys.argv[1]);plan=json.loads(path.read_text())
assert plan['status']=='planned_not_started'
plan.update(status='running',controller_pid=os.getpid(),started_unix=time.time())
def save():path.write_text(json.dumps(plan,indent=2)+'\n')
save()
for job in plan['jobs']:
    print('START',job['name'],flush=True)
    job.update(status='running',started_unix=time.time())
    with (path.parent/(job['name']+'.txt')).open('w') as log:
        process=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT)
        job['pid']=process.pid;save()
        job['returncode']=process.wait()
    job.update(status='completed' if job['returncode']==0 else 'diagnostic_failed',finished_unix=time.time())
    save();print('END',job['name'],job['status'],flush=True)
plan.update(status='completed' if all(j['status']=='completed' for j in plan['jobs']) else 'completed_with_failures',finished_unix=time.time());save()
