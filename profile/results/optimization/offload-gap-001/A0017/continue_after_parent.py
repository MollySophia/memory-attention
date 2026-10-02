"""Fixed conditional A0017 validation; never restarts the parent confirmation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

here=Path(__file__).resolve().parent
root=here.parent
sys.path.insert(0,str(root))
from continue_validation import process_state,save
from analyze_paired import analyze
from audit_final import require_eligible_gain

state_path=here/'conditional-controller.json'
assert not state_path.exists()
parent_path=here/'R02-parent-confirmation/manifest.json'
parent=json.loads(parent_path.read_text())
pid=parent['controller_pid'];identity=process_state(pid)
state=dict(campaign_id='offload-gap-001',attempt_id='A0017',controller_pid=os.getpid(),
           parent_pid=pid,parent_process_identity=identity,status='waiting_parent_confirmation',
           driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),accepted=False,jobs=[])
save(state_path,state)

def run(name,command):
    job=dict(name=name,command=command,status='running',started_unix=time.time())
    state['jobs'].append(job);state['status']=name;save(state_path,state)
    with (here/('conditional-'+name+'.txt')).open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=root.parents[3])
        job['pid']=child.pid;save(state_path,state);job['returncode']=child.wait()
    job.update(status='completed' if job['returncode']==0 else 'failed',finished_unix=time.time());save(state_path,state)
    if job['returncode']:raise RuntimeError(name+' failed; preserved for review')

try:
    while identity is not None and identity[0]!='Z':
        current=process_state(pid)
        if current is None or current[0]=='Z':break
        if current[1]!=identity[1]:raise RuntimeError('Parent PID reused; review required')
        time.sleep(5)
    assert json.loads(parent_path.read_text())['status']=='completed'
    result=analyze(parent_path);save(here/'parent-confirmation-analysis.json',result)
    if not result['confirmation_promising']:
        state['status']='parent_not_promising_needs_verdict';save(state_path,state);raise SystemExit(0)
    require_eligible_gain(result,json.loads((here/'record.json').read_text()))
    common=[sys.executable,str(root/'run_paired.py'),'--baseline-root','/home/molly/workspace-memory-attn/offload-gap-001-A0000',
            '--candidate-root','/home/molly/workspace-memory-attn/offload-gap-001-A0017',
            '--screen-manifest',str(here/'R01/manifest.json'),'--stage','confirmation']
    run('R03-plan',common+['--plan-only','--output',str(here/'R03-baseline-confirmation-plan')])
    run('R03-confirmation',common+['--output',str(here/'R03-baseline-confirmation')])
    run('validation',[sys.executable,str(root/'continue_validation.py'),'--attempt','A0017',
        '--confirmation',str(here/'R03-baseline-confirmation/manifest.json'),
        '--baseline-root','/home/molly/workspace-memory-attn/offload-gap-001-A0000',
        '--candidate-root','/home/molly/workspace-memory-attn/offload-gap-001-A0017',
        '--screen-manifest',str(here/'R01/manifest.json'),'--controller-name','validation-v1',
        '--first-run-index','4','--correctness-first','--reuse-resident-correctness',str(root/'A0001/R06-full-correctness')])
    nested=json.loads((here/'validation-v1.json').read_text())
    state.update(status='conditional_chain_finished_needs_review',validation_status=nested['status'])
    save(state_path,state)
except Exception as error:
    state.update(status='failed_needs_review',error=repr(error));save(state_path,state);raise
