"""Explicit recovery after a metadata-only preparation error; no timing repeats."""
import os,subprocess,sys,time
from pathlib import Path
from study import D,read,save
previous=read(D/'workflow-controller.json')
assert previous['status']=='stopped_confirmation_needs_review'
assert read(D/'screen-controller.json')['status']=='completed'
assert read(D/'A0011/screen-bulk-addendum/manifest.json')['status']=='completed'
plan=read(D/'confirmation-plan.json');assert plan['status']=='frozen_before_independent_confirmation'
assert not (D/'confirmation-controller.json').exists()
state=dict(status='confirmation_run',pid=os.getpid(),started_unix=previous['started_unix'],resumed_unix=time.time(),prior_record='preparation-failure-01/workflow-controller.json',reason='Historical reference was measured in original repository at correct SHA/hash. Audit now checks recorded command/cwd module path, preserving strict source identity. No GPU measurements were repeated.',stages=[])
command=[sys.executable,str(D/'confirm.py'),'run'];entry=dict(action='run',command=command,started_unix=time.time());state['stages'].append(entry)
with (D/'confirmation-run-controller.txt').open('w') as log:
 child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);entry['pid']=child.pid;save(D/'workflow-controller.json',state);entry['returncode']=child.wait()
entry['finished_unix']=time.time();state.update(status='measurements_complete_pending_report' if entry['returncode']==0 else 'stopped_confirmation_needs_review',finished_unix=time.time());save(D/'workflow-controller.json',state)
raise SystemExit(entry['returncode'])
