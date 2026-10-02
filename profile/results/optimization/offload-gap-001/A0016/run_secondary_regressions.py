"""Run the fixed A0016 secondary checks sequentially, preserving all results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

directory = Path(__file__).resolve().parent
root = directory.parent
state_path = directory / 'secondary-regressions-controller.json'
assert not state_path.exists(), 'Never restart an existing controller automatically'
state = dict(attempt_id='A0016', campaign_id='offload-gap-001',
             controller_pid=os.getpid(), status='running', accepted=False, jobs=[])


def save():
    state_path.write_text(json.dumps(state, indent=2) + '\n')


save()
for original, selection_file, output in (
    ('R04-full-validation', 'full-regression-selection.json', 'R07-regression'),
    ('R05-generation-validation', 'generation-regression-selection.json', 'R08-regression'),
):
    selection = json.loads((directory / selection_file).read_text())['selected']
    plan = json.loads((directory / (output + '-plan') / 'manifest.json').read_text())
    assert plan['planned_work']['new_processes'] == 8 * len(selection)
    command = [sys.executable, str(root / 'extend_pairs.py'), '--manifest',
               str(directory / original / 'manifest.json'), '--select', *selection,
               '--output', str(directory / output)]
    job = dict(name=output, command=command, status='running', started_unix=time.time(),
               planned_work=plan['planned_work'])
    state['jobs'].append(job)
    with (directory / (output + '-controller.txt')).open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        job['pid'] = child.pid
        save()
        job['returncode'] = child.wait()
    job['finished_unix'] = time.time()
    manifest_path = directory / output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    passed = job['returncode'] == 0 and manifest.get('status') == 'completed'
    passed = passed and all(j['status'] in ('completed', 'reused') for j in manifest.get('jobs', []))
    job['status'] = 'completed' if passed else 'failed'
    save()
    if not passed:
        state['status'] = 'failed_needs_review'
        save()
        raise SystemExit(1)
state['status'] = 'completed_needs_analysis_and_audit'
save()
