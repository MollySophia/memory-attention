"""Resume an interrupted bounded audit without repeating completed samples."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from run_paired import save, source_hash, validate_balanced_order

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('manifest', type=Path)
p.add_argument('--plan-only', action='store_true')
a = p.parse_args()
m = json.loads(a.manifest.read_text())
assert m['measurement_plan_id'] == 'regression_reuse1_add2_balanced_v1'
validate_balanced_order(m['jobs'])
assert not Path('/proc', str(m['controller_pid'])).exists(), 'Original controller still exists'
expected = dict(torch=importlib.metadata.version('torch'), flash_attn=importlib.metadata.version('flash_attn'),
                python=platform.python_version(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', ''))
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip().split(', ')
expected['gpu'] = gpu[0]
pending = []
recovered = []
for j in m['jobs']:
    if j.get('pid'):
        assert not Path('/proc', str(j['pid'])).exists(), 'Previous child still exists'
    path = Path(j['source_result']) if j['status'] == 'reused' else a.manifest.parent / (j['name'] + '.json')
    if path.exists():
        d = json.loads(path.read_text())
        assert d['status'] == 'completed', 'Failed/partial payload needs explicit audit, never overwrite'
        assert d['source']['source_sha256'] == source_hash(Path(j['cwd']))
        assert d['source']['git_commit']['stdout'].strip() == j['candidate_sha']
        for k, v in expected.items():
            assert d['env'][k] == v, (k, d['env'][k], v)
        env = d['environment_before']
        assert env['gpu_telemetry']['stdout'].splitlines()[1].split(', ')[:2] == gpu
        assert env['cpu_affinity'] == sorted(os.sched_getaffinity(0))
        assert env['thread_environment'] == {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
        if j['status'] not in ('reused', 'completed'):
            recovered.append(j['name'])
            j.update(status='completed', recovery_note='Completed raw payload recovered after controller interruption; no resampling',
                     finished_unix=path.stat().st_mtime)
    else:
        assert j['status'] == 'pending', 'Interrupted child without completed payload needs separate preservation'
        pending.append(j)
plan = dict(remaining_processes=len(pending), recovered_completed_jobs=recovered,
            estimated_seconds=sum(j['estimated_seconds'] for j in pending),
            original_controller_pid=m['controller_pid'], environment_matches=True)
print(json.dumps(plan), flush=True)
if a.plan_only:
    raise SystemExit(0)
backup = a.manifest.with_name('manifest-before-resume.json')
assert not backup.exists(), 'Recovery already started; inspect its controller before continuing'
backup.write_bytes(a.manifest.read_bytes())
m.update(controller_pid=os.getpid(), status='running', recovery=plan, resumed_unix=time.time())
save(a.manifest, m)
for j in pending:
    print('START ' + j['name'], flush=True)
    j['started_unix'] = time.time()
    with (a.manifest.parent / (j['name'] + '.txt')).open('x') as log:
        child = subprocess.Popen(j['command'], cwd=j['cwd'], stdout=log, stderr=subprocess.STDOUT)
        j.update(pid=child.pid, status='running')
        save(a.manifest, m)
        j['returncode'] = child.wait()
    result = a.manifest.parent / (j['name'] + '.json')
    j['status'] = json.loads(result.read_text())['status'] if result.exists() else 'benchmark_failed'
    if j['returncode'] and j['status'] == 'completed':
        j['status'] = 'benchmark_failed'
    j['finished_unix'] = time.time()
    save(a.manifest, m)
    print('END ' + j['name'] + ' ' + j['status'], flush=True)
m['status'] = 'completed'
save(a.manifest, m)
