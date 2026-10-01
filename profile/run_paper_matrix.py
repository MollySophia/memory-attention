"""Run the complete GOAL.md matrix in isolated sequential subprocesses.

A new directory is required. Interrupted jobs retain command/log/return status;
never treat incomplete or missing measurements as zero or silently skip them.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def jobs():
    shapes = [(8, 2048), (1, 2048), (4, 2048), (16, 2048),
              (8, 512), (8, 4096), (8, 8192)]
    for batch, length in shapes:
        for mode in ('prefill', 'decode'):
            for variant in ('ma_offload', 'ma_gpu', 'ma_gpu_unfolded'):
                yield dict(mode=mode, variant=variant, batch=batch, length=length)
    for batch in (1, 8):
        for variant in ('ma_offload', 'ma_gpu', 'ma_gpu_unfolded'):
            yield dict(mode='generation', variant=variant, batch=batch, length=2048)


def command(job, path):
    return [sys.executable, str(ROOT / 'profile/bench_fla.py'),
            '--mode', job['mode'], '--variants', job['variant'],
            '--batch-size', str(job['batch']), '--seq-len', str(job['length']),
            '--context-len', str(job['length']), '--num-layers', '24',
            '--hidden-size', '2048', '--num-heads', '32', '--num-kv-heads', '32',
            '--intermediate-size', '5632', '--vocab-size', '32000', '--seed', '1234',
            '--warmup', '30', '--repeats', '30', '--rounds', '5',
            '--logits-to-keep', '1', '--prefill-workload', 'inference',
            '--generation-steps', '128', '--policy', 'auto', '--group-size', '1',
            '--prefetch-depth', '4', '--json', str(path)]


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    # Refuse changed or untracked Python implementation files at launch.
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'fla', 'profile/*.py'], cwd=ROOT, text=True)
    if dirty and not args.plan_only:
        raise RuntimeError('commit implementation before measurement: '+dirty)
    plan = []
    for i, job in enumerate(jobs()):
        name = f"J{i+1:02d}-{job['mode']}-{job['variant']}-b{job['batch']}-l{job['length']}"
        plan.append(dict(**job, name=name, command=command(job, output/(name+'.json')), status='pending'))
    manifest = dict(candidate_sha=commit, python=sys.executable, controller_pid=os.getpid(),
                    started_unix=time.time(), jobs=plan, status='planned')
    path = output/'manifest.json'
    save(path, manifest)
    if args.plan_only:
        return
    manifest['status'] = 'running'
    for job in plan:
        print('START '+job['name'], flush=True)
        job['started_unix'] = time.time()
        with (output/(job['name']+'.txt')).open('w') as log:
            process = subprocess.Popen(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            job.update(status='running', pid=process.pid)
            save(path, manifest)
            try:
                job['returncode'] = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                job['returncode'] = process.wait()
                job['status'] = 'interrupted'
                manifest['status'] = 'interrupted'
                save(path, manifest)
                raise
        job['finished_unix'] = time.time()
        result_path = output/(job['name']+'.json')
        if result_path.exists():
            result = json.loads(result_path.read_text())
            job['status'] = result.get('status', 'benchmark_failed')
            if job['returncode'] != 0 and job['status'] == 'completed':
                job['status'] = 'benchmark_failed'
        else:
            job['status'] = 'benchmark_failed'
            job['reason'] = 'Process produced no structured result; inspect log and returncode (including signal exits).'
        save(path, manifest)
        print('END '+job['name']+' '+job['status'], flush=True)
    manifest['status'] = 'completed'
    manifest['finished_unix'] = time.time()
    save(path, manifest)


if __name__ == '__main__':
    main()
