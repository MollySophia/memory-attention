"""Run staged GOAL.md workloads in isolated sequential subprocesses.

Default screening runs eight primary-shape jobs. Use --stage
full_validation for the full matrix, or legacy for original A0000 sampling.
Confirmation runs here are not a substitute for independent process pairs.
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
sys.path.insert(0, str(ROOT / 'profile'))
from sampling_plan import sampling_plan


def jobs(stage="screening", variants=None, initial_baseline=False, include_batch16=False):
    variants = variants or (("ma_offload", "ma_gpu") if stage in ("screening", "confirmation") else ("ma_offload", "ma_gpu", "ma_gpu_unfolded"))
    shapes = [(8, 2048), (1, 2048), (4, 2048), (16, 2048),
              (8, 512), (8, 4096), (8, 8192)]
    if stage in ('screening', 'confirmation'):
        shapes = [(1, 2048), (8, 2048)]
        if initial_baseline:
            shapes += [(4, 2048), (16, 2048)]
        elif include_batch16:
            shapes += [(16, 2048)]
    elif stage == 'generation_validation':
        shapes = []
    for batch, length in shapes:
        for mode in ('prefill', 'decode'):
            for variant in variants:
                yield dict(mode=mode, variant=variant, batch=batch, length=length)
    if stage in ("screening", "confirmation"):
        return
    for batch in (1, 8):
        for variant in variants:
            yield dict(mode='generation', variant=variant, batch=batch, length=2048)


def command(job, path, stage="screening"):
    plan = sampling_plan(stage, job["mode"])
    return [sys.executable, str(ROOT / 'profile/bench_fla.py'),
            '--mode', job['mode'], '--variants', job['variant'],
            '--batch-size', str(job['batch']), '--seq-len', str(job['length']),
            '--context-len', str(job['length']), '--num-layers', '24',
            '--hidden-size', '2048', '--num-heads', '32', '--num-kv-heads', '32',
            '--intermediate-size', '5632', '--vocab-size', '32000', '--seed', '1234',
            '--stage', stage, '--warmup', str(plan['warmup']),
            '--repeats', str(plan['repeats']), '--rounds', str(plan['rounds']),
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
    parser.add_argument('--stage', choices=('screening','confirmation','full_validation','generation_validation','legacy'), default='screening')
    parser.add_argument('--variants', nargs='+', choices=('ma_offload','ma_gpu','ma_gpu_unfolded'))
    parser.add_argument('--initial-baseline', action='store_true')
    parser.add_argument('--include-batch16', action='store_true')
    parser.add_argument('--estimate-setup-seconds', type=float, default=30)
    parser.add_argument('--estimate-prefill-ms', type=float, default=1000)
    parser.add_argument('--estimate-decode-ms', type=float, default=30)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    # Refuse changed or untracked Python implementation files at launch.
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'fla', 'profile/*.py'], cwd=ROOT, text=True)
    if dirty and not args.plan_only:
        raise RuntimeError('commit implementation before measurement: '+dirty)
    plan = []
    for i, job in enumerate(jobs(args.stage, tuple(dict.fromkeys(args.variants)) if args.variants else None, args.initial_baseline, args.include_batch16)):
        name = f"J{i+1:02d}-{job['mode']}-{job['variant']}-b{job['batch']}-l{job['length']}"
        plan.append(dict(**job, name=name, command=command(job, output/(name+'.json'), args.stage), sampling_plan=sampling_plan(args.stage, job['mode']), status='pending'))
    manifest = dict(campaign_id='offload-gap-001', protocol_version='offload_gap_v1', stage=args.stage, candidate_sha=commit, python=sys.executable, controller_pid=os.getpid(),
                    started_unix=time.time(), jobs=plan, status='planned')
    manifest['planned_work'] = dict(jobs=len(plan),
        samples=sum(j['sampling_plan']['repeats']*j['sampling_plan']['rounds'] for j in plan),
        model_calls_including_warmup=sum((j['sampling_plan']['warmup']+j['sampling_plan']['repeats']*j['sampling_plan']['rounds'])*(129 if j['mode']=='generation' else 1) for j in plan),
        note='Excludes model loading and decode prefix setup; confirmation here is one side of one run, not independent paired evidence.')
    manifest['planned_work']['estimated_seconds'] = sum(args.estimate_setup_seconds + (j['sampling_plan']['warmup'] + j['sampling_plan']['repeats']*j['sampling_plan']['rounds'])*(args.estimate_prefill_ms + 128*args.estimate_decode_ms if j['mode']=='generation' else args.estimate_prefill_ms if j['mode']=='prefill' else args.estimate_decode_ms)/1000 for j in plan)
    manifest['planned_work']['estimate_inputs'] = vars(args).copy()
    manifest['planned_work']['estimate_inputs']['output'] = str(output)
    print(json.dumps(manifest['planned_work']), flush=True)
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
