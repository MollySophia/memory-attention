"""Reproduce the eight matched-plan screening subprocesses; no formal gain claims."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
BASELINE = ROOT.parent / 'baseline-paper-001'
sys.path.insert(0, str(ROOT / 'profile'))
from run_paper_matrix import command


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASELINE, text=True).strip() == 'd949640ebf2f13f56021bd08c5c9f10e65d571c3'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--', 'fla', 'profile/*.py'], cwd=ROOT, text=True).strip()
    jobs = []
    for variant in ('ma_offload', 'ma_gpu'):
        for mode in ('prefill', 'decode'):
            for side in (('baseline', 'candidate') if mode == 'prefill' else ('candidate', 'baseline')):
                name = f'{side}-{variant}-{mode}'
                args = command(dict(mode=mode, variant=variant, batch=8, length=2048), output / f'{name}.json', 'screening')
                if side == 'baseline':
                    args[1] = str(BASELINE / 'profile/bench_fla.py')
                    i = args.index('--stage')
                    del args[i:i+2]
                jobs.append(dict(name=name, side=side, command=args, status='pending'))
    manifest = dict(stage='screening', plan=dict(warmup=3, repeats=5, rounds=1),
                    note='Frozen baseline CLI predates stage IDs; explicit counts and scope match. Screening is not confirmation.',
                    estimated_cost='8 isolated model setups plus 64 model calls and 4 decode prefixes; approximately 3-8 minutes, dominated by model initialization. No generation/full matrix.',
                    job_count=8, jobs=jobs, status='running')
    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    for job in jobs:
        job.update(status='running', started=time.time())
        with (output / f"{job['name']}.txt").open('w') as log:
            p = subprocess.Popen(job['command'], cwd=BASELINE if job['side']=='baseline' else ROOT, stdout=log, stderr=subprocess.STDOUT)
            job['pid'] = p.pid
            save()
            job['returncode'] = p.wait()
        job.update(status='complete' if job['returncode']==0 else 'failed', ended=time.time())
        save()
        print(job['name'], job['status'], flush=True)
        if job['returncode']:
            manifest['status'] = 'failed'
            save()
            return
    manifest['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
