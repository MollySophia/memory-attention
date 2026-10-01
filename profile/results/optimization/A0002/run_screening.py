"""Reproduce the sixteen matched-plan screening subprocesses; no formal gain claims."""
import json
import os
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
    for index, (batch, length, mode) in enumerate(((8,2048,'prefill'),(8,2048,'decode'),(1,2048,'decode'),(8,512,'decode'))):
        for variant in ('ma_offload', 'ma_gpu'):
            for side in (('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')):
                name = f'{side}-{variant}-{mode}-b{batch}-l{length}'
                args = command(dict(mode=mode, variant=variant, batch=batch, length=length), output / f'{name}.json', 'screening')
                if side == 'baseline':
                    args[1] = str(BASELINE / 'profile/bench_fla.py')
                    i = args.index('--stage')
                    del args[i:i+2]
                checkout = BASELINE if side == 'baseline' else ROOT
                commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
                jobs.append(dict(name=name, mode=mode, variant=variant, batch=batch, length=length,
                                 side=side, expected_commit=commit, command=args, status='pending'))
    manifest = dict(stage='screening', controller_pid=os.getpid(), started=time.time(), plan=dict(warmup=3, repeats=5, rounds=1),
                    note='Frozen baseline CLI predates stage IDs; explicit counts and scope match. Screening is not confirmation.',
                    estimated_cost='16 isolated model setups plus128 model calls and12 decode prefixes; about5-7 minutes from measured setup costs. Include batch1 and short-context regressions before expensive confirmation/full validation.',
                    job_count=16, jobs=jobs, status='running')
    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    save()
    for job in jobs:
        checkout = BASELINE if job['side'] == 'baseline' else ROOT
        env = dict(os.environ, PYTHONPATH=str(checkout))
        probe = subprocess.check_output(
            [sys.executable, '-c', 'import fla.models.utils as u; print(u.__file__)'],
            cwd=checkout, env=env, text=True).strip()
        assert Path(probe).resolve() == checkout / 'fla/models/utils.py', probe
        job.update(status='running', started=time.time(), pythonpath=str(checkout),
                   verified_cache_module=probe)
        with (output / f"{job['name']}.txt").open('w') as log:
            p = subprocess.Popen(job['command'], cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT)
            job['pid'] = p.pid
            save()
            job['returncode'] = p.wait()
        path = output / (job['name'] + '.json')
        status = json.loads(path.read_text()).get('status') if path.exists() else 'missing'
        job.update(status='complete' if job['returncode']==0 and status=='completed' else 'failed', result_status=status, ended=time.time())
        save()
        print(job['name'], job['status'], flush=True)
        if job['status'] == 'failed':
            manifest['status'] = 'failed'
            save()
            raise SystemExit(1)
    manifest['status'] = 'complete'
    manifest['ended'] = time.time()
    save()


if __name__ == '__main__':
    main()
