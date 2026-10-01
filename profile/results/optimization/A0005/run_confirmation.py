"""Predeclared independent process pairs for A0005 incremental offload primary confirmation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
BASELINE = ROOT.parent / 'accepted-a0002'
sys.path.insert(0, str(ROOT / 'profile'))
from run_paper_matrix import command


def main():
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    expected = {}
    for side, checkout in [('baseline', BASELINE), ('candidate', ROOT)]:
        expected[side] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
        assert not subprocess.check_output(['git', 'status', '--porcelain', '--', 'fla', 'profile/*.py'], cwd=checkout, text=True).strip()
    assert expected['baseline'] == 'ec6384d2e9081ecbb2926fe8a6193bc48ea650a3'
    assert not subprocess.check_output(['git', 'diff', '3fa9ce2', 'HEAD', '--', 'fla'], cwd=ROOT, text=True).strip()
    jobs = []
    for variant in ('ma_offload',):
        for pair in range(1, 4):
            for mode in ('prefill', 'decode'):
                for side in (('baseline', 'candidate') if pair % 2 else ('candidate', 'baseline')):
                    name = f'p{pair}-{side}-{variant}-{mode}'
                    args = command(dict(mode=mode, variant=variant, batch=8, length=2048), output / f'{name}.json', 'confirmation')
                    if side == 'baseline':
                        args[1] = str(BASELINE / 'profile/bench_fla.py')
                    jobs.append(dict(name=name, pair=pair, mode=mode, variant=variant, side=side,
                                     expected_commit=expected[side], command=args, status='pending'))
    manifest = dict(stage='confirmation', controller_pid=os.getpid(),
                    plan=dict(warmup=10, repeats=10, rounds=3, independent_pairs=3),
                    baseline_attempt='A0002', note='Incremental comparison against accepted A0002; matching formal sampling on both sides.',
                    estimated_cost='12 isolated model setups,480 timed/warmup calls and6 decode prefixes. Screening setup durations imply about4-5minutes, including about50seconds of prefill calls. No full matrix/generation.',
                    analysis_plan='Per-process median of 3 round means. For each placement/mode, pair log(baseline/candidate), report all 3 ratios and geometric mean with 95% Student-t interval (df=2). Either primary lower bound >1 supports improvement; either upper bound <1 is resolved regression. A straddling interval is unresolved. Inspect raw samples/environment; do not add runs just to obtain favorable results. Acceptance additionally requires full matrix/generation and correctness.',
                    job_count=len(jobs), jobs=jobs, status='running', started=time.time())
    def save():
        temp = output / 'manifest.tmp'
        temp.write_text(json.dumps(manifest, indent=2)+'\n')
        temp.replace(output / 'manifest.json')
    save()
    for job in jobs:
        checkout = BASELINE if job['side'] == 'baseline' else ROOT
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()==job['expected_commit']
        env = dict(os.environ, PYTHONPATH=str(checkout))
        probe = subprocess.check_output([sys.executable, '-c', 'import fla.models.utils as u; print(u.__file__)'], cwd=checkout, env=env, text=True).strip()
        assert Path(probe).resolve() == checkout / 'fla/models/utils.py', probe
        job.update(status='running', started=time.time(), pythonpath=str(checkout), verified_cache_module=probe)
        with (output / f"{job['name']}.txt").open('w') as log:
            p = subprocess.Popen(job['command'], cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT)
            job['pid'] = p.pid
            save()
            job['returncode'] = p.wait()
        result_path = output / f"{job['name']}.json"
        result_status = json.loads(result_path.read_text()).get('status') if result_path.exists() else 'missing'
        job.update(status='complete' if job['returncode']==0 and result_status=='completed' else 'failed',
                   result_status=result_status, ended=time.time())
        save()
        print(job['name'], job['status'], flush=True)
        if job['status'] != 'complete':
            manifest.update(status='failed', ended=time.time())
            save()
            raise SystemExit(1)
    manifest.update(status='complete', ended=time.time())
    save()


if __name__ == '__main__':
    main()
