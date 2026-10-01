"""Matched full matrix with explicit reuse of the first confirmed primary pair."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
BASELINE = ROOT.parent / 'baseline-paper-001'
ATTEMPT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'profile'))
from run_paper_matrix import command, jobs
from sampling_plan import sampling_plan


def source_hash(root):
    paths = sorted(set((root/'fla').rglob('*.py')) | set((root/'profile').glob('*.py')) |
                   {p for p in (root/'setup.py', root/'pyproject.toml') if p.exists()})
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes()+b'\0')
    return digest.hexdigest()


def make_plan(output):
    confirmation = json.loads((ATTEMPT/'confirmation-summary.json').read_text())
    assert confirmation['nominated_for_full_validation']
    assert confirmation['offload_memory_saving_preserved']
    previous = json.loads((ATTEMPT/'R04-confirmation/manifest.json').read_text())
    expected = {}
    for side, checkout in [('baseline', BASELINE), ('candidate', ROOT)]:
        sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
        assert not subprocess.check_output(['git','status','--porcelain','--','fla','profile/*.py'],cwd=checkout,text=True).strip()
        if side == 'baseline':
            assert sha == 'd949640ebf2f13f56021bd08c5c9f10e65d571c3'
        fingerprint = source_hash(checkout)
        assert confirmation['source_hashes'][side] == [fingerprint]
        expected[side] = dict(commit=sha, source_sha256=fingerprint)
    plan = []
    for index, point in enumerate(jobs('full_validation'), 1):
        reuse = point['batch']==8 and point['length']==2048 and point['mode']!='generation' and point['variant'] in ('ma_offload','ma_gpu')
        for side in (('baseline','candidate') if index % 2 else ('candidate','baseline')):
            name = f"J{index:02d}-{side}-{point['mode']}-{point['variant']}-b{point['batch']}-l{point['length']}"
            entry = dict(**point, name=name, side=side, expected_commit=expected[side]['commit'],
                         expected_source_sha256=expected[side]['source_sha256'],
                         sampling_plan=sampling_plan('full_validation',point['mode']), status='pending')
            if reuse:
                old_name = f"p1-{side}-{point['variant']}-{point['mode']}"
                old_job = next(j for j in previous['jobs'] if j['name']==old_name)
                assert old_job['status']=='complete'
                path = ATTEMPT/'R04-confirmation'/f'{old_name}.json'
                data = json.loads(path.read_text())
                assert data['source']['source_sha256'] == entry['expected_source_sha256']
                assert (data['config']['warmup'],data['config']['repeats'],data['config']['rounds']) == (10,10,3)
                entry.update(status='reused', result=str(path), expected_commit=old_job['expected_commit'],
                             sampling_plan=sampling_plan('confirmation',point['mode']),
                             original_manifest=str(ATTEMPT/'R04-confirmation/manifest.json'),
                             original_job=old_name, verified_cache_module=old_job['verified_cache_module'],
                             command=old_job['command'], pid=old_job['pid'], result_status='completed')
            else:
                path = output/f'{name}.json'
                args = command(point,path,'full_validation')
                if side == 'baseline':
                    args[1] = str(BASELINE/'profile/bench_fla.py')
                    i = args.index('--stage');del args[i:i+2]
                entry.update(command=args,result=str(path))
            plan.append(entry)
    assert len(plan)==96 and sum(j['status']=='reused' for j in plan)==8
    return dict(stage='full_validation', controller_pid=os.getpid(), status='planned',
                started=time.time(), jobs=plan, new_processes=88, reused_processes=8,
                cost_plan=str(ATTEMPT/'full-validation-plan.json'),
                reuse_policy='Use first confirmation pair without selecting for performance; all three pairs retained for primary uncertainty. Identical source hashes, scope/counts; stages differ but formal measurement_plan_id is shared.',
                interpretation='Non-primary points have one paired process each; round ranges are descriptive, not process-level confidence intervals. Investigate any regression before acceptance. OOM/unsupported/failures retained explicitly.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    manifest=make_plan(output)
    output.mkdir(parents=True,exist_ok=False)
    def save():
        temp=output/'manifest.tmp';temp.write_text(json.dumps(manifest,indent=2)+'\n');temp.replace(output/'manifest.json')
    save()
    if args.plan_only:
        print('plan verified: 96 points, 8 reused, 88 new processes');return
    manifest['status']='running';save()
    for job in manifest['jobs']:
        if job['status']=='reused':continue
        checkout=BASELINE if job['side']=='baseline' else ROOT
        env=dict(os.environ,PYTHONPATH=str(checkout))
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()==job['expected_commit']
        probe=subprocess.check_output([sys.executable,'-c','import fla.models.utils as u; print(u.__file__)'],cwd=checkout,env=env,text=True).strip()
        assert Path(probe).resolve()==checkout/'fla/models/utils.py',probe
        job.update(status='running',started=time.time(),pythonpath=str(checkout),verified_cache_module=probe)
        with (output/(job['name']+'.txt')).open('w') as log:
            p=subprocess.Popen(job['command'],cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT)
            job['pid']=p.pid;save();job['returncode']=p.wait()
        result=Path(job['result'])
        data=json.loads(result.read_text()) if result.exists() else {}
        status=data.get('status','missing')
        job.update(result_status=status,ended=time.time(),status='completed' if status=='completed' and job['returncode']==0 else status if status in ('oom','unsupported') else 'benchmark_failed')
        save();print(job['name'],job['status'],flush=True)
    manifest.update(status='complete' if all(j['status'] in ('completed','reused','oom','unsupported') for j in manifest['jobs']) else 'complete_with_failures',ended=time.time());save()


if __name__=='__main__':main()
