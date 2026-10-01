"""Matched full matrix with verified reuse of formal points and unchanged frozen-baseline records."""
import argparse
import csv
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import statistics
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


def environment_signature(data):
    gpu=next(csv.DictReader(io.StringIO(data['environment_before']['gpu_telemetry']['stdout'].strip()),skipinitialspace=True))
    return dict(versions={k:v for k,v in data['env'].items() if k!='git_commit'},driver=gpu['driver_version'],
                settings={k:data['environment_before'][k] for k in ('torch_threads','torch_interop_threads','thread_environment','cpu_affinity','platform')})


def reuse_records():
    records={}
    sources=[(ATTEMPT.parent/'A0001/R06-full-validation/manifest.json',True),
             (ATTEMPT/'R02-small-batch/manifest.json',False),
             (ATTEMPT/'R03-confirmation/manifest.json',False)]
    for source,baseline_only in sources:
        manifest=json.loads(source.read_text())
        for job in manifest['jobs']:
            if job['status'] not in ('complete','completed'):continue
            if baseline_only and job['side']!='baseline':continue
            if not baseline_only and job['pair']!=1:continue
            path=Path(job.get('result',source.parent/(job['name']+'.json')))
            data=json.loads(path.read_text());c=data['config']
            key=(job['side'],c['mode'],c['variants'][0],c['batch_size'],c['context_len'])
            records[key]=(job,path,data,source)
    return records


def estimated_cost(plan):
    baseline=ATTEMPT.parent/'A0000/R01'
    old=json.loads((baseline/'manifest.json').read_text())
    latency={}
    for job in old['jobs']:
        data=json.loads((baseline/(job['name']+'.json')).read_text())
        latency[(job['mode'],job['variant'],job['batch'],job['length'])]=data['results'][0]['median_ms']/1000
    recent=json.loads((ATTEMPT/'R02-small-batch/manifest.json').read_text())
    overhead=[]
    for job in recent['jobs']:
        data=json.loads(Path(job['result']).read_text())
        work=data['results'][0]['median_ms']/1000*40
        prefix=latency[('prefill',job['variant'],1,2048)]
        overhead.append(job['ended']-job['started']-work-prefix)
    setup=statistics.median(overhead)
    model_work=0;new=0
    for job in plan:
        if job['status']=='reused':continue
        new+=1;p=job['sampling_plan'];calls=p['warmup']+p['repeats']*p['rounds']
        model_work+=latency[(job['mode'],job['variant'],job['batch'],job['length'])]*calls
        if job['mode']=='decode':model_work+=latency[('prefill',job['variant'],job['batch'],job['length'])]
    return dict(new_processes=new,reused_processes=96-new,estimated_model_work_seconds=model_work,
                setup_seconds_per_process=setup,import_probe_seconds_per_process=3,
                estimated_total_minutes=(model_work+new*(setup+3))/60,
                assumptions='Original baseline latencies for both sides (ignores expected candidate decode gain); median R02 setup overhead plus3s import probe; include warmups, trajectories and decode prefixes. Estimate only, never a timeout.')


def make_plan(output):
    confirmation = json.loads((ATTEMPT/'confirmation-summary.json').read_text())
    assert confirmation['nominated_for_full_validation']
    assert confirmation['offload_memory_saving_preserved']
    reusable = reuse_records()
    reference_environment = environment_signature(reusable[('baseline','prefill','ma_offload',8,2048)][2])
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
        for side in (('baseline','candidate') if index % 2 else ('candidate','baseline')):
            name = f"J{index:02d}-{side}-{point['mode']}-{point['variant']}-b{point['batch']}-l{point['length']}"
            entry = dict(**point, name=name, side=side, expected_commit=expected[side]['commit'],
                         expected_source_sha256=expected[side]['source_sha256'],
                         sampling_plan=sampling_plan('full_validation',point['mode']), status='pending')
            record=reusable.get((side,point['mode'],point['variant'],point['batch'],point['length']))
            can_reuse=False
            if record is not None:
                old_job,path,data,source=record
                c=data['config'];reuse_plan=entry['sampling_plan']
                can_reuse=(data['status']=='completed' and data['protocol_version']=='paper_v1'
                           and data['source']['source_sha256']==entry['expected_source_sha256']
                           and not data['source']['source_patch']['stdout']
                           and environment_signature(data)==reference_environment
                           and all(c[k]==reuse_plan[k] for k in ('warmup','repeats','rounds')))
            if can_reuse:
                original_stage=data.get('stage','full_validation')
                entry.update(status='reused',result=str(path),expected_commit=old_job['expected_commit'],
                             sampling_plan=sampling_plan(original_stage,point['mode']),
                             original_manifest=str(source),original_job=old_job['name'],
                             verified_cache_module=old_job['verified_cache_module'],command=old_job['command'],
                             pid=old_job['pid'],result_status='completed')
            else:
                path = output/f'{name}.json'
                args = command(point,path,'full_validation')
                if side == 'baseline':
                    args[1] = str(BASELINE/'profile/bench_fla.py')
                    i = args.index('--stage');del args[i:i+2]
                entry.update(command=args,result=str(path))
            plan.append(entry)
    assert len(plan)==96
    reused=sum(j['status']=='reused' for j in plan)
    return dict(stage='full_validation', controller_pid=os.getpid(), status='planned',
                started=time.time(), jobs=plan, new_processes=96-reused, reused_processes=reused,
                cost_plan=str(ATTEMPT/'full-validation-plan.json'), estimated_cost=estimated_cost(plan),
                reuse_policy='Deterministic priority: current first primary/batch1 formal pairs, then completed frozen-baseline points from A0001/R06. Never reuse A0001 candidate data. Verify source hash, counts, scope and recorded environment; otherwise rerun. All formal pairs retained for uncertainty.',
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
        print(f"plan verified:96 points,{manifest['reused_processes']} reused,{manifest['new_processes']} new processes");return
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
