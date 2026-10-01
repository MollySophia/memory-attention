"""Continue an existing confirmation into nominated validation, sequentially.

Waits on the recorded live controller; never restarts it. No acceptance verdict
is issued here. After all stages, a human/agent audit still evaluates regressions,
memory, exactness, and the complete original objective.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def process_state(pid):
    path=Path('/proc')/str(pid)/'stat'
    try:fields=path.read_text().rsplit(') ',1)[1].split()
    except FileNotFoundError:return None
    return fields[0],fields[19]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',required=True)
    p.add_argument('--confirmation',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--candidate-root',type=Path,required=True)
    p.add_argument('--screen-manifest',type=Path,required=True)
    args=p.parse_args()
    attempt=ROOT/args.attempt
    state_path=attempt/'validation-controller.json'
    assert not state_path.exists(),'Refuse to duplicate an existing validation controller'
    confirmation=json.loads(args.confirmation.read_text())
    pid=confirmation['controller_pid'];initial=process_state(pid)
    state=dict(campaign_id='offload-gap-001',attempt_id=args.attempt,status='waiting_confirmation',
               controller_pid=__import__('os').getpid(),confirmation_pid=pid,
               confirmation_process_identity=initial,jobs=[],accepted=False)
    save(state_path,state)
    while initial is not None and initial[0]!='Z':
        current=process_state(pid)
        if current is None or current[0]=='Z':break
        if current[1]!=initial[1]:raise RuntimeError('Confirmation PID reused; inspect before proceeding')
        time.sleep(5)
    confirmation=json.loads(args.confirmation.read_text())
    if confirmation['status']!='completed':
        state.update(status='confirmation_incomplete',reason='Recorded controller terminal without completed manifest')
        save(state_path,state);return 1

    def run(name,command):
        job=dict(name=name,command=command,status='running',started_unix=time.time())
        state['jobs'].append(job);state['status']=name;save(state_path,state)
        with (attempt/(name+'-controller.txt')).open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT.parents[3],stdout=log,stderr=subprocess.STDOUT)
            job['pid']=proc.pid;save(state_path,state)
            job['returncode']=proc.wait()
        job.update(status='completed' if job['returncode']==0 else 'failed',finished_unix=time.time())
        save(state_path,state)
        if job['returncode']!=0:raise RuntimeError(name+' failed; inspect persisted output')

    try:
        analysis=attempt/'confirmation-analysis.json'
        run('analyze-confirmation',[sys.executable,str(ROOT/'analyze_paired.py'),str(args.confirmation),'--output',str(analysis)])
        result=json.loads(analysis.read_text())
        if not result['confirmation_promising']:
            state.update(status='confirmation_needs_review',reason='Predeclared primary statistical/memory gate did not pass')
            save(state_path,state);return 0
        state['nomination']='promising confirmation; full acceptance gates remain'
        common=[sys.executable,str(ROOT/'run_paired.py'),'--baseline-root',str(args.baseline_root),
                '--candidate-root',str(args.candidate_root),'--screen-manifest',str(args.screen_manifest)]
        for stage,run_id in [('full_validation','R03-full-validation'),('generation_validation','R04-generation-validation')]:
            cmd=common+['--stage',stage]
            if stage=='full_validation':cmd+=['--reuse-confirmation',str(args.confirmation)]
            run(run_id+'-plan',cmd+['--plan-only','--output',str(attempt/(run_id+'-plan'))])
            run(run_id,cmd+['--output',str(attempt/run_id)])
            manifest=json.loads((attempt/run_id/'manifest.json').read_text())
            if any(j['status'] not in ('completed','reused') for j in manifest['jobs']):
                state.update(status=run_id+'_needs_review',reason='OOM/unsupported/failure preserved; inspect matrix before additional spend')
                save(state_path,state);return 0
        destination=attempt/'R05-full-correctness';destination.mkdir(exist_ok=False)
        for batch in (1,8):
            for seed in (1234,4321):
                outputs={}
                for impl,source,variant in [('baseline',args.baseline_root,'ma_gpu'),('candidate',args.candidate_root,'ma_offload')]:
                    name=f'correctness-b{batch}-s{seed}-{impl}'
                    outputs[impl]=destination/(name+'.json')
                    run(name,[sys.executable,str(ROOT/'full_correctness.py'),'--source-root',str(source),
                              '--variant',variant,'--batch-size',str(batch),'--seed',str(seed),'--output',str(outputs[impl])])
                run(f'compare-correctness-b{batch}-s{seed}',[sys.executable,str(ROOT/'compare_correctness.py'),
                    str(outputs['baseline']),str(outputs['candidate']),'--output',str(destination/f'comparison-b{batch}-s{seed}.json')])
        state.update(status='validation_complete_needs_audit',accepted=False)
        save(state_path,state);return 0
    except Exception as exc:
        state.update(status='failed_needs_review',error=repr(exc));save(state_path,state);raise

if __name__=='__main__':raise SystemExit(main())
