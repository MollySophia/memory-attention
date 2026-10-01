"""Three fresh alternating batch1 decode pairs per folded placement for A0002."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
ATTEMPT=Path(__file__).resolve().parent
BASELINE=ROOT.parent/'baseline-paper-001'
sys.path.insert(0,str(ROOT/'profile'))
from run_paper_matrix import command


def main():
    output=Path(sys.argv[1]).resolve();output.mkdir(parents=True,exist_ok=False)
    expected={side:subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
              for side,checkout in [('baseline',BASELINE),('candidate',ROOT)]}
    for checkout in (ROOT,BASELINE):
        assert not subprocess.check_output(['git','status','--porcelain','--','fla','profile/*.py'],cwd=checkout,text=True).strip()
    jobs=[]
    for variant in ('ma_offload','ma_gpu'):
        point=dict(mode='decode',variant=variant,batch=1,length=2048)
        for pair in (1,2,3):
            order=('baseline','candidate') if pair % 2 else ('candidate','baseline')
            for side in order:
                name=f'p{pair}-{side}-{variant}'
                path=output/f'{name}.json';args=command(point,path,'confirmation')
                if side=='baseline':
                    args[1]=str(BASELINE/'profile/bench_fla.py');i=args.index('--stage');del args[i:i+2]
                jobs.append(dict(**point,name=name,pair=pair,side=side,command=args,result=str(path),
                                 expected_commit=expected[side],status='pending'))
    manifest=dict(stage='small_batch_regression_confirmation',controller_pid=os.getpid(),started=time.time(),
                  status='running',jobs=jobs,new_processes=12,reused_processes=0,
                  sampling_plan=dict(warmup=10,repeats=10,rounds=3),
                  estimated_cost='12 process setups plus decode prefixes and40 calls each; about4-5 minutes.',
                  analysis_plan='Three independent paired log speedups per placement, geometric mean and95% Student-t interval (df2). Upper bound below1 confirms regression. Same criteria as primary confirmation; no favorable optional stopping.')
    def save():
        p=output/'manifest.tmp';p.write_text(json.dumps(manifest,indent=2)+'\n');p.replace(output/'manifest.json')
    save()
    for job in jobs:
        if job['status']=='reused':continue
        checkout=BASELINE if job['side']=='baseline' else ROOT;env=dict(os.environ,PYTHONPATH=str(checkout))
        probe=subprocess.check_output([sys.executable,'-c','import fla.models.utils as u; print(u.__file__)'],cwd=checkout,env=env,text=True).strip()
        assert Path(probe).resolve()==checkout/'fla/models/utils.py'
        job.update(status='running',started=time.time(),verified_cache_module=probe,pythonpath=str(checkout))
        with (output/(job['name']+'.txt')).open('w') as log:
            p=subprocess.Popen(job['command'],cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT)
            job['pid']=p.pid;save();job['returncode']=p.wait()
        result=Path(job['result']);status=json.loads(result.read_text()).get('status') if result.exists() else 'missing'
        job.update(status='completed' if status=='completed' and job['returncode']==0 else 'failed',ended=time.time(),result_status=status);save()
        print(job['name'],job['status'],flush=True)
        if job['status']=='failed':
            manifest['status']='failed';save();raise SystemExit(1)
    manifest.update(status='complete',ended=time.time());save()


if __name__=='__main__':main()
