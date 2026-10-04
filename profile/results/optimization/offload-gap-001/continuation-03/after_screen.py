"""Observe a live complete-matrix screen, then execute the fixed sequential workflow.
Never restart timings or accept a candidate automatically.
"""
import argparse,os,time,subprocess,sys
import driver as d
import confirm
import validation
from after_confirmation import identity

def main(a):
 path=d.C/a/'continuation-controller.json';assert not path.exists()
 screen=d.C/a/'R01-complete-screen/manifest.json';p=d.read(screen);pid=p['controller_pid'];initial=identity(pid)
 assert initial is not None or p['status']=='completed'
 state=dict(status='waiting_existing_screen',controller_pid=os.getpid(),observed_pid=pid,observed_identity=initial,started_unix=time.time(),accepted=False);d.save(path,state)
 try:
  while initial is not None and initial[0]!='Z':
   current=identity(pid)
   if current is None or current[0]=='Z':break
   assert current[1]==initial[1],'Screen PID reused; inspect before proceeding'
   time.sleep(10)
  assert d.read(screen)['status']=='completed','Existing screen exited without completed evidence; no restart'
  state['status']='preparing_confirmation';d.save(path,state)
  plan=confirm.prepare(a);state['confirmation_work']=plan['planned_work']
  command=[sys.executable,str(d.D/'confirm.py'),'run','--attempt',a]
  state.update(status='confirmation_running',confirmation_command=command)
  with (d.C/a/'R02-parent-confirmation-controller.txt').open('w') as log:
   proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);state['confirmation_pid']=proc.pid;d.save(path,state);state['confirmation_returncode']=proc.wait()
  d.save(path,state);assert state['confirmation_returncode']==0,'Confirmation failed; inspect raw evidence'
  result=confirm.analyze(a);eligible={tuple(w) for w in d.record(a)['eligible_gain_workloads']}
  state['confirmation_result']={k:v for k,v in result.items() if k!='rows'}
  if not eligible.intersection(map(tuple,result['confirmed_local_gains'])) or result['resolved_regressions']:
   state.update(status='confirmation_needs_review',finished_unix=time.time());d.save(path,state);return
  state['status']='preparing_validation';d.save(path,state);validation.prepare(a)
  state.update(status='validation_running',validation_plan=str(d.C/a/'validation-plan.json'));d.save(path,state)
  validation.run(a);finished=d.read(d.C/a/'validation-controller.json')
  state.update(status=finished['status'],finished_unix=time.time());d.save(path,state)
 except BaseException as e:
  state.update(status='stopped_needs_review',error=repr(e),finished_unix=time.time());d.save(path,state);raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);args=p.parse_args();main(args.attempt)
