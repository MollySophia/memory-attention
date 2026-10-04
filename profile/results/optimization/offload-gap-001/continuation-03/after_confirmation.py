"""Observe the existing confirmation process, then conditionally validate sequentially."""
import argparse,os,time,subprocess,sys
from pathlib import Path
import driver as d
import confirm
import validation

def identity(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
  return fields[0],fields[19]
 except FileNotFoundError:return None

def main(a):
 path=d.C/a/'continuation-controller.json';assert not path.exists()
 manifest=d.C/a/'R02-parent-confirmation/manifest.json';p=d.read(manifest)
 pid=p['controller_pid'];initial=identity(pid)
 assert initial is not None or p['status']=='completed'
 state=dict(status='waiting_existing_confirmation',controller_pid=os.getpid(),observed_pid=pid,observed_identity=initial,started_unix=time.time(),accepted=False);d.save(path,state)
 try:
  while initial is not None and initial[0]!='Z':
   current=identity(pid)
   if current is None or current[0]=='Z':break
   assert current[1]==initial[1],'Recorded confirmation PID was reused; inspect before proceeding'
   time.sleep(10)
  assert d.read(manifest)['status']=='completed','Recorded process exited without completed evidence; do not restart'
  result=confirm.analyze(a)
  eligible={tuple(w) for w in d.record(a)['eligible_gain_workloads']}
  if not eligible.intersection(map(tuple,result['confirmed_local_gains'])) or result['resolved_regressions']:
   state.update(status='confirmation_needs_review',confirmation_result={k:v for k,v in result.items() if k!='rows'});d.save(path,state);return
  state['status']='preparing_validation';d.save(path,state)
  validation.prepare(a)
  state['validation_plan']=str(d.C/a/'validation-plan.json');state['status']='validation_running';d.save(path,state)
  validation.run(a)
  finished=d.read(d.C/a/'validation-controller.json')
  state.update(status=finished['status'],finished_unix=time.time());d.save(path,state)
 except BaseException as e:
  state.update(status='stopped_needs_review',error=repr(e),finished_unix=time.time());d.save(path,state);raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);args=p.parse_args();main(args.attempt)
