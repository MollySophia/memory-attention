"""Wait for this exact screen controller, then run the predeclared confirmation.
Never restart GPU jobs; absence of a live expected process before completion stops.
"""
import json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent
from study import read,save
statepath=D/'workflow-controller.json';assert not statepath.exists()
screen=read(D/'screen-controller.json');pid=screen['pid']
def identity(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
 except FileNotFoundError:return None
start=identity(pid);assert start is not None or screen['status']=='completed'
state=dict(status='waiting_for_screen',pid=os.getpid(),screen_pid=pid,screen_starttime=start,started_unix=time.time(),stages=[]);save(statepath,state)
while True:
 screen=read(D/'screen-controller.json')
 if screen['status']=='completed':break
 if identity(pid)!=start:
  state.update(status='stopped_screen_controller_exited',screen_snapshot=screen);save(statepath,state);raise SystemExit(1)
 time.sleep(10)
for action in ('prepare','run'):
 command=[sys.executable,str(D/'confirm.py'),action];entry=dict(action=action,command=command,started_unix=time.time())
 state.update(status='confirmation_'+action);state['stages'].append(entry)
 with (D/('confirmation-'+action+'-controller.txt')).open('w') as log:
  child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);entry['pid']=child.pid;save(statepath,state);entry['returncode']=child.wait()
 entry['finished_unix']=time.time();save(statepath,state)
 if entry['returncode']:
  state.update(status='stopped_confirmation_needs_review');save(statepath,state);raise SystemExit(1)
state.update(status='measurements_complete_pending_report',finished_unix=time.time());save(statepath,state)
