"""End-to-end provenance, coverage, selection, fixed-sampling and source audit."""
import json,subprocess
from pathlib import Path
import study as s
import confirm
D=s.D;C=s.C

def audit():
 plan=s.read(D/'plan.json');assert plan['jobs']==460 and len(plan['attempts'])==18
 assert s.read(D/'workflow-controller.json')['status']=='measurements_complete_pending_report'
 checked=[];environments=set();nominated={}
 for a in s.ATTEMPTS:
  p=s.read(D/a/'screen/manifest.json');assert p['status']=='completed'
  assert len(p['jobs'])==4*len(s.shapes(a))
  keys={(j['mode'],j['batch'],j['length']) for j in p['jobs']};assert keys==set(s.shapes(a))
  summary=s.summarize(D/a/'screen');history=confirm.historical(a)
  nominated[a]={(r['mode'],r['batch'],r['length']) for r in summary['rows']+history if r['screen_promising']}
 for path in D.glob('A*/*/manifest.json'):
  p=s.read(path)
  assert p['status']=='completed',path
  if p['formal']:s.validate_balanced_order(p['jobs'])
  for sig in p['source_signatures'].values():assert s.signature(sig['attempt'])==sig
  if p['formal']:
   shape={(j['mode'],j['batch'],j['length']) for j in p['jobs']}
   if path.parent.name=='confirmation-parent':assert shape==nominated[p['attempt']]
   else:
    parent=s.read(D/p['attempt']/'confirmation-parent/summary.json')
    assert shape=={(r['mode'],r['batch'],r['length']) for r in parent['rows'] if confirm.passes(r)}
   assert len(p['jobs'])==12*len(shape)
  for j in p['jobs']:
   assert j['status']=='completed' and j['returncode']==0
   raw=path.parent/(j['name']+'.json');env,row=s.audit_job(j,raw);environments.add(env);checked.append(str(raw))
   assert j['finished_unix']>=j['started_unix']
   commit_time=int(subprocess.check_output(['git','show','-s','--format=%ct',j['candidate_sha']],cwd=s.root(j['attempt']),text=True))
   assert commit_time<=j['started_unix']
 assert len(environments)==1
 cp=s.read(D/'confirmation-plan.json');expected=sum(len(shapes)*(1 if s.record(a)['parent_attempt_id']=='A0016' else 2) for a,shapes in nominated.items())
 assert cp['family_size']==expected
 for a,shapes in nominated.items():
  if shapes:assert (D/a/'confirmation-parent/summary.json').exists()
 confirm.analyze();summary=s.read(D/'confirmation-summary.json')
 assert len(summary['family'])==expected
 for f in summary['family']:
  if f['status']!='measured':
   assert f['comparator']=='current' and f['p_for_multiplicity']==1 and not f['confirmed_local_gain']
  else:
   r=f['result'];assert f['confirmed_local_gain']==(confirm.passes(r) and f['holm_adjusted_p']<.05)
 # Source-equivalence and historical evidence checks, excluding new report/helper files.
 repo=C.parents[3];retained=s.record('A0016')['candidate_sha']
 assert s.source_hash(repo)==s.source_hash(s.root('A0016'))
 historical=[str(p.relative_to(repo)) for p in C.glob('A[0-9][0-9][0-9][0-9]')]+[str((C/'final').relative_to(repo)),str((C/'continuation-01/final').relative_to(repo)),str((C/'continuation-02/final').relative_to(repo))]
 assert not subprocess.check_output(['git','diff','0afe89bf6e1f69db4036f00ce8e3c9877889c2cd','--',*historical],cwd=repo)
 result=dict(status='passed',campaign_id='offload-gap-001',study='supplemental-01',screen_processes=460,checked_new_raw_results=len(checked),independent_confirmation_processes=len(checked)-460,nominees=sum(map(len,nominated.values())),fixed_confirmation_family=expected,confirmed_local_comparisons=summary['confirmed_local_comparisons'],retained_source=retained,historical_evidence_unchanged=True,execution_source_unchanged=True,environment_classes=len(environments),source_results=checked,limitations='No new attempt accepted. Local performance evidence does not replace full applicable correctness/integration/regression gates. Shape omissions and n=3 assumptions remain explicit.')
 s.save(D/'final/audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_results'}))
if __name__=='__main__':audit()
