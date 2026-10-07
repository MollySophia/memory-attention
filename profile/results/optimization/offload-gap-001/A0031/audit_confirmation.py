"""Recompute fixed confirmation and verify complete raw coverage before archival."""
import hashlib,json,math,statistics,sys
from pathlib import Path
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base.parent/'continuation-03'))
import driver as d
import confirm
folder=base/'R02-parent-confirmation';p=d.read(folder/'manifest.json')
assert p['status']=='completed' and len(p['jobs'])==192
assert d.planned_blocks(p)==6 and p['fixed_gain_family']==8
assert d.validate_balanced_order(p['jobs'],6)
for sig in p['source_signatures'].values():assert d.signature(sig['attempt'])==sig
selected=confirm.nominations(d.read(base/'R01-complete-screen/summary.json')['rows'])
assert {(r['mode'],r['batch'],r['length']):r['reasons'] for r in p['nomination_reasons']}==selected
cells=[];envs=set();raw=[]
for j in p['jobs']:
 assert j['status']=='completed' and j['returncode']==0
 path=folder/(j['name']+'.json');env,row=d.audit_job(j,path);envs.add(env)
 assert math.isclose(row['mean_ms'],statistics.mean(map(statistics.mean,row['samples_ms'])),rel_tol=1e-12,abs_tol=1e-12)
 cells.append(tuple(j[k] for k in ('mode','batch','length','block','implementation','variant')))
 raw.append(dict(name=j['name'],sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert len(envs)==1 and len(cells)==len(set(cells))==192
assert set(cells)=={(*w,b,i,v) for w in selected for b in range(1,7) for i in ('baseline','candidate') for v in ('ma_offload','ma_gpu')}
old=d.read(folder/'analysis.json');new=confirm.analyze('A0031');assert json.loads(json.dumps(new))==old
out=dict(status='passed',attempt='A0031',checked_confirmation_results=192,screen_audit='screen-audit.json',independent_blocks=6,fixed_family=8,complete_unique_coverage=True,one_environment=True,raw_means_recomputed=True,analysis_recomputed_identical=True,confirmed_local_gains=new['confirmed_local_gains'],resolved_regressions=new['resolved_regressions'],raw_files=raw,accepted=False)
d.save(base/'confirmation-audit.json',out)
print(json.dumps({k:v for k,v in out.items() if k!='raw_files'},indent=2))
