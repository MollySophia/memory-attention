"""Audit the completed fixed six-block confirmation; no automatic retention."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'continuation-03'))
import driver as d
import confirm

folder=BASE/'R02-parent-confirmation';manifest=d.read(folder/'manifest.json')
assert manifest['status']=='completed' and d.planned_blocks(manifest)==6
assert manifest['fixed_gain_family']==10 and len(manifest['jobs'])==240
assert d.validate_balanced_order(manifest['jobs'],6)
for sig in manifest['source_signatures'].values():assert d.signature(sig['attempt'])==sig
selected={(r['mode'],r['batch'],r['length']) for r in manifest['nomination_reasons']}
assert selected==set(confirm.nominations(d.read(BASE/'R01-complete-screen/summary.json')['rows']))
cells=[];envs=set();raw_files=[]
for j in manifest['jobs']:
 assert j['status']=='completed' and j['returncode']==0
 path=folder/(j['name']+'.json');env,row=d.audit_job(j,path);envs.add(env)
 assert math.isclose(row['mean_ms'],statistics.mean(statistics.mean(xs) for xs in row['samples_ms']),rel_tol=1e-12,abs_tol=1e-12)
 cells.append(tuple(j[k] for k in ('mode','batch','length','block','implementation','variant')))
 raw_files.append(dict(name=j['name'],raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert len(envs)==1
assert len(cells)==len(set(cells)) and set(cells)=={(*w,b,i,v) for w in selected for b in range(1,7) for i in ('baseline','candidate') for v in ('ma_offload','ma_gpu')}
old=d.read(folder/'analysis.json');new=confirm.analyze('A0030')
assert json.loads(json.dumps(new))==old
assert all(r['statistics']['offload_reduction_ms']['n']==6 for r in new['rows'])
remaining=d.read(BASE/'R03-remaining-confirmation/manifest.json')
assert d.planned_blocks(remaining)==6
assert remaining['source_signatures']==manifest['source_signatures']
assert len(remaining['jobs'])==144 and d.validate_balanced_order(remaining['jobs'],6)
rest={(j['mode'],j['batch'],j['length']) for j in remaining['jobs']}
assert not rest.intersection(selected) and rest|selected==set(d.WORKLOADS)
assert len({tuple(j[k] for k in ('mode','batch','length','block','implementation','variant')) for j in remaining['jobs']})==144
result=dict(status='passed',attempt='A0030',fixed_gain_family=10,independent_blocks=6,checked_raw_results=240,complete_unique_coverage=True,balanced_order=True,one_environment=True,raw_means_recomputed=True,analysis_recomputed_identical=True,confirmed_local_gains=new['confirmed_local_gains'],resolved_regressions=new['resolved_regressions'],remaining_validation_workloads=sorted(rest),remaining_validation_processes=144,raw_files=raw_files,accepted=False,note='Two confirmed local gains versus A0028; complete remaining all16 validation, folding and full-model correctness before retention. Selection data are not final target evidence.')
d.save(BASE/'confirmation-audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='raw_files'},indent=2))
