import sys,json,math,statistics
from pathlib import Path
D=Path(__file__).resolve().parent.parent/'continuation-03';sys.path.insert(0,str(D))
import driver as d
import confirm
base=d.C/'A0031';directory=base/'R01-complete-screen';p=d.read(directory/'manifest.json')
assert p['status']=='completed' and len(p['jobs'])==64
cells=set();envs=set()
for j in p['jobs']:
 assert j['status']=='completed' and j['returncode']==0
 cell=tuple(j[k] for k in ('mode','batch','length','implementation','variant'))
 assert cell not in cells;cells.add(cell)
 env,row=d.audit_job(j,directory/(j['name']+'.json'));envs.add(env)
 assert math.isclose(row['mean_ms'],statistics.mean(statistics.mean(xs) for xs in row['samples_ms']),rel_tol=1e-12,abs_tol=1e-12)
assert cells=={(*w,i,v) for w in d.WORKLOADS for i in ('baseline','candidate') for v in ('ma_offload','ma_gpu')}
assert len(envs)==1
old=d.read(directory/'summary.json');new=d.summarize(directory)
assert old==json.loads(json.dumps(new))
selected=confirm.nominations(new['rows']);formal=d.read(base/'R02-parent-confirmation/manifest.json')
assert d.planned_blocks(formal)==6 and formal['fixed_gain_family']==len(selected)==8
assert formal['source_signatures']==p['source_signatures']
assert {(r['mode'],r['batch'],r['length']):r['reasons'] for r in formal['nomination_reasons']}==selected
assert len(formal['jobs'])==192
assert d.validate_balanced_order(formal['jobs'],6)
expected={(*w,b,i,v) for w in selected for b in range(1,7) for i in ('baseline','candidate') for v in ('ma_offload','ma_gpu')}
actual=[tuple(j[k] for k in ('mode','batch','length','block','implementation','variant')) for j in formal['jobs']]
assert len(actual)==len(set(actual)) and set(actual)==expected
for j in formal['jobs']:
 c=j['command'];assert c[c.index('--rounds')+1]=='3'
 assert c[c.index('--repeats')+1]==('5' if j['mode']=='generation' else '10')
 assert c[c.index('--warmup')+1]==('2' if j['mode']=='generation' else '10')
result=dict(status='passed',attempt='A0031',screen_processes=64,screen_workloads=16,raw_means_recomputed=True,one_environment=True,complete_unique_coverage=True,confirmation_blocks=6,confirmation_workloads=len(selected),confirmation_processes=len(actual),confirmation_estimated_seconds=formal['planned_work']['estimated_seconds'],fixed_family=8,nomination_reasons=formal['nomination_reasons'],balanced_confirmation_coverage=True,accepted=False,note='Screen complete and confirmation plan audited; no confirmed gain or retention decision yet.')
d.save(base/'screen-audit.json',result)
print(json.dumps(result,indent=2))
for r in new['rows']:
 print(r['mode'],r['batch'],r['length'],'offload reduction',r['statistics']['offload_reduction_ms']['mean'],'gap reduction',r['statistics']['gap_reduction_ms']['mean'])
