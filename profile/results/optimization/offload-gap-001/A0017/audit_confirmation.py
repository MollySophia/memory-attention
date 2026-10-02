"""Audit the fixed A0017 parent confirmation without changing its verdict."""
import json,sys
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parent;sys.path.insert(0,str(root))
from run_paired import source_hash,validate_balanced_order
from audit_continuation import validate_offload_memory
m=json.loads((here/'R02-parent-confirmation/manifest.json').read_text())
assert m['status']=='completed' and len(m['jobs'])==48
validate_balanced_order(m['jobs']);hashes={};rows=[];envs=set()
for j in m['jobs']:
 p=here/'R02-parent-confirmation'/(j['name']+'.json');d=json.loads(p.read_text());row=d['results'][0]
 assert j['status']==d['status']=='completed'
 assert d['source']['git_commit']['stdout'].strip()==j['candidate_sha']
 if j['cwd'] not in hashes:hashes[j['cwd']]=source_hash(Path(j['cwd']))
 assert d['source']['source_sha256']==hashes[j['cwd']]
 assert d['stage']=='confirmation' and d['measurement_plan_id']=='formal_v1_w10_n10_r3'
 assert d['config']['warmup']==10 and len(row['samples_ms'])==3 and all(len(x)==10 for x in row['samples_ms'])
 e=d['env'];b=d['environment_before']
 envs.add(tuple(e[k] for k in ('torch','torch_cuda','flash_attn','gpu','python','cuda_visible_devices'))+(tuple(b['cpu_affinity']),b['torch_threads'],b['torch_interop_threads'],json.dumps(b['thread_environment'],sort_keys=True),b['gpu_telemetry']['stdout'].splitlines()[1].split(', ')[1]))
 if j['variant']=='ma_offload':validate_offload_memory('A0017' if j['implementation']=='candidate' else 'A0016',row['memory_after'])
 rows.append(dict(path=str(p),implementation=j['implementation'],source_sha=j['candidate_sha'],memory=row['memory_after']))
assert len(envs)==1
(here/'confirmation-memory-audit.json').write_text(json.dumps(dict(campaign_id='offload-gap-001',attempt_id='A0017',status='passed',checked_raw_results=48,environment_signature=next(iter(envs)),jobs=rows),indent=2)+'\n')
print('48 source, environment, sampling and memory checks passed')
