import json,sys
from pathlib import Path
root=Path('/home/molly/workspace-memory-attn/memory-attention/profile/results/optimization/offload-gap-001');sys.path.insert(0,str(root))
from audit_continuation import validate_offload_memory
from run_paired import source_hash
p=root/'A0017';r=json.loads((p/'record.json').read_text());m=json.loads((p/'R01/manifest.json').read_text());sha=source_hash(Path(r['frozen_checkout']));rows=[]
assert m['status']=='completed' and len(m['jobs'])==12
for job in m['jobs']:
 d=json.loads((p/'R01'/(job['name']+'.json')).read_text());row=d['results'][0]
 assert job['status']==d['status']=='completed'
 assert d['source']['git_commit']['stdout'].strip()==r['candidate_sha']
 assert d['source']['source_sha256']==sha
 if job['variant']=='ma_offload':
  validate_offload_memory('A0016',row['memory_after']) # A0017 preserves parent capacities exactly.
 rows.append(dict(job=job['name'],source_hash_checked=True,memory=row['memory_after']))
(p/'screen-source-memory-audit.json').write_text(json.dumps(dict(campaign_id='offload-gap-001',attempt_id='A0017',status='passed',note='A0017 changes allocation flags only; all parent A0016 byte-capacity rules still apply.',jobs=rows),indent=2)+'\n')
r.update(workflow_stage='confirmation',screen_summary='R01-summary.json',performance='screen-comparison.json',screen_decision='Target b8 prefill offload/gap reductions2.020062/1.692548 ms versus A0016; b1/b16 decode slower0.370803/0.431870 ms preserved. Predeclare fixed48 parent confirmation; only b8 prefill may qualify.',parent_confirmation_manifest='R02-parent-confirmation/manifest.json')
r['not_run']['confirmation']='Fixed48-process parent confirmation now starting; no outcome yet.'
(p/'record.json').write_text(json.dumps(r,indent=2)+'\n');print('12 sources and parent-compatible capacities passed')
