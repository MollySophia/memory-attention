"""Check all A0019 screen sources and unchanged parent capacities."""
import json,sys
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parent;sys.path.insert(0,str(root))
from audit_continuation import validate_offload_memory
from run_paired import source_hash
r=json.loads((here/'record.json').read_text());m=json.loads((here/'R01/manifest.json').read_text());sha=source_hash(Path(r['frozen_checkout']));rows=[]
assert m['status']=='completed' and len(m['jobs'])==12
for j in m['jobs']:
 d=json.loads((here/'R01'/(j['name']+'.json')).read_text());row=d['results'][0]
 assert j['status']==d['status']=='completed'
 assert d['source']['git_commit']['stdout'].strip()==r['candidate_sha'] and d['source']['source_sha256']==sha
 if j['variant']=='ma_offload':validate_offload_memory('A0019',row['memory_after'])
 rows.append(dict(job=j['name'],memory=row['memory_after']))
(here/'screen-source-memory-audit.json').write_text(json.dumps(dict(campaign_id='offload-gap-001',attempt_id='A0019',status='passed',checked_raw_results=12,jobs=rows,note='No added buffers; exact inherited A0016 capacities and all cached allocations checked.'),indent=2)+'\n')
print('12 sources and all inherited slot capacities pass')
