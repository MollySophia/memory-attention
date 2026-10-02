import json,sys
from pathlib import Path
root=Path('/home/molly/workspace-memory-attn/memory-attention/profile/results/optimization/offload-gap-001')
sys.path.insert(0,str(root))
from run_paired import source_hash
from audit_continuation import validate_offload_memory
r=json.loads((root/'A0016/record.json').read_text())
b=json.loads((root/'A0000/record.json').read_text())
broot=json.loads((root/'manifest.json').read_text())['frozen_checkout']
hashes={'baseline':source_hash(Path(broot)),'candidate':source_hash(Path(r['frozen_checkout']))}
shas={'baseline':b['candidate_sha'],'candidate':r['candidate_sha']}
checked=[];mem=[]
for name in ('full-summary.json','generation-summary.json'):
 s=json.loads((root/'A0016'/name).read_text())
 points=s['points'];lookup={(p['implementation'],p['mode'],p['batch'],p['length'],p['variant']):p for p in points}
 for p in points:
  assert p['status'] in ('completed','reused')
  for path in p['source_results']:
   d=json.loads(Path(path).read_text());row=d['results'][0]
   assert d['status']=='completed'
   assert d['source']['source_sha256']==hashes[p['implementation']]
   assert d['source']['git_commit']['stdout'].strip()==shas[p['implementation']]
   if p['implementation']=='candidate' and p['variant']=='ma_offload':
    validate_offload_memory('A0016',row['memory_after'])
   checked.append(path)
  if p['variant']=='ma_offload':
   gpu=lookup[(p['implementation'],p['mode'],p['batch'],p['length'],'ma_gpu')]
   saving=gpu['gpu_peak_allocated_bytes_min']-p['gpu_peak_allocated_bytes_max']
   assert saving>0
   mem.append(dict(summary=name,implementation=p['implementation'],mode=p['mode'],batch=p['batch'],length=p['length'],conservative_gpu_peak_saving_bytes=saving))
result=dict(campaign_id='offload-gap-001',attempt_id='A0016',status='completed_original_matrix_source_and_memory_checks_passed',accepted=False,source_results=checked,checked_raw_results=len(checked),memory_savings=mem,limitation='Original matrices only; secondary repeats, complete final source/environment audit and retention verdict remain.')
(root/'A0016/original-matrix-memory-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],len(checked),'raw results; minimum saving MiB',min(m['conservative_gpu_peak_saving_bytes'] for m in mem)/2**20)
