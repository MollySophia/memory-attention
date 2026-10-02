"""Export the complete finalized campaign ledger from authoritative records."""
import csv,json
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parent;out=here/'final'
records=[json.loads(p.read_text()) for p in sorted(root.glob('A[0-9][0-9][0-9][0-9]/record.json'))]
assert len(records)==21
assert all(r['workflow_stage']=='final_verdict' for r in records[1:])
assert [r['attempt_id'] for r in records]==[f'A{i:04d}' for i in range(21)]
(out/'attempt-ledger.json').write_text(json.dumps(dict(campaign_id='offload-gap-001',continuation_id='continuation-02',records=records),indent=2)+'\n')
fields=['attempt_id','label','status','parent_attempt_id','candidate_sha','accepted_step','decision']
with (out/'attempt-ledger.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in records:w.writerow({k:r.get(k) for k in fields})
lines=['# Complete attempt ledger','', 'Campaign `offload-gap-001`. A0000 is the frozen baseline; A0001–A0020 are actual finalized attempts. Only A0011–A0020 belong to this ten-attempt continuation.','', '| Attempt | Change | Parent | Verdict | Accepted step | Source |','|---|---|---|---|---:|---|']
for r in records:
 name=r['attempt_id'];lines.append(f"| [{name}](../../{name}/record.json) | {r['label']} | {r.get('parent_attempt_id') or '—'} | {r.get('status','baseline')} | {r['accepted_step'] if r.get('accepted_step') is not None else '—'} | `{r['candidate_sha']}` |")
lines+=['','Full hypotheses, profiles, correctness logs, commands, raw results, uncertainties and reasons for unmeasured stages remain in each linked record. [Machine-readable ledger](attempt-ledger.json) preserves every field; [CSV](attempt-ledger.csv) provides the comparison columns.']
(out/'LEDGER.md').write_text('\n'.join(lines)+'\n')
print('Exported all21 records; ten additional finalized attempts')
