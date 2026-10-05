import sys,json,contextlib,io
from pathlib import Path
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D))
import driver as d
import confirm
import audit_completed
saved=[]
original_save=d.save
d.save=lambda path,payload: None
for number in range(21,30):
 a=f'A{number:04d}'
 for stage in ('R01-complete-screen','R02-parent-confirmation','R03-remaining-confirmation'):
  path=d.C/a/stage/'summary.json'
  if not path.exists():continue
  old=d.read(path)
  new=d.summarize(path.parent)
  assert json.loads(json.dumps(new))==old,(a,stage)
 with contextlib.redirect_stdout(io.StringIO()):new=confirm.analyze(a)
 assert json.loads(json.dumps(new))==d.read(d.C/a/'R02-parent-confirmation/analysis.json'),a
 saved.append(a)
counts={a:audit_completed.audit_manifests(a) for a in ('A0023','A0028')}
result=dict(status='passed',historical_screen_and_formal_summaries_identical=saved,historical_confirmation_analyses_identical=saved,accepted_full_matrix_integrity=counts,note='Read-only recomputation. Historical files and verdicts unchanged. Six-block helpers tested separately on synthetic plans.')
original_save(D/'A0030-helper-history-audit.json',result)
print(json.dumps(result))
