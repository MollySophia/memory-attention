import driver as d

def test_complete_scope_has_no_duplicate_or_missing_workload():
 assert len(d.WORKLOADS)==len(set(d.WORKLOADS))==16
 assert {(b,l) for m,b,l in d.WORKLOADS if m=='decode'}==set(d.SHAPES)
 assert {(b,l) for m,b,l in d.WORKLOADS if m=='prefill'}==set(d.SHAPES)
 assert {(b,l) for m,b,l in d.WORKLOADS if m=='generation'}=={(1,2048),(8,2048)}

def test_screen_plans_all_placements_and_short_full_trajectories(tmp_path):
 p=d.plan('A0021',d.WORKLOADS,'A0016',False,tmp_path)
 assert len(p['jobs'])==64
 for j in p['jobs']:
  c=j['command'];assert c[c.index('--generation-steps')+1]=='128'
  assert c[c.index('--rounds')+1]=='1'
  if j['mode']=='generation':
   assert c[c.index('--repeats')+1]=='3'
   assert c[c.index('--warmup')+1]=='2'
 assert len({(j['mode'],j['batch'],j['length'],j['implementation'],j['variant']) for j in p['jobs']})==64

def test_confirmation_blocks_remain_balanced_and_separate(tmp_path):
 p=d.plan('A0021',[('prefill',8,512),('generation',8,2048)],'A0016',True,tmp_path)
 assert len(p['jobs'])==24
 assert d.validate_balanced_order(p['jobs'])
 assert all(j['command'][j['command'].index('--rounds')+1]=='3' for j in p['jobs'])

def test_nomination_keeps_secondary_gains_primary_controls_and_regressions():
 from confirm import nominations,PRIMARY
 rows=[]
 for mode,b,l in d.WORKLOADS:
  rows.append(dict(mode=mode,batch=b,length=l,screen_promising=False,memory_savings_preserved=True,statistics={k:dict(mean=0.) for k in ('offload_reduction_ms','gap_reduction_ms','gpu_reduction_ms')},latencies=[dict(baseline_offload_ms=100.,baseline_gpu_ms=100.)]))
 assert set(nominations(rows))==PRIMARY
 gen=next(r for r in rows if r['mode']=='generation' and r['batch']==8);gen['screen_promising']=True
 adverse=next(r for r in rows if (r['mode'],r['batch'],r['length'])==('decode',8,8192));adverse['statistics']['offload_reduction_ms']['mean']=-1.01
 minor=next(r for r in rows if (r['mode'],r['batch'],r['length'])==('prefill',8,4096));minor['statistics']['offload_reduction_ms']['mean']=-.5
 assert set(nominations(rows))==PRIMARY|{('generation',8,2048),('decode',8,8192)}

def test_fixed_family_holm_does_not_drop_unfavorable_values():
 from confirm import holm
 import pytest
 assert holm([.001,.04,1.])==pytest.approx([.003,.08,1.])
