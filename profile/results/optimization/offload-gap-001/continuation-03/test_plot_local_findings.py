"""Counts remain explicit when historical3-block and new6-block findings coexist."""
import plot_local_findings as p


def test_historical_metrics_unchanged_with_explicit_counts():
    rows=p.collect('A0029')
    old=p.d.read(p.d.D/'local-findings-through-A0029/findings.json')['rows']
    assert len(rows)==len(old)==84
    for row,prior in zip(rows,old):
        assert row['independent_blocks']==3 and row['degrees_of_freedom']==2
        assert {k:v for k,v in row.items() if k not in ('independent_blocks','degrees_of_freedom')}==prior


def test_synthetic_six_block_export_records_count_and_df(monkeypatch):
    # Synthetic finalized record only; this does not finalize the live A0030.
    monkeypatch.setattr(p.d,'record',lambda a:dict(status='accepted',confirmation_blocks=6))
    pairs=[dict(baseline_offload_ms=110.,baseline_gpu_ms=100.,candidate_offload_ms=109.,candidate_gpu_ms=100.) for _ in range(6)]
    row=dict(mode='decode',batch=8,length=2048,latencies=pairs,joint_one_sided_p=0.,holm_adjusted_p=0.,
             confirmed_local_gain=True,memory_savings_preserved=True,resolved_offload_regression=False,
             resolved_gap_regression=False,resolved_resident_slowdown=False,
             source_manifest=str(p.d.C/'A0021/R02-parent-confirmation/manifest.json'),
             statistics={metric:p.d.interval([value]*6) for metric,value in zip(p.METRICS,(1.,1.,0.))})
    monkeypatch.setattr(p.d,'read',lambda path:dict(status='completed',fixed_gain_family=1,baseline='A0016',rows=[row]))
    result=p.collect('A0021')
    assert len(result)==1
    assert result[0]['independent_blocks']==6 and result[0]['degrees_of_freedom']==5
    assert result[0]['offload_reduction_ms_mean']==1.
