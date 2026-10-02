"""Keep all twenty-one attempt labels visible without rendering during timing."""
from pathlib import Path
import plot_history as plots


def test_twenty_one_attempts_expand_layout_and_keep_all_ticks(tmp_path,monkeypatch):
    entries=[]
    for number in range(21):
        entries.append(dict(directory=Path('/unused'),record=dict(attempt_id=f'A{number:04d}',
            label='test fixture',status='accepted' if number<2 else 'rejected',
            accepted_step=number if number<2 else None),
            rows={key:dict(measurement_plan_id='layout_test',offload_ms=20-number/100,
                           offload_sample_sd_ms=.1) for key in plots.WORKLOADS}))
    checked=[]
    def inspect(fig,output,name,*args,**kwargs):
        assert fig.get_size_inches()[0]>=18
        for ax in fig.axes:
            assert [tick.get_text() for tick in ax.get_xticklabels()]==[f'A{i:04d}' for i in range(21)]
        checked.append(name)
        plots.plt.close(fig)
    monkeypatch.setattr(plots,'save',inspect)
    rows=plots.history(entries,tmp_path)
    assert len(rows)==84
    assert checked==['attempt-history-layout_test']
    assert all(row['incumbent_attempt']=='A0001' for row in rows if row['attempt']>'A0001')
