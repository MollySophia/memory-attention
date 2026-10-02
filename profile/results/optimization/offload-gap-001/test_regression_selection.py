import pytest
from select_regression_checks import select, SHAPES


def fixture(stage):
    workloads=([(mode,batch,length) for mode in ('prefill','decode') for batch,length in SHAPES]
               if stage=='full_validation' else [('generation',batch,2048) for batch in (1,8)])
    return dict(points=[dict(mode=mode,batch=batch,length=length,implementation=impl,variant=variant,
                            status='completed',independent_processes=1,latency_ms=10.)
        for mode,batch,length in workloads for impl in ('baseline','candidate') for variant in ('ma_offload','ma_gpu')])


def test_any_adverse_secondary_is_selected_but_primary_is_not_extended():
    summary=fixture('full_validation')
    for row in summary['points']:
        if row['implementation']=='candidate' and row['mode']=='prefill' and row['length']==2048:
            if row['batch']==1 or row['batch']==4 and row['variant']=='ma_offload' or row['batch']==16 and row['variant']=='ma_gpu':
                row['latency_ms']=10.0001
    result=select(summary,'full_validation')
    assert len(result['reviewed'])==10
    assert result['selected']==['prefill:4:2048','prefill:16:2048']
    assert result['original_pair_retained'] and result['extra_pairs_per_placement']==2


@pytest.mark.parametrize('failure', ['repeated','failed'])
def test_already_repeated_or_failed_points_cannot_be_reselected(failure):
    summary=fixture('generation_validation')
    if failure=='repeated':summary['points'][0]['independent_processes']=3
    else:summary['points'][0]['status']='oom'
    with pytest.raises(AssertionError):select(summary,'generation_validation')


def test_both_generation_shapes_reviewed_without_invented_improvement():
    result=select(fixture('generation_validation'),'generation_validation')
    assert len(result['reviewed'])==2 and result['selected']==[]
