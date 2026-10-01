"""Check uncertainty units and preserve failed points without numeric zeros."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('collector',ROOT/'collect_matrix.py')
collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)


def row():
    return dict(mean_ms=10.,round_ms=[5.,10.,15.],samples_ms=[[10.]*10]*3,
                gpu_parameter_mib=1,cpu_parameter_mib=1,
                memory_after={key:1024 for key in collector.MEMORY_FIELDS})


def test_process_uncertainty_does_not_use_rounds_as_independent_runs():
    result=collector.summarize_rows([row(),row(),row()],'prefill',1,2048)
    assert result['latency_low_ms']==result['latency_high_ms']==10
    assert result['independent_processes']==3 and result['raw_sample_count']==90
    assert 'independent process' in result['uncertainty_kind']


def test_one_process_round_range_is_not_confidence_interval():
    result=collector.summarize_rows([row()],'decode',8,2048)
    assert (result['latency_low_ms'],result['latency_high_ms'])==(5,15)
    assert result['tokens_per_second']==800
    assert 'NOT a confidence interval' in result['uncertainty_kind']


def test_oom_retained_without_zero_latency(tmp_path):
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps(dict(status='completed',stage='full_validation',jobs=[dict(
        implementation='candidate',mode='prefill',variant='ma_offload',batch=8,
        length=8192,candidate_sha='test_fixture',name='oom',status='oom')])) )
    point=collector.collect(path)['points'][0]
    assert point['status']=='oom' and point['latency_ms'] is None
    assert point['tokens_per_second'] is None
