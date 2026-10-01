"""Check CLI sampling plans and their propagation into isolated job commands."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'profile'))
from sampling_plan import sampling_plan


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'profile'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(monkeypatch, mode='prefill', extra=()):
    bench = load('bench_fla')
    monkeypatch.setattr(sys, 'argv', ['bench_fla.py','--mode',mode,'--variants','ma_offload',
        '--batch-size','8','--seq-len','2048','--context-len','2048','--json','unused.json',*extra])
    return bench.parse_args()


@pytest.mark.parametrize('mode,extra,expected', [
    ('prefill', (), (3,5,1)), ('decode', (), (3,5,1)),
    ('generation', (), (2,5,3)),
    ('decode', ('--stage','confirmation'), (10,10,3)),
    ('prefill', ('--stage','full_validation'), (10,10,3)),
    ('generation', ('--stage','full_validation'), (2,5,3)),
    ('generation', ('--stage','legacy'), (30,30,5)),
    ('prefill', ('--stage','legacy'), (30,30,5)),
])
def test_cli_defaults(monkeypatch, mode, extra, expected):
    a = args(monkeypatch, mode, extra)
    assert (a.warmup,a.repeats,a.rounds) == expected
    assert a.measurement_plan_id


def test_explicit_override_is_identified(monkeypatch):
    a=args(monkeypatch, extra=('--warmup','0','--repeats','7'))
    assert (a.warmup,a.repeats,a.rounds)==(0,7,1)
    assert a.measurement_plan_id=='screen_v1_w0_n7_r1_custom'


@pytest.mark.parametrize('mode,extra', [
    ('generation',('--stage','screening')), ('prefill',('--stage','generation_validation')),
    ('decode',('--warmup','-1')), ('decode',('--repeats','0')),
])
def test_invalid_plans_rejected(monkeypatch,mode,extra):
    with pytest.raises(SystemExit): args(monkeypatch,mode,extra)


@pytest.mark.parametrize('stage,count', [('screening',8),('confirmation',8),('full_validation',48),('generation_validation',6),('legacy',48)])
def test_matrix_child_plan_matches_manifest(monkeypatch,stage,count):
    runner=load('run_paper_matrix');bench=load('bench_fla')
    jobs=list(runner.jobs(stage))
    assert len(jobs)==count
    for job in jobs:
        cmd=runner.command(job,Path('unused.json'),stage)
        monkeypatch.setattr(sys,'argv',cmd[1:])
        parsed=bench.parse_args()
        plan=sampling_plan(stage,job['mode'])
        assert all(getattr(parsed,k)==v for k,v in plan.items())
    if stage=='screening':
        assert all(j['batch'] in (1,8) and j['length']==2048 and j['mode']!='generation' for j in jobs)


def test_baseline_and_pipeline_screen_shapes():
    runner=load("run_paper_matrix")
    assert len(list(runner.jobs(initial_baseline=True))) == 16
    assert len(list(runner.jobs(include_batch16=True))) == 12
