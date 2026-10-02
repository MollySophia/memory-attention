"""Check actual frozen memory records and reject missing allocation classes."""
import copy
import json
from pathlib import Path
import pytest
from audit_continuation import validate_offload_memory

ROOT = Path(__file__).resolve().parent

@pytest.mark.parametrize('attempt', ['A0008','A0009','A0010','A0011','A0012','A0013','A0014','A0015','A0016'])
def test_completed_screen_memory(attempt):
    paths = list((ROOT / attempt / 'R01').glob('*ma_offload*.json'))
    assert len(paths) == 6
    for path in paths:
        validate_offload_memory(attempt, json.loads(path.read_text())['results'][0]['memory_after'])

@pytest.mark.parametrize('attempt,kind', [('A0011','ids'),('A0013','table'),('A0014','depth'),('A0015','inverse'),('A0016','depth')])
def test_omitted_or_wrong_capacity_rejected(attempt, kind):
    path = ROOT / attempt / 'R01/J05-prefill-ma_offload-b8-l2048.json'
    memory = copy.deepcopy(json.loads(path.read_text())['results'][0]['memory_after'])
    cap = memory['offloader_capacities'][0]
    if kind == 'ids':
        memory['offload_pinned_bytes'] -= cap['batch'] * cap['length'] * 8
        cap['host_bytes'] -= cap['batch'] * cap['length'] * 8
    elif kind == 'table':
        memory['offload_pinned_bytes'] -= memory.pop('cpu_table_pinned_bytes')
    elif kind == 'depth':
        increment = 3 * cap['host_bytes']
        cap['host_bytes'] += increment
        cap['gpu_bytes'] += increment
        memory['offload_pinned_bytes'] += increment
        memory['offload_gpu_buffer_bytes'] += increment
    else:
        increment = cap['batch'] * cap['length'] * 8
        cap['gpu_bytes'] -= increment
        memory['offload_gpu_buffer_bytes'] -= increment
    with pytest.raises(AssertionError):
        validate_offload_memory(attempt, memory)
