import copy
import json
from pathlib import Path
import pytest
from reuse_correctness import validate_reference, reference_index
from run_paired import source_hash

ROOT = Path(__file__).resolve().parent
BASELINE = Path('/home/molly/workspace-memory-attn/offload-gap-001-A0000')
DIRECTORY = ROOT / 'A0001/R06-full-correctness'
CURRENT = ROOT / 'A0016/R03-baseline-confirmation/B1-J001-baseline-prefill-ma_offload-b1-l2048.json'


def test_all_four_frozen_reference_fingerprints_validate():
    assert len(reference_index(DIRECTORY,BASELINE,json.loads(CURRENT.read_text()))) == 4


@pytest.mark.parametrize('change', ['seed','source','threads','config','missing_step','missing_hidden','nonfinite','shape'])
def test_reference_mismatch_is_rejected(change):
    payload = json.loads((DIRECTORY/'correctness-b1-s1234-baseline.json').read_text())
    current = json.loads(CURRENT.read_text())
    sha = payload['source']['git_commit']['stdout'].strip()
    digest = source_hash(BASELINE)
    if change == 'seed': payload['seed'] = 4321
    elif change == 'source': payload['source']['source_sha256'] = 'bad'
    elif change == 'threads': payload['environment']['torch_threads'] += 1
    elif change == 'config': payload['model_config']['hidden_size'] = 1024
    elif change == 'missing_step': payload['checkpoints'].pop()
    elif change == 'missing_hidden': payload['checkpoints'][0]['hidden_states'].pop()
    elif change == 'nonfinite': payload['checkpoints'][1]['kv_states'][0][0]['finite'] = False
    else: payload['checkpoints'][2]['logits']['shape'] = [1,32000]
    with pytest.raises(AssertionError):
        validate_reference(payload,current,BASELINE,sha,digest,1,1234)
