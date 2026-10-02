"""Guard paired inference and preservation of original unfavorable evidence."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_regression import analyze
from merge_validation_evidence import merge
from collect_matrix import collect


def test_generation_audit_keeps_original_unfavorable_pair():
    result = analyze(ROOT / 'A0001/R07-generation-regression/manifest.json')
    stat = result['rows'][0]['statistics']['offload_reduction_ms']
    assert stat['n'] == 3
    assert stat['values'][0] < -40
    assert stat['lower_95'] < 0 < stat['upper_95']


def test_missing_pair_cannot_pass(tmp_path):
    source = ROOT / 'A0001/R07-generation-regression/manifest.json'
    manifest = json.loads(source.read_text())
    manifest['jobs'].pop()
    target = tmp_path / 'manifest.json'
    target.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        analyze(target)


def test_derived_manifest_preserves_three_sources_and_unaffected_points(tmp_path):
    original = ROOT / 'A0001/R05-generation-validation/manifest.json'
    audit = ROOT / 'A0001/R07-generation-regression/manifest.json'
    before = original.read_bytes()
    derived = merge(original, audit)
    assert original.read_bytes() == before
    assert len(derived['jobs']) == 12
    destination = tmp_path / 'derived.json'
    destination.write_text(json.dumps(derived))
    exported = collect(destination)
    for point in exported['points']:
        expected = 3 if point['batch'] == 1 and point['variant'] != 'ma_gpu_unfolded' else 1
        assert point['independent_processes'] == expected
        assert len(point['source_results']) == expected
        assert point['raw_sample_count'] == expected * 15
