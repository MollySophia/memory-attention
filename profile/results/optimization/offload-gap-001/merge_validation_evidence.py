"""Reference bounded regression repeats in a derived validation manifest.

Original manifests and raw payloads are never rewritten. Each affected point
references all three processes, including the original unfavorable observation.
"""
import argparse
import copy
import json
from pathlib import Path
from analyze_regression import analyze


def merge(original_path, audit_path):
    analyze(audit_path)  # Validate every required point and paired plan first.
    original = json.loads(original_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert Path(audit['source_manifest']).resolve() == original_path.resolve()
    assert original['status'] == 'completed'
    result = copy.deepcopy(original)
    key = lambda j: tuple(j[k] for k in ('mode', 'batch', 'length', 'implementation', 'variant'))
    repeats = {}
    for job in audit['jobs']:
        path = Path(job['source_result']) if job['status'] == 'reused' else audit_path.parent / (job['name'] + '.json')
        repeats.setdefault(key(job), []).append(str(path.resolve()))
    for job in result['jobs']:
        if key(job) in repeats:
            paths = repeats.pop(key(job))
            assert len(paths) == len(set(paths)) == 3
            original_file = str((original_path.parent / (job['name'] + '.json')).resolve())
            assert original_file in paths
            job.update(status='reused', reused_results=paths,
                       reuse_reason='Original point plus exactly two predeclared independent regression pairs')
        elif not job.get('reused_results'):
            job['reused_results'] = [str((original_path.parent / (job['name'] + '.json')).resolve())]
    assert not repeats
    result.update(derived_from=str(original_path.resolve()), regression_audit=str(audit_path.resolve()),
                  evidence_note='Derived references only. Original commands/stages and every raw sample remain unchanged. Repeated points use all three independent processes.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('original', type=Path)
    p.add_argument('audit', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(json.dumps(merge(args.original, args.audit), indent=2) + '\n')
