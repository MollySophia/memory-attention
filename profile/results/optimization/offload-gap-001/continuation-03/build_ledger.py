"""Snapshot every finalized campaign record without revising historical verdicts."""
import argparse
import csv
import json
from pathlib import Path

D = Path(__file__).resolve().parent
C = D.parent


def build(through):
    last = int(through[1:])
    records = []
    for number in range(last + 1):
        attempt = f'A{number:04d}'
        record = json.loads((C / attempt / 'record.json').read_text())
        assert record['attempt_id'] == attempt
        assert record['campaign_id'] == 'offload-gap-001'
        if number:
            assert record['workflow_stage'] in ('final_verdict', 'verdict')
            assert record['status'] in ('accepted', 'rejected', 'within_noise',
                                        'correctness_failed', 'benchmark_failed', 'oom', 'unsupported')
        records.append(record)
    retained = [r for r in records if r.get('accepted_step') is not None]
    assert [r['accepted_step'] for r in retained] == list(range(len(retained)))
    out = D / f'ledger-through-{through}'
    out.mkdir(exist_ok=True)
    (out / 'attempt-ledger.json').write_text(json.dumps(dict(
        campaign_id='offload-gap-001', through=through, status='finalized_attempt_snapshot',
        final_goal_accepted=False, records=records), indent=2) + '\n')
    fields = ['attempt_id', 'label', 'status', 'parent_attempt_id', 'candidate_sha',
              'accepted_step', 'workflow_stage', 'record_path']
    with (out / 'attempt-ledger.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {k: record.get(k) for k in fields}
            row['record_path'] = f"{record['attempt_id']}/record.json"
            writer.writerow(row)
    lines = [f'# Complete campaign ledger through {through}', '',
             f'A0000 is the frozen baseline; A0001–{through} are {last} finalized attempts. '
             f'There are {len(retained) - 1} retained optimization steps. '
             'This snapshot does not claim final all-workload acceptance.', '',
             '| Attempt | Change | Parent | Historical verdict | Accepted step | Source |',
             '| --- | --- | --- | --- | ---: | --- |']
    for r in records:
        attempt = r['attempt_id']
        label = r.get('label', '').replace('|', '\\|').replace('\n', ' ')
        step = r.get('accepted_step')
        lines.append(f"| [{attempt}](../../{attempt}/record.json) | {label} | "
                     f"{r.get('parent_attempt_id') or '—'} | {r['status']} | "
                     f"{step if step is not None else '—'} | `{r.get('candidate_sha', 'unavailable')}` |")
    lines += ['',
              '[JSON](attempt-ledger.json) preserves every authoritative record field, including '
              'correctness, failures, incomplete stages, adverse evidence and verdict rationale. '
              '[CSV](attempt-ledger.csv) provides an index. Raw timings and exact commands remain '
              'under the linked attempt directories.', '',
              'The [supplemental survey](../../supplemental-01/final/REPORT.md) preserves later '
              'recovered local findings without changing historical decisions. The '
              '[verified chain](../CHAIN.md) links these findings to retained and unintegrated '
              'mechanisms. Local effectiveness and whole-candidate retention are separate decisions.', '',
              'Reproduce this snapshot from the repository root with:', '', '```sh',
              f'python profile/results/optimization/offload-gap-001/continuation-03/build_ledger.py --through {through}',
              '```']
    (out / 'LEDGER.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(output=str(out), records=len(records), retained_steps=len(retained)-1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--through', required=True)
    build(parser.parse_args().through)
