"""Fresh all16 target verification, separate from candidate selection/retention.

Prepare only after choosing a retained source and fixed independent block counts.
Commit plan.json before run. No implicit resume, extension or evidence reuse.
This verifies the latency criterion, not all campaign deliverables.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from scipy.stats import t
import driver as d

ALPHA = .05
FAMILY = 16


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def helper_hashes():
    paths = [Path(__file__), d.D / 'driver.py', d.C / 'run_paired.py',
             d.C / 'audit_continuation.py', d.C / 'analyze_paired.py']
    return {str(p.relative_to(d.C)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def interval(values):
    assert len(values) >= 3 and all(math.isfinite(x) for x in values)
    n = len(values)
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    half = float(t.ppf(.975, n - 1)) * se
    return dict(mean=mean, lower_95=mean-half, upper_95=mean+half, n=n)


def assess(gpu, offload):
    assert len(gpu) == len(offload) and len(gpu) >= 3
    assert all(math.isfinite(x) and x > 0 for x in [*gpu, *offload])
    gaps = [o-g for o, g in zip(offload, gpu)]
    tolerances = [max(.1, .01*g) for g in gpu]
    residuals = [gap-tol for gap, tol in zip(gaps, tolerances)]
    n = len(residuals)
    upper = statistics.mean(residuals) + float(t.ppf(1-ALPHA/FAMILY, n-1)) * statistics.stdev(residuals)/math.sqrt(n)
    return dict(independent_pairs=n, gpu_ms=interval(gpu), offload_ms=interval(offload),
                gap_ms=interval(gaps), relative_overhead=interval([o/g-1 for o, g in zip(offload, gpu)]),
                tolerance_ms=tolerances, residual_ms=interval(residuals), residual_samples_ms=residuals,
                residual_upper_simultaneous_95_ms=upper, within_target=upper <= 0)


def location(attempt):
    return d.C / attempt / 'final-independent-target'


def block_counts(uniform=None, rows=None):
    assert (uniform is None) != (rows is None), 'Choose uniform or per-workload counts'
    if rows is None:
        rows = [dict(mode=m, batch=b, length=l, blocks=uniform) for m,b,l in d.WORKLOADS]
    counts = {}
    for row in rows:
        key = tuple(row[k] for k in ('mode', 'batch', 'length'))
        assert key not in counts, 'Duplicate workload'
        n = row['blocks']
        assert isinstance(n, int) and not isinstance(n, bool) and n >= 3
        counts[key] = n
    assert set(counts) == set(d.WORKLOADS) and len(counts) == FAMILY
    return counts


def prepare(attempt, counts):
    assert d.record(attempt)['status'] == 'accepted', 'Retained source required'
    counts = block_counts(rows=[dict(mode=w[0],batch=w[1],length=w[2],blocks=n) for w,n in counts.items()])
    directory = location(attempt)
    assert not directory.exists(), 'Never replace or extend an existing verification'
    sig = d.signature(attempt)
    jobs = []
    for block in range(1, max(counts.values()) + 1):
        sequence = list(enumerate(d.WORKLOADS))
        if block % 2 == 0:
            sequence.reverse()
        for wi, shape in sequence:
            if block > counts[shape]:
                continue
            variants = ['ma_offload', 'ma_gpu']
            if (block - 1 + wi) % 2:
                variants.reverse()
            for variant in variants:
                mode, batch, length = shape
                name = f'B{block}-J{len(jobs)+1:04d}-{mode}-{variant}-b{batch}-l{length}'
                command = d.cmd(attempt, shape, directory/(name+'.json'), True)
                command[command.index('PLACEHOLDER')] = variant
                jobs.append(dict(name=name, block=block, mode=mode, batch=batch, length=length,
                                 variant=variant, implementation='candidate', attempt=attempt,
                                 candidate_sha=sig['commit'], source_sha256=sig['source_sha256'],
                                 cwd=sig['root'], command=command,
                                 estimated_seconds=d.estimate(attempt, shape, variant, True)))
    plan = dict(purpose='final_independent_target', attempt=attempt, created_unix=time.time(),
                source_signatures={'candidate': sig}, helper_hashes=helper_hashes(),
                workloads=[dict(mode=w[0], batch=w[1], length=w[2], blocks=counts[w]) for w in d.WORKLOADS],
                criterion=dict(absolute_ms=.1, relative_fraction=.01, alpha=ALPHA, family=FAMILY,
                               rule='Per-pair signed residual; one-sided Bonferroni16 t upper bound <=0 at every workload'),
                stopping_rule='Exactly the frozen independent pairs; no optional extension, selection-data reuse or automatic acceptance',
                planned_work=dict(jobs=len(jobs), estimated_seconds=sum(j['estimated_seconds'] for j in jobs)), jobs=jobs)
    directory.mkdir()
    d.save(directory/'plan.json', plan)
    manifest = dict(attempt=attempt, status='planned', stage='final_independent_target',
                    source_signatures=plan['source_signatures'], plan_sha256=digest(plan),
                    jobs=[dict(j, status='pending') for j in jobs])
    d.save(directory/'manifest.json', manifest)
    print(json.dumps(plan['planned_work']))


def validate(plan, manifest):
    assert plan['purpose'] == manifest['stage'] == 'final_independent_target'
    assert manifest['attempt'] == plan['attempt']
    assert manifest['plan_sha256'] == digest(plan)
    assert plan['helper_hashes'] == helper_hashes(), 'Audit/runner changed after freeze'
    assert plan['source_signatures'] == manifest['source_signatures'] == {'candidate': d.signature(plan['attempt'])}
    assert d.record(plan['attempt'])['status'] == 'accepted'
    assert plan['criterion']['absolute_ms'] == .1 and plan['criterion']['relative_fraction'] == .01
    assert plan['criterion']['alpha'] == ALPHA and plan['criterion']['family'] == FAMILY
    counts = block_counts(rows=plan['workloads'])
    assert len(manifest['jobs']) == len(plan['jobs']) == 2 * sum(counts.values())
    seen, names = set(), set()
    orders = {}
    positions = {}
    sig = plan['source_signatures']['candidate']
    directory = location(plan['attempt'])
    for index, (expected, actual) in enumerate(zip(plan['jobs'], manifest['jobs'])):
        assert all(actual[k] == v for k,v in expected.items()), 'Changed planned job/order'
        shape = tuple(actual[k] for k in ('mode', 'batch', 'length'))
        block, variant = actual['block'], actual['variant']
        assert shape in counts and 1 <= block <= counts[shape]
        assert variant in ('ma_gpu', 'ma_offload')
        assert actual['name'] not in names and Path(actual['name']).name == actual['name'], 'Repeated/invalid output name'
        names.add(actual['name'])
        assert (actual['attempt'],actual['implementation'],actual['candidate_sha'],actual['source_sha256'],actual['cwd']) == (
            plan['attempt'],'candidate',sig['commit'],sig['source_sha256'],sig['root'])
        command = d.cmd(plan['attempt'],shape,directory/(actual['name']+'.json'),True)
        command[command.index('PLACEHOLDER')] = variant
        assert actual['command'] == command, 'Changed source, sampling or result command'
        cell = (*shape, block, variant)
        assert cell not in seen, 'Repeated evidence cell'
        seen.add(cell)
        orders.setdefault((shape, block), []).append(variant)
        positions.setdefault((shape, block), []).append(index)
    for wi, shape in enumerate(d.WORKLOADS):
        for block in range(1, counts[shape]+1):
            expected = ['ma_offload', 'ma_gpu']
            if (block - 1 + wi) % 2:
                expected.reverse()
            assert orders[shape, block] == expected, 'Unbalanced placement order'
            first, second = positions[shape, block]
            assert second == first + 1, 'Matched placements must run consecutively'
    return counts


def committed_plan(directory):
    path = (directory/'plan.json').resolve()
    repo = Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],cwd=d.D,text=True).strip())
    relative = str(path.relative_to(repo))
    committed = subprocess.check_output(['git','show',f'HEAD:{relative}'],cwd=repo)
    assert committed == path.read_bytes(), 'Commit the exact frozen plan before running'
    return subprocess.check_output(['git','log','-1','--format=%H','--',relative],cwd=repo,text=True).strip()


def run(attempt):
    directory = location(attempt)
    plan, manifest = [d.read(directory/name) for name in ('plan.json','manifest.json')]
    validate(plan, manifest)
    assert manifest['status'] == 'planned'
    assert all(j['status'] == 'pending' for j in manifest['jobs'])
    assert all(not (directory/(j['name']+ext)).exists() for j in plan['jobs'] for ext in ('.json','.txt'))
    manifest['frozen_plan_commit'] = committed_plan(directory)
    d.save(directory/'manifest.json', manifest)
    d.run(directory)
    audit(attempt)


def audit(attempt):
    directory = location(attempt)
    plan, manifest = [d.read(directory/name) for name in ('plan.json','manifest.json')]
    counts = validate(plan, manifest)
    assert manifest['status'] == 'completed'
    assert manifest['started_unix'] >= plan['created_unix']
    assert manifest['frozen_plan_commit'] == committed_plan(directory)
    data, environments = {}, set()
    for j in manifest['jobs']:
        assert j['status'] == 'completed' and j['returncode'] == 0
        assert j['started_unix'] >= manifest['started_unix']
        assert j['finished_unix'] >= j['started_unix']
        path = directory/(j['name']+'.json')
        assert path.resolve().parent == directory.resolve(), 'External/reused result path'
        env, row = d.audit_job(j, path)
        environments.add(env)
        recomputed = statistics.mean(statistics.mean(samples) for samples in row['samples_ms'])
        assert math.isclose(recomputed,row['mean_ms'],rel_tol=1e-12,abs_tol=1e-12)
        data[tuple(j[k] for k in ('mode','batch','length','block','variant'))] = row
    assert len(environments) == 1, 'Environment mismatch'
    rows = []
    for shape in d.WORKLOADS:
        gpu = [data[*shape,b,'ma_gpu']['mean_ms'] for b in range(1,counts[shape]+1)]
        offload = [data[*shape,b,'ma_offload']['mean_ms'] for b in range(1,counts[shape]+1)]
        savings = [data[*shape,b,'ma_gpu']['memory_after']['gpu_peak_allocated_bytes'] -
                   data[*shape,b,'ma_offload']['memory_after']['gpu_peak_allocated_bytes'] for b in range(1,counts[shape]+1)]
        rows.append(dict(mode=shape[0],batch=shape[1],length=shape[2],**assess(gpu,offload),
                         gpu_memory_savings_bytes=savings,memory_savings_preserved=min(savings)>0))
    result = dict(status='audited', attempt=attempt, source=plan['source_signatures']['candidate'],
                  plan_sha256=digest(plan), frozen_plan_commit=manifest['frozen_plan_commit'], rows=rows,
                  all_workloads_within_target=all(r['within_target'] for r in rows), goal_accepted=False,
                  all_memory_savings_preserved=all(r['memory_savings_preserved'] for r in rows),
                  note='Fresh fixed-plan latency evidence only; overall goal additionally requires cumulative baseline comparison and deliverables. Parametric t bounds assume independent approximately normal process-block differences.')
    d.save(directory/'analysis.json', result)
    with (directory/'summary.csv').open('w') as f:
        flat = [dict(mode=r['mode'],batch=r['batch'],length=r['length'],pairs=r['independent_pairs'],
                     gpu_ms=r['gpu_ms']['mean'],offload_ms=r['offload_ms']['mean'],gap_ms=r['gap_ms']['mean'],
                     mean_tolerance_ms=statistics.mean(r['tolerance_ms']),
                     residual_upper_simultaneous_95_ms=r['residual_upper_simultaneous_95_ms'],within_target=r['within_target']) for r in rows]
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source')}))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','audit'))
    parser.add_argument('--attempt',required=True)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--blocks',type=int)
    group.add_argument('--block-counts',type=Path,help='JSON list: mode,batch,length,blocks for all16 workloads')
    args=parser.parse_args()
    if args.action=='prepare':
        counts=block_counts(args.blocks,d.read(args.block_counts) if args.block_counts else None)
        prepare(args.attempt,counts)
    else:
        assert args.blocks is None and args.block_counts is None, 'Frozen plan controls all sampling'
        (run if args.action=='run' else audit)(args.attempt)
