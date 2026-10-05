"""Coverage and uncertainty must follow the predeclared independent block count."""
import pytest
import driver as d
import plot_workload_scaling as p


def test_six_block_plot_data_checks_coverage_and_uses_every_process(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'C', tmp_path)
    monkeypatch.setattr(d, 'record', lambda a: dict(status='accepted', confirmation_blocks=6))
    directory = tmp_path / 'A0030'
    directory.mkdir()
    manifest_path = directory / 'manifest.json'
    jobs, rows = [], []
    for mode, batch, length in d.WORKLOADS:
        latencies = [dict(candidate_gpu_ms=100.+block, candidate_offload_ms=101.+block) for block in range(1, 7)]
        rows.append(dict(mode=mode, batch=batch, length=length, latencies=latencies, source_manifest=str(manifest_path)))
        for block in range(1, 7):
            for variant in p.VARIANTS:
                jobs.append(dict(name=f'{mode}-{batch}-{length}-{block}-{variant}',
                                 mode=mode, batch=batch, length=length, block=block,
                                 attempt='A0030', implementation='candidate', variant=variant, status='completed'))
    manifest = dict(status='completed', formal=True, attempt='A0030', independent_blocks=6, jobs=jobs)
    d.save(manifest_path, manifest)
    d.save(directory/'full-parent-analysis.json', dict(status='audited', rows=rows))

    def raw(job, path):
        value = (100. if job['variant']=='ma_gpu' else 101.) + job['block']
        return ('same environment',), dict(mean_ms=value, memory_after={k: 100 for k in p.MEMORY})

    monkeypatch.setattr(d, 'audit_job', raw)
    actual, raw_rows = p.load('A0030')
    assert len(actual)==32 and len(raw_rows)==192
    row = next(r for r in actual if (r['mode'],r['batch'],r['length'],r['variant'])==('decode',1,2048,'ma_gpu'))
    interval = d.interval([101.,102.,103.,104.,105.,106.])
    assert row['blocks']==6
    for field in ('mean','lower_95','upper_95'):
        assert row['latency_ms_'+field]==interval[field]
    # Dropping the last block must not silently produce an n=5 figure.
    manifest['jobs'].pop()
    d.save(manifest_path,manifest)
    with pytest.raises(AssertionError):p.load('A0030')


def test_archived_three_block_plot_data_remains_exact():
    for attempt in ('A0023','A0028'):
        rows,raw=p.load(attempt)
        old=d.read(d.C/attempt/'workload-diagnostics/workloads.json')
        assert rows==old['rows'] and raw==old['raw_processes']
