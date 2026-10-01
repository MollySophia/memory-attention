"""Diagnostic CPU/CUDA trace, separate from offload_gap_v1 wall-time measurements.

Run after the primary baseline controller exits; a full matrix is unnecessary.
Profiler overhead invalidates these
latencies for speedup claims. The offload producer thread may not be fully
captured by torch.profiler; use CUDA copies and end-to-end timings as well.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import time
import threading
from types import SimpleNamespace

import torch

from bench_fla import build, rollback, env_fingerprint
from benchmark_telemetry import environment_details, source_state, memory_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('pytorch', 'nsys', 'events', 'offload'), default='pytorch')
    parser.add_argument('--mode', choices=('prefill','decode'), required=True)
    parser.add_argument('--variant', choices=('ma_offload','ma_gpu'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True, help='Completed baseline manifest (primary screening is sufficient); concurrent profiling is refused')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Diagnostic warmup calls, outside capture (default: 10)')
    parser.add_argument('--batch-size', type=int, choices=(1,8), required=True)
    args=parser.parse_args()
    if args.warmup < 0:
        parser.error('--warmup must be nonnegative')
    manifest=json.loads(args.matrix.read_text())
    if manifest['status'] != 'completed' or any(j['status'] in ('running','pending') for j in manifest['jobs']):
        raise RuntimeError('Baseline matrix has not finished; profiling would contaminate it')
    args.output.mkdir(parents=True,exist_ok=False)
    config=SimpleNamespace(seed=1234, hidden_size=2048, num_layers=24, num_heads=32,
                           num_kv_heads=32, vocab_size=32000, hidden_ratio=4,
                           intermediate_size=5632, group_size=1, prefetch_depth=4,
                           policy='auto', device='cuda:0')
    source=source_state()
    environment=environment_details()
    model=build(config)
    try:
        if args.variant=='ma_offload':
            model.enable_memory_offload()
            model.set_offload_offloader(args.batch_size,2048)
            model.set_offload_offloader(args.batch_size,1)
        else:
            model.fold_memory_table_on_gpu()
        torch.manual_seed(1235)
        prefix=torch.randint(0,32000,(args.batch_size,2048),device='cuda')
        token=torch.randint(0,32000,(args.batch_size,1),device='cuda')
        cache=None
        if args.mode=='decode':
            output=model(input_ids=prefix,use_cache=True,logits_to_keep=1)
            cache=output.past_key_values
            del output

        def one():
            return model(input_ids=prefix if args.mode=='prefill' else token,
                         past_key_values=cache,use_cache=True,logits_to_keep=1)

        for _ in range(args.warmup):
            if cache is not None: rollback(cache,2048)
            one()
        if cache is not None: rollback(cache,2048)
        torch.cuda.synchronize()
        if args.backend == 'offload':
            # Instrument CPU gather on the producer too; no extra synchronizations
            # inside the model. CPU spans overlap and must not be added together.
            from fla.layers.memory_offload import GroupTicket
            original_index, original_to = torch.index_select, torch.Tensor.to
            original_copy, original_acquire = torch.Tensor.copy_, GroupTicket.acquire
            records, transfers = [], []

            def timed(label, function, *a, **kw):
                begin = time.perf_counter()
                result = function(*a, **kw)
                records.append(dict(label=label, cpu_ms=(time.perf_counter()-begin)*1000,
                                    thread=threading.current_thread().name))
                return result

            def index(source, *a, **kw):
                if source.device.type == 'cpu':
                    return timed('cpu_table_gather', original_index, source, *a, **kw)
                return original_index(source, *a, **kw)

            def to(source, *a, **kw):
                if source.is_cuda and a and str(a[0]) == 'cpu':
                    return timed('id_d2h_staging', original_to, source, *a, **kw)
                return original_to(source, *a, **kw)

            def copy(destination, source, *a, **kw):
                if destination.is_cuda and source.device.type == 'cpu':
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    result = timed('h2d_submit', original_copy, destination, source, *a, **kw)
                    end.record()
                    transfers.append((begin, end, source.numel()*source.element_size()))
                    return result
                return original_copy(destination, source, *a, **kw)

            def acquire(ticket, offset):
                return timed('ticket_acquire', original_acquire, ticket, offset)

            try:
                torch.index_select, torch.Tensor.to = index, to
                torch.Tensor.copy_, GroupTicket.acquire = copy, acquire
                begin = time.perf_counter()
                output = one()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter()-begin)*1000
            finally:
                torch.index_select, torch.Tensor.to = original_index, original_to
                torch.Tensor.copy_, GroupTicket.acquire = original_copy, original_acquire
            transfer_rows = [dict(stream_ms=a.elapsed_time(b), bytes=n) for a,b,n in transfers]
            (args.output/'offload.json').write_text(json.dumps(dict(
                campaign_id='offload-gap-001', batch_size=args.batch_size, mode=args.mode,
                variant=args.variant, diagnostic_wall_ms=elapsed, cpu_spans=records,
                transfers=transfer_rows, caveat='Instrumented single call. CPU spans overlap; stream spans include submission gaps; not headline timings.'), indent=2)+'\n')
            (args.output/'metadata.json').write_text(json.dumps(dict(
                campaign_id='offload-gap-001', status='completed', backend='offload',
                source=source, environment=environment, env=env_fingerprint(),
                config=model.config.to_dict(), warmup=args.warmup,
                memory=memory_snapshot(model,'cuda',output.past_key_values)), indent=2,default=str)+'\n')
            return
        if args.backend == 'events':
            # Diagnostic stream intervals, NOT isolated kernel durations:
            # event instrumentation and CPU launch gaps affect these spans.
            original_cat = torch.cat
            copies = []
            def traced_cat(tensors, *cat_args, **cat_kwargs):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                result = original_cat(tensors, *cat_args, **cat_kwargs)
                end.record()
                copies.append((begin, end, result.numel()*result.element_size()))
                return result
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            try:
                torch.cat = traced_cat
                start.record()
                output = one()
                stop.record()
                torch.cuda.synchronize()
            finally:
                torch.cat = original_cat
            spans = [dict(stream_ms=a.elapsed_time(b), output_bytes=size) for a,b,size in copies]
            (args.output/'events.json').write_text(json.dumps(dict(
                total_stream_ms=start.elapsed_time(stop), cat_spans=spans,
                cat_sum_stream_ms=sum(x['stream_ms'] for x in spans),
                cat_output_bytes=sum(x['output_bytes'] for x in spans),
                caveat='Instrumented stream spans include CPU launch gaps; not kernel time or headline latency.'), indent=2)+'\n')
            (args.output/'metadata.json').write_text(json.dumps(dict(
                status='completed', backend='events', campaign_id='offload-gap-001',batch_size=args.batch_size,mode=args.mode, variant=args.variant,
                source=source, environment=environment, env=env_fingerprint(),
                config=model.config.to_dict(), warmup=args.warmup, profiled_calls=1,
                memory=memory_snapshot(model,'cuda',output.past_key_values)),indent=2,default=str)+'\n')
            del output
            return
        if args.backend == 'nsys':
            # External Nsight Systems captures only this model call; warmup,
            # setup, and cache rollback remain outside its capture range.
            torch.cuda.profiler.start()
            output = one()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            memory = memory_snapshot(model, 'cuda', output.past_key_values)
            del output
            (args.output/'metadata.json').write_text(json.dumps(dict(
                status='capture_requested', backend='nsys', campaign_id='offload-gap-001',batch_size=args.batch_size,mode=args.mode,
                variant=args.variant, source=source, environment=environment,
                env=env_fingerprint(), config=model.config.to_dict(), memory=memory,
                warmup=args.warmup, profiled_calls=1,
                scope='diagnostic only; inspect external nsys report for GPU events'),
                indent=2, default=str)+'\n')
            return
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True,profile_memory=True,with_stack=False) as profiler:
            with torch.profiler.record_function('paper_diagnostic_model_call'):
                output=one()
                torch.cuda.synchronize()
        memory=memory_snapshot(model,'cuda',output.past_key_values)
        del output
        trace=args.output/'trace.json'
        profiler.export_chrome_trace(str(trace))
        with trace.open('rb') as src, gzip.open(args.output/'trace.json.gz','wb') as dst:
            shutil.copyfileobj(src,dst)
        trace.unlink()
        averages=profiler.key_averages(group_by_input_shape=True)
        rows=[dict(operator=event.key,count=event.count,input_shapes=event.input_shapes,
                   cpu_total_us=event.cpu_time_total,cpu_self_us=event.self_cpu_time_total,
                   device_total_us=event.device_time_total,device_self_us=event.self_device_time_total,
                   cpu_memory_bytes=event.cpu_memory_usage,device_memory_bytes=event.device_memory_usage)
              for event in averages]
        (args.output/'operators.json').write_text(json.dumps(rows,indent=2)+'\n')
        (args.output/'operators.txt').write_text(averages.table(sort_by='self_device_time_total',row_limit=60))
        (args.output/'metadata.json').write_text(json.dumps(dict(status='completed' if any(r['device_total_us'] for r in rows) else 'completed_cpu_only',campaign_id='offload-gap-001',batch_size=args.batch_size,mode=args.mode,
            variant=args.variant,source=source,environment=environment,env=env_fingerprint(),
            config=model.config.to_dict(),memory=memory,warmup=args.warmup,profiled_calls=1,
            scope='diagnostic only; profiler overhead; NOT offload_gap_v1 latency',
            limitation='CPU producer thread instrumentation may be incomplete'),indent=2,default=str)+'\n')
    finally:
        model.close_memory_offload()


if __name__=='__main__':
    with torch.inference_mode():
        main()
