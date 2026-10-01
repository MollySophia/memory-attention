"""Diagnostic CPU/CUDA trace, separate from paper_v1 wall-time measurements.

Run only after the matrix controller exits. Profiler overhead invalidates these
latencies for speedup claims. The offload producer thread may not be fully
captured by torch.profiler; use CUDA copies and end-to-end timings as well.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch

from bench_fla import build, rollback, env_fingerprint
from benchmark_telemetry import environment_details, source_state, memory_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prefill','decode'), required=True)
    parser.add_argument('--variant', choices=('ma_offload','ma_gpu'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True, help='Completed baseline manifest; concurrent profiling is refused')
    args=parser.parse_args()
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
            model.set_offload_offloader(8,2048)
            model.set_offload_offloader(8,1)
        else:
            model.fold_memory_table_on_gpu()
        torch.manual_seed(1235)
        prefix=torch.randint(0,32000,(8,2048),device='cuda')
        token=torch.randint(0,32000,(8,1),device='cuda')
        cache=None
        if args.mode=='decode':
            output=model(input_ids=prefix,use_cache=True,logits_to_keep=1)
            cache=output.past_key_values
            del output

        def one():
            return model(input_ids=prefix if args.mode=='prefill' else token,
                         past_key_values=cache,use_cache=True,logits_to_keep=1)

        for _ in range(30):
            if cache is not None: rollback(cache,2048)
            one()
        if cache is not None: rollback(cache,2048)
        torch.cuda.synchronize()
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
        (args.output/'metadata.json').write_text(json.dumps(dict(status='completed',mode=args.mode,
            variant=args.variant,source=source,environment=environment,env=env_fingerprint(),
            config=model.config.to_dict(),memory=memory,warmup=30,profiled_calls=1,
            scope='diagnostic only; profiler overhead; NOT paper_v1 latency',
            limitation='CPU producer thread instrumentation may be incomplete'),indent=2,default=str)+'\n')
    finally:
        model.close_memory_offload()


if __name__=='__main__':
    with torch.inference_mode():
        main()
