"""Measure instrumented FlashAttention spans on a specified accepted checkout.

Run separately from benchmarks. CPU wall spans include Python/launch overhead;
CUDA-event spans include launch gaps, not pure kernel durations. Never promote
these values to headline performance claims.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.checkout.resolve()
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:argv=proc.read_bytes().split(b'\0')
        except OSError:continue
        if any(Path(x.decode(errors='replace')).name in ('run_confirmation.py','run_full_validation.py','run_screening.py','run_regression_followup.py') for x in argv):
            raise RuntimeError(f'Timing still live: {proc.parent.name}')
    sys.path[:0]=[str(root),str(root/'profile')]
    import torch
    import fla.models.utils as cache_module
    assert Path(cache_module.__file__).resolve()==root/'fla/models/utils.py'
    from bench_fla import build,rollback
    from benchmark_telemetry import source_state,environment_details
    args.output.mkdir(parents=True,exist_ok=False)
    config=SimpleNamespace(seed=1234,hidden_size=2048,num_layers=24,num_heads=32,num_kv_heads=32,
        vocab_size=32000,hidden_ratio=4,intermediate_size=5632,group_size=1,prefetch_depth=4,policy='auto',device='cuda:0')
    import fla.layers.memory_attn as attention_module
    original=attention_module.flash_attn_func
    with torch.inference_mode():
        model=build(config)
        try:
            model.enable_memory_offload();torch.manual_seed(1235)
            prefix=torch.randint(0,32000,(8,2048),device='cuda');token=torch.randint(0,32000,(8,1),device='cuda')
            out=model(input_ids=prefix,use_cache=True,logits_to_keep=1);cache=out.past_key_values
            for _ in range(10):
                model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1);rollback(cache,2048)
            torch.cuda.synchronize()
            spans=[]
            for _ in range(24):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();end.record();spans.append(dict(begin=begin,end=end))
            total_start,total_end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            total_start.record();total_end.record();torch.cuda.synchronize()
            cursor=0
            def instrumented(q,k,v,*args,**kwargs):
                nonlocal cursor
                entry=spans[cursor];cursor+=1
                entry['begin'].record();start=time.perf_counter()
                output=original(q,k,v,*args,**kwargs)
                entry['cpu_ms']=(time.perf_counter()-start)*1000;entry['end'].record()
                entry['incoming_bytes']=sum(x.numel()*x.element_size() for x in (q,k,v))
                entry['attention_calls']=1
                return output
            attention_module.flash_attn_func=instrumented
            total_start.record();start=time.perf_counter()
            out=model(input_ids=token,past_key_values=cache,use_cache=True,logits_to_keep=1)
            total_end.record();torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
            assert cursor==24
            rows=[dict(layer=i,cpu_ms=e['cpu_ms'],stream_ms=e['begin'].elapsed_time(e['end']),
                       incoming_bytes=e['incoming_bytes'],attention_calls=e['attention_calls']) for i,e in enumerate(spans)]
            data=dict(source=source_state(),environment=environment_details(),imported_cache=str(cache_module.__file__),
                      mode='decode',batch=8,context=2048,diagnostic_calls=1,warmup=10,
                      instrumented_total_wall_ms=wall,instrumented_total_stream_ms=total_start.elapsed_time(total_end),
                      attention_cpu_ms=sum(r['cpu_ms'] for r in rows),attention_stream_ms=sum(r['stream_ms'] for r in rows),
                      attention_calls=sum(r['attention_calls'] for r in rows),spans=rows,
                      limitation='Instrumented spans include launch gaps/overhead; not isolated kernels or headline timing.')
            (args.output/'events.json').write_text(json.dumps(data,indent=2)+'\n')
            print(json.dumps({k:data[k] for k in ('instrumented_total_wall_ms','attention_cpu_ms','attention_stream_ms','attention_calls')}))
        finally:
            attention_module.flash_attn_func=original;model.close_memory_offload()


if __name__=='__main__':main()
