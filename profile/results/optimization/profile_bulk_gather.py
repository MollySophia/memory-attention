"""Accepted-model diagnostic: CPU gather cost and thread-pool sensitivity.

Two fixed30-call blocks, default threads then one thread, in one diagnostic
process. Not independent process evidence, not a candidate speedup claim.
"""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.checkout.resolve()
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:argv=proc.read_bytes().split(b'\0')
        except OSError:continue
        if any(Path(x.decode(errors='replace')).name in ('run_confirmation.py','run_screening.py','run_full_validation.py','run_regression_followup.py') for x in argv):raise RuntimeError('Live timing controller')
    sys.path[:0]=[str(root),str(root/'profile')]
    import torch
    from bench_fla import build,rollback
    from benchmark_telemetry import source_state,environment_details
    import fla.models.utils as utils
    assert Path(utils.__file__).resolve()==root/'fla/models/utils.py'
    args.output.mkdir(parents=True,exist_ok=False)
    config=SimpleNamespace(seed=1234,hidden_size=2048,num_layers=24,num_heads=32,num_kv_heads=32,vocab_size=32000,hidden_ratio=4,intermediate_size=5632,group_size=1,prefetch_depth=4,policy='auto',device='cuda:0')
    original=torch.index_select;threads=torch.get_num_threads();rows=[];gathers=[]
    getcpu=ctypes.CDLL(None).sched_getcpu
    def measured(*args,**kwargs):
        if args[0].device.type!='cpu':return original(*args,**kwargs)
        cpu=getcpu();start=time.perf_counter();out=original(*args,**kwargs)
        gathers.append(dict(cpu=cpu,cpu_after=getcpu(),wall_ms=(time.perf_counter()-start)*1000))
        return out
    with torch.inference_mode():
        model=build(config)
        try:
            model.enable_memory_offload();torch.manual_seed(1235)
            prefix=torch.randint(0,32000,(8,2048),device='cuda');tokens=torch.randint(0,32000,(40,8,1),device='cuda')
            cache=model(input_ids=prefix,use_cache=True,logits_to_keep=1).past_key_values
            for n in (threads,1):
                torch.set_num_threads(n)
                for ids in tokens[:10]:model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1);rollback(cache,2048)
                torch.cuda.synchronize();torch.index_select=measured
                for step,ids in enumerate(tokens[10:]):
                    before=len(gathers);start=time.perf_counter()
                    model(input_ids=ids,past_key_values=cache,use_cache=True,logits_to_keep=1);torch.cuda.synchronize()
                    elapsed=(time.perf_counter()-start)*1000
                    assert len(gathers)==before+1
                    rows.append(dict(threads=n,step=step,model_wall_ms=elapsed,gather=gathers[-1]));rollback(cache,2048)
                torch.index_select=original
            data=dict(source=source_state(),environment=environment_details(),scope='Accepted A0002 batch8/context2048 offload decode; instrumented one-process blocks; distinct predetermined tokens,no output caching.',plan='Default thread count then1;10warmups+30calls each; fixed diagnostic,no headline performance comparison.',rows=rows,original_threads=threads)
            (args.output/'diagnostic.json').write_text(json.dumps(data,indent=2)+'\n')
            import statistics
            for n in (threads,1):
                r=[x for x in rows if x['threads']==n];print(n,'model median',statistics.median(x['model_wall_ms'] for x in r),'gather median',statistics.median(x['gather']['wall_ms'] for x in r),'gather max',max(x['gather']['wall_ms'] for x in r),'CPUs',sorted({x['gather']['cpu'] for x in r}))
        finally:torch.index_select=original;torch.set_num_threads(threads);model.close_memory_offload()


if __name__=='__main__':main()
