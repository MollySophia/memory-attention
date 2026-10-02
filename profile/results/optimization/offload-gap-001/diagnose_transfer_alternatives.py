"""Fresh-ID one-layer transfer microdiagnostics, separate from model timing.

Compare ordinary CPU gather/H2D, direct mapped-host GPU gather, and per-call
ID deduplication plus GPU expansion. Not an optimization attempt or acceptance.
"""
import argparse,json,sys,time,statistics,hashlib
from pathlib import Path
import torch
import triton
import triton.language as tl

@triton.jit
def mapped_rows(table, ids, output, N:tl.constexpr, D:tl.constexpr, L:tl.constexpr, V:tl.constexpr, BLOCK:tl.constexpr):
    row=tl.program_id(0)
    col=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    token=tl.load(ids+row)
    value=tl.load(table+token*(L*D)+col,mask=(col<D)&(token>=0)&(token<V),other=0)
    tl.store(output+row*D+col,value,mask=col<D)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[1,8,16]);a=p.parse_args()
    sys.path.insert(0,str(a.source_root/'profile'))
    from benchmark_telemetry import environment_details,source_state
    from bench_fla import env_fingerprint
    payload=dict(command=sys.argv,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=1234,campaign_id='offload-gap-001',purpose=__doc__,environment=environment_details(),source=source_state(),env=env_fingerprint(),warmup=10,repeats=10,records=[],limitations=['One layer, no compute overlap or whole-model claim.','Fresh IDs per trial; all three methods receive identical IDs within a trial. CPU ID staging is outside this component diagnostic.','Mapped table is fully pinned;3000MiB host capacity. Output buffers are reused, but every call performs fresh lookup; no persistent GPU lookup cache.','Dedup includes unique/inverse creation, inverse H2D and GPU expansion.','Normal finite BF16 weights only; full model correctness gates remain required.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def save():a.output.write_text(json.dumps(payload,indent=2)+'\n')
    with torch.inference_mode():
        torch.manual_seed(1234)
        table=torch.empty((32000,24,2048),dtype=torch.bfloat16,pin_memory=True).normal_()
        for batch in a.batches:
            n=batch*2048
            host=torch.empty((n,2048),dtype=table.dtype,pin_memory=True)
            gpu=torch.empty((n,2048),dtype=table.dtype,device='cuda')
            expanded=torch.empty_like(gpu)
            samples={k:[] for k in ('gather_h2d','mapped_host','dedup_h2d_expand')};errors={};distinct=[]
            for trial in range(20):
                ids=torch.randint(0,32000,(n,),device='cpu');ids_gpu=ids.to('cuda')
                expected=table[:,0].index_select(0,ids)
                def baseline():
                    torch.index_select(table[:,0],0,ids,out=host)
                    gpu.copy_(host,non_blocking=True)
                    return gpu
                def mapped():
                    mapped_rows[(n,triton.cdiv(2048,512))](table,ids_gpu,gpu,n,2048,24,32000,512,num_warps=4)
                    return gpu
                def dedup():
                    unique,inverse=torch.unique(ids,sorted=True,return_inverse=True)
                    count=unique.numel()
                    torch.index_select(table[:,0],0,unique,out=host[:count])
                    gpu[:count].copy_(host[:count],non_blocking=True)
                    torch.index_select(gpu[:count],0,inverse.to('cuda'),out=expanded)
                    return expanded
                methods=[('gather_h2d',baseline),('mapped_host',mapped),('dedup_h2d_expand',dedup)]
                shift=trial%3;methods=methods[shift:]+methods[:shift]
                for name,fn in methods:
                    if name in errors:continue
                    try:
                        torch.cuda.synchronize();start=time.perf_counter_ns()
                        result=fn();torch.cuda.synchronize();elapsed=(time.perf_counter_ns()-start)/1e6
                        torch.testing.assert_close(result.cpu(),expected,rtol=0,atol=0)
                        if trial>=10:samples[name].append(elapsed)
                    except Exception as exc:
                        errors[name]=repr(exc)
                        if name!='mapped_host':raise
                if trial>=10:distinct.append(ids.unique().numel())
            row=dict(batch=batch,tokens=n,samples_ms=samples,errors=errors,mean_ms={k:statistics.mean(v) if v else None for k,v in samples.items()},distinct_counts=distinct,bit_exact_for_successful_methods=True)
            payload['records'].append(row);save();print(json.dumps({k:row[k] for k in ('batch','mean_ms','errors')}),flush=True)
        payload['status']='completed';save()

if __name__=='__main__':main()
