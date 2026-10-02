"""Compare pinned allocation modes for fresh CPU gather plus H2D, separately from timing."""
import argparse
import ctypes
import hashlib
import json
import sys
import time
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
sys.path[:0]=[str(a.source_root),str(a.source_root/'profile')]
import torch
from benchmark_telemetry import source_state,environment_details
library=Path(torch.__file__).resolve().parents[1]/'nvidia/cu13/lib/libcudart.so.13'
runtime=ctypes.CDLL(str(library))
runtime.cudaHostAlloc.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t,ctypes.c_uint]
runtime.cudaHostAlloc.restype=ctypes.c_int
runtime.cudaFreeHost.argtypes=[ctypes.c_void_p]
runtime.cudaFreeHost.restype=ctypes.c_int

def allocate(tokens,flags):
    pointer=ctypes.c_void_p()
    size=tokens*2048*2
    status=runtime.cudaHostAlloc(ctypes.byref(pointer),size,flags)
    if status:
        raise RuntimeError(f'cudaHostAlloc({flags}) failed: {status}')
    buffer=(ctypes.c_ubyte*size).from_address(pointer.value)
    tensor=torch.frombuffer(buffer,dtype=torch.bfloat16).view(tokens,1,2048)
    return tensor,pointer,buffer

with torch.inference_mode():
    torch.manual_seed(1234)
    torch.cuda.init()
    table=torch.randn(32000,24,2048,dtype=torch.bfloat16)
    results=[]
    for tokens in (2048,16384,32768):
        allocated=[]
        host={}
        try:
            host['torch_pinned']=torch.empty(tokens,1,2048,dtype=table.dtype,pin_memory=True)
            for name,flags in [('cuda_default',0),('cuda_write_combined',4)]:
                tensor,pointer,buffer=allocate(tokens,flags)
                allocated.append((pointer,buffer))
                host[name]=tensor
                assert tensor.is_pinned(), 'CUDA host allocation must be recognized as pinned'
            gpu=torch.empty(tokens,1,2048,dtype=table.dtype,device='cuda')
            samples={name:[] for name in host}
            for trial in range(20):
                ids=torch.randint(0,32000,(tokens,))
                source=table[:,trial%24:trial%24+1]
                expected=source.index_select(0,ids)
                methods=list(host)
                methods=methods[trial%3:]+methods[:trial%3]
                for name in methods:
                    torch.cuda.synchronize()
                    start=time.perf_counter_ns()
                    torch.index_select(source,0,ids,out=host[name])
                    gpu.copy_(host[name],non_blocking=True)
                    torch.cuda.synchronize()
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    torch.testing.assert_close(gpu.cpu(),expected,rtol=0,atol=0)
                    if trial>=10:
                        samples[name].append(elapsed)
            results.append(dict(tokens=tokens,samples_ms=samples,
                mean_ms={k:sum(v)/len(v) for k,v in samples.items()}))
        finally:
            torch.cuda.synchronize()
            host.clear()
            # External allocations remain valid through every CPU/GPU use above.
            for pointer,buffer in allocated:
                status=runtime.cudaFreeHost(pointer)
                if status:
                    raise RuntimeError(f'cudaFreeHost failed: {status}')
    payload=dict(campaign_id='offload-gap-001',purpose=__doc__,source=source_state(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=sys.argv,
        environment=environment_details(),cuda_runtime=str(library),warmup=10,trials=10,
        seed=1234,exactness='All changing-ID outputs exact',results=results,
        limitations='One-layer component diagnostic; ordinary CPU source table. GPU synchronized around component; does not measure overlap with model compute. cuda_default controls native allocation separately from write-combined flags.')
    a.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='samples_ms'} for r in results]))
