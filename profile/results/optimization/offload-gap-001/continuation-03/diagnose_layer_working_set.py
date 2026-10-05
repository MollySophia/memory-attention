"""Compare table layouts across an entire layer walk, outside model timings.

The historical layout microbenchmark repeatedly gathered one fixed layer group.
This diagnostic visits every layer per call, without an artificial cache flush.
No model optimization or end-to-end gain claim follows from these timings.
Run only after GPU performance/profile/correctness jobs have stopped.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def require_idle():
    active=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:parts=path.read_bytes().split(b'\0')
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        # Match script arguments, not shell source text containing their names.
        if any(Path(arg.decode(errors='replace')).name in ('bench_fla.py','full_correctness.py','test_memory_offload.py') for arg in parts if arg):
            active.append(int(path.parent.name))
    assert not active, f'Wait for live benchmark/correctness processes: {active}'


def walk(torch,table,ids,out):
    for layer in range(table.shape[1]):
        torch.index_select(table[:,layer:layer+1],0,ids,out=out)


def verify(torch,original,layered,ids):
    # Check every layer, not just the final buffer left by a complete walk.
    for layer in range(original.shape[1]):
        expected=original[:,layer:layer+1].index_select(0,ids)
        actual=layered[:,layer:layer+1].index_select(0,ids)
        assert torch.equal(actual,expected)


def measure(torch,original,layered,ids,out,warmup,repeats):
    assert warmup>=0 and repeats>=1
    samples={'vocab_major':[],'layer_major':[]}
    for iteration in range(warmup+repeats):
        order=[('vocab_major',original),('layer_major',layered)]
        if iteration%2:order.reverse()
        for label,table in order:
            start=time.perf_counter_ns();walk(torch,table,ids,out)
            elapsed=(time.perf_counter_ns()-start)/1e6
            if iteration>=warmup:samples[label].append(elapsed)
    return samples


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,nargs='+',default=[2048,4096,8192,16384])
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--repeats',type=int,default=30)
    args=p.parse_args()
    assert args.warmup>=0 and args.repeats>=1 and all(n>0 for n in args.tokens)
    assert not args.output.exists(), 'Keep earlier diagnostics under distinct output names'
    require_idle()
    sys.path.insert(0,str(args.source_root.resolve()/'profile'))
    import torch
    from benchmark_telemetry import environment_details,source_state
    torch.set_num_threads(16);torch.manual_seed(1234)
    rows=[]
    with torch.inference_mode():
        original=torch.randn(32000,24,2048,dtype=torch.bfloat16,pin_memory=True)
        storage=torch.empty(24,32000,2048,dtype=original.dtype,pin_memory=True)
        layered=storage.permute(1,0,2)
        layered.copy_(original)
        for tokens in args.tokens:
            require_idle()
            ids=torch.randint(0,32000,(tokens,))
            out=torch.empty(tokens,1,2048,dtype=original.dtype,pin_memory=True)
            verify(torch,original,layered,ids)
            samples=measure(torch,original,layered,ids,out,args.warmup,args.repeats)
            rows.append(dict(tokens=tokens,layers=24,output_bytes_per_walk=tokens*24*2048*2,
                             unique_token_rows=int(ids.unique().numel()),samples_ms=samples,
                             mean_ms={k:statistics.mean(v) for k,v in samples.items()},
                             median_ms={k:statistics.median(v) for k,v in samples.items()},exact_all_layers=True))
            print(tokens,rows[-1]['mean_ms'],flush=True)
        result=dict(status='diagnostic_only',scope='CPU index_select through all24 layers per call',
                    command=sys.argv,source=source_state(),environment=environment_details(),
                    helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    weights_shape=list(original.shape),table_pinned=original.is_pinned() and layered.is_pinned(),
                    diagnostic_table_bytes=2*original.numel()*original.element_size(),original_stride=list(original.stride()),layer_major_stride=list(layered.stride()),
                    warmup=args.warmup,repeats=args.repeats,torch_threads=torch.get_num_threads(),rows=rows,
                    limitations=['Same-process alternating synthetic-table diagnostic, not independent full-model performance evidence.',
                                 'No GPU compute/H2D overlap; packing time excluded and two complete layouts coexist.',
                                 'Mapped bulk currently requires contiguous vocab-major storage; any layout candidate must preserve and validate small-input paths.',
                                 'Table-layout changes are not implemented or registered by this diagnostic.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
