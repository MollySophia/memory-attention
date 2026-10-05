"""Isolated mapped gather launch diagnostic; run only with no live GPU benchmark.

Compares ordinary JIT dispatch with its exact compiled kernel's runner. This
uses aligned ID buffers only; it does not establish safe reuse for offset IDs.
No whole-model performance or retention claim follows from these timings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    # Avoid accidental diagnostic/benchmark overlap when invoked later.
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if path.parent.name == str(os.getpid()):
            continue
        try:
            command = path.read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if any(Path(part.decode(errors='replace')).name in ('bench_fla.py', 'full_correctness.py') for part in command):
            raise RuntimeError(f'GPU benchmark/correctness child still live: {path.parent.name}')
    sys.path.insert(0, str(source))
    import torch
    import triton
    from fla.layers import memory_mapped_bulk as module
    assert Path(module.__file__).resolve() == source / 'fla/layers/memory_mapped_bulk.py'
    torch.manual_seed(1234)
    torch.set_num_threads(16)
    records = []
    with torch.inference_mode():
        weights = torch.randn(32000, 24, 2048, dtype=torch.bfloat16).pin_memory()
        width = 24 * 2048
        stream = torch.cuda.Stream()
        for batch in (1, 8):
            cpu_ids = [torch.randint(0, 32000, (batch,)) for _ in range(32)]
            ids = [v.to('cuda') for v in cpu_ids]
            assert all(v.data_ptr() % 16 == 0 for v in ids)
            output = torch.empty((batch, 24, 2048), dtype=weights.dtype, device='cuda')
            stream.wait_stream(torch.cuda.current_stream())
            grid = (batch, triton.cdiv(width, 512), 1)
            with torch.cuda.stream(stream):
                compiled = module._mapped_bulk_gather[grid](weights, ids[0], output, width, 32000, 512, num_warps=4)
                runner = compiled[grid]
                def launch(method, index):
                    if method == 'jit':
                        module._mapped_bulk_gather[grid](weights, ids[index], output, width, 32000, 512, num_warps=4)
                    else:
                        runner(weights, ids[index], output, width, 32000, 512, stream=stream.cuda_stream)
                # Fresh values and IDs must remain visible to the exact runner.
                for method in ('jit', 'compiled'):
                    for index in range(4):
                        stream.synchronize()
                        weights[int(cpu_ids[index][0])].add_(.125)
                        expected = weights.index_select(0, cpu_ids[index])
                        launch(method, index)
                        stream.synchronize()
                        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                for block in range(5):
                    for method in (('jit', 'compiled') if block % 2 == 0 else ('compiled', 'jit')):
                        for i in range(10):
                            launch(method, i)
                        stream.synchronize()
                        start = time.perf_counter_ns()
                        for i in range(128):
                            launch(method, i % len(ids))
                        submitted = time.perf_counter_ns()
                        stream.synchronize()
                        finished = time.perf_counter_ns()
                        records.append(dict(batch=batch, block=block+1, method=method, launches=128,
                                            submit_us_per_call=(submitted-start)/128000,
                                            drained_us_per_call=(finished-start)/128000))
            stream.synchronize()
        result = dict(status='diagnostic_only', source=str(source),
                      source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
                      mapped_module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                      torch=torch.__version__, triton=triton.__version__, rows=records,
                      exactness='Fresh IDs and host-row mutations, both methods and batches, bit-exact',
                      limitations=['No model/consumer overlap; isolated kernel launch only.',
                                   'All IDs are aligned; offset-ID specialization remains unvalidated.',
                                   'Compiled runner is the same kernel returned by ordinary JIT launch.',
                                   'No lookup result cache and no retained implementation change.'])
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        for batch in (1,8):
            for method in ('jit','compiled'):
                subset=[r for r in records if r['batch']==batch and r['method']==method]
                print(batch,method,{k:statistics.mean(r[k] for r in subset) for k in ('submit_us_per_call','drained_us_per_call')})


if __name__ == '__main__':
    main()
