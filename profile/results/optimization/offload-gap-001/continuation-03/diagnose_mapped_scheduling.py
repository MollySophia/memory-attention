"""Isolated mapped launch scheduling, not whole-model gain or acceptance evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
for path in Path('/proc').glob('[0-9]*/cmdline'):
    if path.parent.name == str(os.getpid()):
        continue
    try:
        parts = path.read_bytes().split(b'\0')
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
    if any(Path(x.decode(errors='replace')).name in
           ('bench_fla.py', 'full_correctness.py', 'diagnose_generation_path.py') for x in parts):
        raise RuntimeError(f'GPU work still live: {path.parent.name}')
root = a.source.resolve()
sys.path.insert(0, str(root))
import torch
import triton
from fla.layers import memory_mapped_bulk as module
assert Path(module.__file__).resolve() == root / 'fla/layers/memory_mapped_bulk.py'
torch.set_num_threads(16)
torch.manual_seed(1234)
rows = []
with torch.inference_mode():
    weights = torch.randn(32000, 24, 2048, dtype=torch.bfloat16).pin_memory()
    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=device)
    width = 24 * 2048
    for batch in (1, 8):
        cpu_ids = [torch.randint(0, 32000, (batch,)) for _ in range(32)]
        ids = []
        for index, value in enumerate(cpu_ids):
            storage = torch.empty(batch + 1, dtype=torch.long, device=device)
            view = storage[index % 2:index % 2 + batch]
            view.copy_(value)
            ids.append(view)
        assert {x.data_ptr() % 16 for x in ids} == {0, 8}
        output = torch.empty((batch, 24, 2048), dtype=weights.dtype, device=device)
        grid = (batch, triton.cdiv(width, 512), 1)
        stream.wait_stream(torch.cuda.current_stream(device))
        runners = {}
        with torch.cuda.device(device), torch.cuda.stream(stream):
            for value in ids[:2]:
                compiled = module._mapped_bulk_gather[grid](weights, value, output, width, 32000, 512, num_warps=4)
                runners[value.data_ptr() % 16] = compiled[grid]
        stream.synchronize()
        def launch(method, index):
            value = ids[index]
            with torch.cuda.device(device):
                if method == 'jit_stream_context':
                    with torch.cuda.stream(stream):
                        module._mapped_bulk_gather[grid](weights, value, output, width, 32000, 512, num_warps=4)
                else:
                    runners[value.data_ptr() % 16](weights, value, output, width, 32000, 512,
                                                   stream=stream.cuda_stream)
        for method in ('jit_stream_context', 'compiled_explicit_stream'):
            for index in range(8):
                weights[int(cpu_ids[index][0])].add_(.125)
                expected = weights.index_select(0, cpu_ids[index])
                launch(method, index)
                stream.synchronize()
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        for block in range(5):
            methods = ('jit_stream_context', 'compiled_explicit_stream')
            if block % 2:
                methods = methods[::-1]
            for method in methods:
                for i in range(10):
                    launch(method, i)
                stream.synchronize()
                start = time.perf_counter_ns()
                for i in range(128):
                    launch(method, i % 32)
                submitted = time.perf_counter_ns()
                stream.synchronize()
                finished = time.perf_counter_ns()
                rows.append(dict(batch=batch, block=block+1, method=method, launches=128,
                                 submit_us_per_call=(submitted-start)/128000,
                                 drained_us_per_call=(finished-start)/128000))
result = dict(status='diagnostic_only', source=str(root),
              source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              mapped_module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
              torch=torch.__version__, triton=triton.__version__, rows=rows,
              exactness='Fresh IDs and host mutations bit-exact for both methods, batches1/8, aligned and8-byte-offset int64 IDs.',
              limitations=['No consumer overlap or model execution; not a whole-model gain.',
                           'Device context preserved; no event dependencies removed.',
                           'The compiled runner must be keyed by ID pointer alignment; dtype/device/shapes/table/output fixed here.',
                           'No output cache; all kernels read fresh table rows.'])
a.output.write_text(json.dumps(result,indent=2)+'\n')
for batch in (1,8):
    for method in ('jit_stream_context','compiled_explicit_stream'):
        subset=[r for r in rows if r['batch']==batch and r['method']==method]
        print(batch,method,{k:statistics.mean(r[k] for r in subset)
                            for k in ('submit_us_per_call','drained_us_per_call')})
