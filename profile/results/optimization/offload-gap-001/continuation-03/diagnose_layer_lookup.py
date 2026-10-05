"""CPU-only layer-reference lookup diagnostic; no whole-model speedup claim."""
import argparse
import json
import statistics
import timeit
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
layers=torch.nn.ModuleList([torch.nn.Identity() for _ in range(24)])
def indexed_modules():
    for index in range(24):
        layer=layers[index]
    return layer
def per_call_tuple():
    references=tuple(layers)
    for index in range(24):
        layer=references[index]
    return layer
assert indexed_modules() is per_call_tuple()
replacement=torch.nn.Identity();layers[23]=replacement
assert indexed_modules() is replacement and per_call_tuple() is replacement
rows=[]
for block in range(7):
    for name,fn in ([('modulelist',indexed_modules),('tuple_per_call',per_call_tuple)] if block%2==0
                    else [('tuple_per_call',per_call_tuple),('modulelist',indexed_modules)]):
        elapsed=timeit.timeit(fn,number=10000)
        rows.append(dict(block=block+1,method=name,iterations=10000,us_per_24_lookups=elapsed*100))
result=dict(status='cpu_metadata_diagnostic_only',rows=rows,torch=torch.__version__,
            limitations=['No layers execute and no GPU work; only lookup dispatch is measured.',
                         'Tuple construction is included every call; layer replacement remains visible.',
                         'This is not end-to-end evidence or an accepted implementation change.'])
a.output.write_text(json.dumps(result,indent=2)+'\n')
for method in ('modulelist','tuple_per_call'):
    print(method,statistics.mean(r['us_per_24_lookups'] for r in rows if r['method']==method))
