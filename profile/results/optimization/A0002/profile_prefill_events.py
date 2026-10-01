"""Primary prefill module-stream diagnostic, separate from headline timings.

CUDA-event spans include CPU launch gaps and instrumentation. Linear spans are
nested in attention/MLP spans and must never be added to the coarse breakdown.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'profile'))
import torch
from bench_fla import build
from benchmark_telemetry import source_state, environment_details, memory_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=('ma_offload','ma_gpu'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--completed-manifest',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads(args.completed_manifest.read_text())
    assert manifest['status']=='complete' and all(j['status']=='complete' for j in manifest['jobs'])
    pid=manifest.get('controller_pid');proc=Path(f'/proc/{pid}/cmdline')
    assert not proc.exists() or b'run_confirmation.py' not in proc.read_bytes(), 'Timing controller still live'
    args.output.mkdir(parents=True,exist_ok=False)
    config=SimpleNamespace(seed=1234,hidden_size=2048,num_layers=24,num_heads=32,num_kv_heads=32,
        vocab_size=32000,hidden_ratio=4,intermediate_size=5632,group_size=1,prefetch_depth=4,policy='auto',device='cuda:0')
    model=build(config);handles=[]
    try:
        if args.variant=='ma_offload':
            model.enable_memory_offload();model.set_offload_offloader(8,2048);model.set_offload_offloader(8,1)
        else:model.fold_memory_table_on_gpu()
        torch.manual_seed(1235);ids=torch.randint(0,32000,(8,2048),device='cuda')
        def one():return model(input_ids=ids,use_cache=True,logits_to_keep=1)
        for _ in range(10):one()
        torch.cuda.synchronize()
        modules=[('embeddings','coarse',model.model.embeddings),('final_norm','coarse',model.model.norm),('lm_head','coarse',model.lm_head)]
        for i,layer in enumerate(model.model.layers):
            modules.extend([(f'layer{i}.attention','coarse',layer.attn),(f'layer{i}.mlp','coarse',layer.mlp)])
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Linear) and module is not model.lm_head:
                modules.append((name,'nested_linear',module))
        spans=[]
        for name,level,module in modules:
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();end.record()  # Allocate event resources outside diagnostic call.
            entry=dict(name=name,level=level,begin=begin,end=end,calls=0)
            spans.append(entry)
            def before(_module,_inputs,entry=entry):
                entry['calls']+=1;entry['begin'].record()
            def after(_module,_inputs,_output,entry=entry):entry['end'].record()
            handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
        start,stop=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();stop.record();torch.cuda.synchronize()
        start.record();out=one();stop.record();torch.cuda.synchronize()
        rows=[];groups=defaultdict(float)
        for entry in spans:
            assert entry['calls'] in (0,1),entry['name']
            ms=entry['begin'].elapsed_time(entry['end']) if entry['calls'] else None
            rows.append(dict(name=entry['name'],level=entry['level'],calls=entry['calls'],stream_ms=ms))
            if ms is not None:
                category='nested_linear' if entry['level']=='nested_linear' else 'attention' if entry['name'].endswith('.attention') else 'mlp' if entry['name'].endswith('.mlp') else entry['name']
                groups[category]+=ms
        total=start.elapsed_time(stop)
        payload=dict(status='completed',mode='prefill',variant=args.variant,batch=8,length=2048,
            warmup=10,diagnostic_calls=1,total_stream_ms=total,group_stream_ms=dict(groups),spans=rows,
            caveat='Instrumented stream intervals, not isolated kernel durations or headline wall time. Nested linear spans overlap coarse attention/MLP spans; do not sum across levels. Unhooked norms, residual ops, transfer waits and launch gaps contribute to the remaining total.',
            source=source_state(),environment=environment_details(),config=model.config.to_dict(),
            memory=memory_snapshot(model,'cuda',out.past_key_values))
        (args.output/'events.json').write_text(json.dumps(payload,indent=2,default=str)+'\n')
        print(json.dumps(dict(total_stream_ms=total,group_stream_ms=dict(groups)),indent=2))
    finally:
        for handle in handles:handle.remove()
        model.close_memory_offload()


if __name__=='__main__':
    with torch.inference_mode():main()
