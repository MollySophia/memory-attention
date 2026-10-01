"""Independent-source full-model exactness fingerprints, outside performance runs.

Run separately for frozen resident and candidate offload with identical seed
and batch. Checks last-token logits at prefix + all 128 growing decode steps;
all hidden states and full KV storage at prefix and steps 1, 2, 128. SHA256
fingerprints preserve bit equality without retaining multi-GiB tensor dumps.
Small-model regression tests additionally cover padded, gated, QK-normalized,
GQA, non-unit-affine and repeated-forward cases.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--variant',choices=('ma_gpu','ma_offload'),required=True)
    parser.add_argument('--batch-size',type=int,choices=(1,8),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.source_root.resolve()
    sys.path.insert(0,str(root));sys.path.insert(0,str(root/'profile'))
    import torch
    from bench_fla import build,env_fingerprint
    from benchmark_telemetry import source_state,environment_details,memory_snapshot
    config=SimpleNamespace(seed=args.seed,hidden_size=2048,num_layers=24,num_heads=32,
                           num_kv_heads=32,vocab_size=32000,hidden_ratio=4,intermediate_size=5632,
                           group_size=1,prefetch_depth=4,policy='auto',device='cuda:0')
    payload=dict(campaign_id='offload-gap-001',purpose='correctness only, not performance',
                 seed=args.seed,batch_size=args.batch_size,variant=args.variant,source=source_state(),
                 environment=environment_details(),env=env_fingerprint(),checkpoints=[],status='running',
                 scope='logits every step; all hidden and KV states at prefix and decode steps 1,2,128; unpadded full model')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(payload,indent=2)+'\n')
    def fingerprint(tensor):
        tensor=tensor.detach().contiguous()
        finite=bool(torch.isfinite(tensor).all())
        host=tensor.cpu()
        digest=hashlib.sha256(memoryview(host.view(torch.uint8).numpy())).hexdigest()
        return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=digest,finite=finite)
    model=None
    try:
        with torch.inference_mode():
            model=build(config)
            payload['model_config']=model.config.to_dict()
            if args.variant=='ma_offload':model.enable_memory_offload()
            else:model.fold_memory_table_on_gpu()
            torch.manual_seed(args.seed+1)
            prefix=torch.randint(0,32000,(args.batch_size,2048),device='cuda')
            tokens=torch.randint(0,32000,(128,args.batch_size,1),device='cuda').unbind()
            cache=None
            for step,token in enumerate((prefix,*tokens)):
                detailed=step in (0,1,2,128)
                output=model(input_ids=token,past_key_values=cache,use_cache=True,
                             logits_to_keep=1,output_hidden_states=detailed)
                cache=output.past_key_values
                assert list(output.logits.shape)==[args.batch_size,1,32000]
                checkpoint=dict(step=step,context_length=2048+step,logits=fingerprint(output.logits),
                                argmax=output.logits.argmax(-1).cpu().tolist())
                if detailed:
                    checkpoint['hidden_states']=[fingerprint(t) for t in output.hidden_states]
                    checkpoint['kv_states']=[[fingerprint(t) for t in state['attn_state']] for state in cache]
                    assert len(checkpoint['hidden_states'])==25 and len(checkpoint['kv_states'])==24
                payload['checkpoints'].append(checkpoint)
                del output
                if detailed:save()
            payload['memory']=memory_snapshot(model,'cuda',cache)
            payload['status']='completed';save()
    except Exception as exc:
        payload.update(status='correctness_failed',error=repr(exc));save();raise
    finally:
        if model is not None:model.close_memory_offload()

if __name__=='__main__':main()
