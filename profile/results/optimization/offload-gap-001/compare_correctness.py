"""Compare independent frozen-resident/candidate-offload full-model fingerprints."""
import argparse
import json
from pathlib import Path


def compare(reference, candidate):
    for field in ('seed','batch_size','scope'):
        assert reference[field]==candidate[field]
    assert reference['status']==candidate['status']=='completed'
    assert reference['variant']=='ma_gpu' and candidate['variant']=='ma_offload'
    assert len(reference['checkpoints'])==len(candidate['checkpoints'])==129
    mismatches=[]
    def walk(a,b,path):
        if isinstance(a,dict):
            if set(a)!=set(b):mismatches.append(path+': keys differ');return
            for k in a:walk(a[k],b[k],path+'.'+k)
        elif isinstance(a,list):
            if len(a)!=len(b):mismatches.append(path+': length differs');return
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,f'{path}[{i}]')
        elif a!=b or path.endswith('.finite') and not a:
            mismatches.append(path)
    walk(reference['checkpoints'],candidate['checkpoints'],'checkpoints')
    exact=not mismatches
    return dict(campaign_id='offload-gap-001',seed=reference['seed'],batch_size=reference['batch_size'],
                status='completed' if exact else 'correctness_failed',bit_exact=exact,
                max_absolute_error=0 if exact else None,relative_error=0 if exact else None,
                relative_error_near_zero_rule='denominator max(abs(reference),1e-8); zero derived only after exact byte fingerprints match',
                normalized_rms_error=0 if exact else None,argmax_agreement=1 if exact else None,
                evidence='SHA256 fingerprints of contiguous BF16 tensor bytes, shape/dtype checks, finiteness and explicit logit argmax',
                scope=reference['scope'],mismatches=mismatches,
                reference_source=reference['source'],candidate_source=candidate['source'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=compare(json.loads(args.reference.read_text()),json.loads(args.candidate.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],result['batch_size'],result['seed'],result['mismatches'][:10])
    return 0 if result['bit_exact'] else 1

if __name__=='__main__':raise SystemExit(main())
