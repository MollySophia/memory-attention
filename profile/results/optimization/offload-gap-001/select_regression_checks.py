"""Select bounded secondary regression checks under a fixed conservative rule."""
import argparse
import json
import math
from pathlib import Path

POLICY = 'nonprimary_single_pair_any_offload_or_resident_slowdown_v1'
PRIMARY = {(mode,batch,2048) for mode in ('prefill','decode') for batch in (1,8)}
SHAPES = {(batch,2048) for batch in (1,4,8,16)} | {(8,length) for length in (512,4096,8192)}


def select(summary,stage):
    assert stage in ('full_validation','generation_validation')
    keys = ({(mode,batch,length) for mode in ('prefill','decode') for batch,length in SHAPES} - PRIMARY
            if stage=='full_validation' else {('generation',batch,2048) for batch in (1,8)})
    rows = {}
    for point in summary['points']:
        key=(point['mode'],point['batch'],point['length'],point['implementation'],point['variant'])
        assert key not in rows
        rows[key]=point
    reviewed=[];selected=[]
    for mode,batch,length in sorted(keys):
        point={}
        for impl in ('baseline','candidate'):
            for variant in ('ma_offload','ma_gpu'):
                row=rows[(mode,batch,length,impl,variant)]
                assert row['status']=='completed', 'Use original completed matrix points, not repeated or failed results'
                assert row['independent_processes']==1, 'Never select repeated results for another extension'
                assert math.isfinite(row['latency_ms']) and row['latency_ms']>0
                point[(impl,variant)]=row['latency_ms']
        off=point[('baseline','ma_offload')]-point[('candidate','ma_offload')]
        gpu=point[('baseline','ma_gpu')]-point[('candidate','ma_gpu')]
        adverse=off<0 or gpu<0
        reviewed.append(dict(mode=mode,batch=batch,length=length,offload_reduction_ms=off,
                             gpu_reduction_ms=gpu,gap_reduction_ms=off-gpu,requires_bounded_check=adverse))
        if adverse:selected.append(f'{mode}:{batch}:{length}')
    return dict(campaign_id='offload-gap-001',stage=stage,policy=POLICY,reviewed=reviewed,selected=selected,
                extra_pairs_per_placement=2,original_pair_retained=True,
                note='Conservative follow-up trigger, not proof of regression. Exactly three total independent paired blocks; no further extension until favorable. Primary confirmation is not extended.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary',type=Path)
    p.add_argument('--stage',choices=('full_validation','generation_validation'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    assert not args.output.exists(), 'Selection is predeclared once per original stage'
    result=select(json.loads(args.summary.read_text()),args.stage)
    result['source_summary']=str(args.summary)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['selected']))
