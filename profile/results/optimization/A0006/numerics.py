"""Predeclared numerical checks for the changed attention reduction."""
import json
from pathlib import Path
import torch

CONTRACT=json.loads((Path(__file__).with_name('numerical-contract.json')).read_text())

def compare(actual,expected,name,exact=False):
    a=actual.float();b=expected.to(device=a.device,dtype=torch.float32)
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    assert finite,(name,'nonfinite')
    error=(a-b).abs();rms=b.square().mean().sqrt();floor=max(float(rms)*.1,1e-8)
    row=dict(name=name,shape=list(actual.shape),finite=finite,
             max_abs_error=float(error.max()),
             max_relative_error=float((error/b.abs().clamp_min(floor)).max()),
             normalized_rms=float(error.square().mean().sqrt())/max(float(rms),1e-8),
             exact=bool(torch.equal(a,b)),required_exact=exact)
    assert row['normalized_rms']<=CONTRACT['normalized_rms_limit'],row
    assert bool((error<=CONTRACT['absolute_tolerance']+CONTRACT['relative_tolerance']*b.abs()).all()),row
    if exact:assert row['exact'],row
    if name=='logits':
        row.update(argmax_matches=int((a.argmax(-1)==b.argmax(-1)).sum()),argmax_count=a.shape[0]*a.shape[1])
    return row
