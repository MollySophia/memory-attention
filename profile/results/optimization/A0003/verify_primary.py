"""Primary exact gate with both frozen cache and independent A0002 attention."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import torch
from reference import frozen_attention,INCUMBENT_SHA
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'profile'))
spec=importlib.util.spec_from_file_location('primary_gate',ROOT/'profile/verify_primary.py')
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
original=gate.frozen_cache_updates
@contextmanager
def reference():
    with original(),frozen_attention():yield

gate.frozen_cache_updates=reference
with torch.inference_mode():gate.main()
path=Path(sys.argv[sys.argv.index('--output')+1]);d=json.loads(path.read_text());d['reference_attention_sha']=INCUMBENT_SHA;path.write_text(json.dumps(d,indent=2)+'\n')
