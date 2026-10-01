"""Independent accepted-A0002 forward for correctness only, never timing."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import subprocess
import types

INCUMBENT_SHA='ec6384d2e9081ecbb2926fe8a6193bc48ea650a3'

@lru_cache(None)
def reference_module():
    source=subprocess.check_output(['git','show',f'{INCUMBENT_SHA}:fla/layers/memory_attn.py'],cwd=Path(__file__).resolve().parents[4],text=True)
    module=types.ModuleType('accepted_a0002_attention')
    exec(compile(source,f'{INCUMBENT_SHA}/fla/layers/memory_attn.py','exec'),module.__dict__)
    return module

@contextmanager
def frozen_attention():
    from fla.layers.memory_attn import MemoryAttention
    original=MemoryAttention.forward
    MemoryAttention.forward=reference_module().MemoryAttention.forward
    try:yield
    finally:MemoryAttention.forward=original
