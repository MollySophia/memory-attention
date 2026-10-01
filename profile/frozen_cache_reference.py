"""Load the committed A0000 cache update for independent correctness checks."""
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import subprocess
import types

BASELINE_SHA = 'd949640ebf2f13f56021bd08c5c9f10e65d571c3'


@lru_cache(None)
def baseline_module():
    source = subprocess.check_output(
        ['git', 'show', f'{BASELINE_SHA}:fla/models/utils.py'],
        cwd=Path(__file__).resolve().parents[1], text=True)
    module = types.ModuleType('frozen_a0000_cache')
    exec(compile(source, f'{BASELINE_SHA}/fla/models/utils.py', 'exec'), module.__dict__)
    return module


@contextmanager
def frozen_cache_updates():
    # Single-process correctness tooling only; never used by timing scripts.
    from fla.models.utils import FLALayer, LegacyFLACache
    frozen = baseline_module()
    original = FLALayer.update, LegacyFLACache.update
    FLALayer.update, LegacyFLACache.update = frozen.FLALayer.update, frozen.LegacyFLACache.update
    try:
        yield
    finally:
        FLALayer.update, LegacyFLACache.update = original
