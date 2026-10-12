"""Sampling effort only; paper_v1 model/timing scope is unchanged."""
STAGES = ('screening', 'confirmation', 'full_validation', 'generation_validation', 'legacy')


def sampling_plan(stage, mode, warmup=None, repeats=None, rounds=None):
    if stage not in STAGES:
        raise ValueError(f'unknown measurement stage: {stage}')
    if mode == 'generation' and stage in ('screening', 'confirmation'):
        raise ValueError('generation requires generation_validation, full_validation or legacy')
    if stage == 'generation_validation' and mode != 'generation':
        raise ValueError('generation_validation requires mode generation')
    if stage == 'legacy':
        defaults, family = (30, 30, 5), 'legacy'
    elif mode == 'generation':
        defaults, family = (2, 5, 3), 'generation'
    elif stage == 'screening':
        defaults, family = (3, 5, 1), 'screen'
    else:
        defaults, family = (10, 10, 3), 'formal'
    values = tuple(default if value is None else value
                   for value, default in zip((warmup, repeats, rounds), defaults))
    w, n, r = values
    if w < 0 or n <= 0 or r <= 0:
        raise ValueError('warmup must be nonnegative; repeats and rounds must be positive')
    plan_id = f'{family}_v1_w{w}_n{n}_r{r}'
    if values != defaults:
        plan_id += '_custom'
    return dict(stage=stage, measurement_plan_id=plan_id, warmup=w, repeats=n, rounds=r,
                sampling_unit='trajectory' if mode == 'generation' else 'model_call')
