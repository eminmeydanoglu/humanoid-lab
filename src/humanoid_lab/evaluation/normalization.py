"""CPU inspection using the checkpoint-owned history quantiles."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=4)
def saved_quantiles(checkpoint: str):
    from safetensors.numpy import load_file
    directory = Path(checkpoint)
    pipeline = json.loads((directory / 'policy_preprocessor.json').read_text())
    stage = next(step for step in pipeline['steps'] if step['registry_name'] == 'flux3_observation_history_normalizer')
    if stage['config']['action_representation'] != 'absolute':
        raise ValueError('Evaluation requires absolute checkpoint actions')
    return load_file(directory / stage['state_file']), float(stage['config']['normalization_clip'])


def normalize(values, stream, checkpoint):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1:] != (28,) or not np.isfinite(values).all():
        raise ValueError('Expected finite 28D values')
    quantiles, limit = saved_quantiles(str(checkpoint))
    low, high = quantiles[f'{stream}.q01'], quantiles[f'{stream}.q99']
    span = np.where(high - low > 1e-6, high - low, np.ones_like(low))
    scaled = 2 * (values - low) / span - 1
    clipped = np.clip(scaled, -limit, limit)
    report = {'clip_limit': limit, 'clipped_values': int(np.count_nonzero(scaled != clipped)),
              'outside_quantiles': int(np.count_nonzero(np.abs(scaled) > 1)),
              'maximum_absolute_unclipped': float(np.abs(scaled).max()) if scaled.size else 0.0}
    return clipped, report
