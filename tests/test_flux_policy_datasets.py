import json

import av
import numpy as np
import pytest

from humanoid_lab.evaluation.datasets import Dex3Dataset
from humanoid_lab.evaluation.video import write_video


@pytest.fixture
def dex3(tmp_path):
    index, root = tmp_path / 'index', tmp_path / 'dataset'
    index.mkdir(); root.mkdir()
    frames = np.stack([np.full((64,96,3), i * 2, dtype=np.uint8) for i in range(80)])
    write_video(root / 'shared.mp4', frames, 30)
    rows = np.arange(80 * 56, dtype=np.float32).reshape(80,56)
    np.save(index / 'rows.f32.npy', rows)
    names = [f'joint-{i}' for i in range(28)]
    episodes = [{'episode_index': n, 'episode_id': f'Apple/episode_{n}', 'source_episode_index': n,
                 'split': 'train' if n == 0 else 'val', 'n_frames': 40, 'caption': 'apple',
                 'valid_ranges': [[1,40]], 'from_index': n * 40,
                 'videos': {'head': {'file':'shared.mp4','first_frame':n * 40}}} for n in (0,1)]
    manifest = {'state_dim':28,'action_dim':28,'state_names':names,'action_names':names,
                'total_frames':80,'fps':30,'cameras':{'head':'observation.images.cam_left_high'},'episodes':episodes}
    (index / 'manifest.json').write_text(json.dumps(manifest))
    return Dex3Dataset(index, root), rows, frames


def test_row_video_offsets_and_timeline(dex3):
    dataset, rows, _ = dex3
    observation = dataset.observation('1', 5)
    np.testing.assert_array_equal(observation['state'], rows[45,:28])
    np.testing.assert_array_equal(observation['previous_action'], rows[44,28:])
    assert observation['row_index'] == observation['video_frame'] == 45
    reference = dataset.reference('1',5)
    np.testing.assert_array_equal(reference['actions'], rows[45:77,28:])
    assert reference['frames'].shape == (32,192,256,3)
    direct = dataset.read_frames('1',6,32,resized=True)
    np.testing.assert_array_equal(reference['frames'], direct)
    with av.open(str(dataset.root / 'shared.mp4')) as movie:
        sequential = [frame.to_ndarray(format='rgb24') for frame in movie.decode(video=0)]
    np.testing.assert_array_equal(observation['image'], sequential[45])
    assert dataset.describe('1')['split'] == 'validation'


def test_episode_boundaries_and_partial_dream_reference(dex3):
    dataset, rows, _ = dex3
    dataset.validate_window('1',7)
    with pytest.raises(ValueError, match='32'):
        dataset.validate_window('1',8)
    with pytest.raises(ValueError, match='geçerli'):
        dataset.observation('1',0)
    with pytest.raises(ValueError, match='aralığında'):
        dataset.observation('1',40)
    ref = dataset.reference('0',35,320)
    assert ref['steps'] == 4
    np.testing.assert_array_equal(ref['actions'],rows[35:39,28:])
    with pytest.raises(ValueError, match='sınır'):
        dataset.read_frames('0',35,6)
    assert dataset.reference('1',39,320)['frames'].shape == (0,192,256,3)


def test_manifest_joint_order_guard(dex3):
    dataset, _, _ = dex3
    path = dataset.index / 'manifest.json'
    m = json.loads(path.read_text()); m['action_names'].reverse()
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='order'):
        Dex3Dataset(dataset.index, dataset.root)
