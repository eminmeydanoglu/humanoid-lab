"""Original Dex3 windows with manifest-owned row and shared-video offsets."""
import json
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
INDEX = Path('/home/aksoy-lab/code/flux-training/flux-action/outputs/dex3/index')
DATASET = ROOT / 'data/datasets/first_tur_ham/unitree-g1-dex3'


class Dex3Dataset:
    def __init__(self, index=INDEX, root=DATASET):
        self.index, self.root = Path(index), Path(root)
        self.manifest = json.loads((self.index / 'manifest.json').read_text())
        m = self.manifest
        if m['state_dim'] != 28 or m['action_dim'] != 28 or m['state_names'] != m['action_names']:
            raise ValueError('Dataset must use matching 28D G1 measured-state/action order')
        self.joint_names = m['state_names']
        self.rows = np.load(self.index / 'rows.f32.npy', mmap_mode='r')
        if self.rows.shape != (m['total_frames'], 56):
            raise ValueError('Dataset row shape disagrees with manifest')
        self.episodes = {str(e['episode_index']): e for e in m['episodes']}
        for e in self.episodes.values():
            if e['from_index'] < 0 or e['n_frames'] < 1 or e['from_index'] + e['n_frames'] > len(self.rows):
                raise ValueError('Episode row range leaves the dataset')
            if e['videos']['head']['first_frame'] < 0 or any(not 0 <= lo < hi <= e['n_frames'] for lo, hi in e['valid_ranges']):
                raise ValueError('Invalid episode video offset or valid range')
        self.fps = float(m['fps'])

    def episode(self, episode_id):
        try:
            return self.episodes[str(episode_id)]
        except KeyError as exc:
            raise ValueError('Episode bulunamadı.') from exc

    def fingerprint(self, episode_id):
        episode = self.episode(episode_id)
        paths = [self.index / 'manifest.json', self.index / 'rows.f32.npy', self.root / episode['videos']['head']['file']]
        return {str(path): {'bytes': path.stat().st_size, 'modified_ns': path.stat().st_mtime_ns} for path in paths}

    def describe(self, episode_id):
        e = self.episode(episode_id)
        task = e['episode_id'].split('/')[0]
        return {'id': str(e['episode_index']), 'label': f"{task} · episode {e['source_episode_index']} · {e['split']}",
                'dataset': task, 'source_episode': e['source_episode_index'], 'split': 'validation' if e['split'] == 'val' else e['split'],
                'prompt': e['caption'], 'fps': self.fps, 'length': e['n_frames'],
                'camera': self.manifest['cameras']['head'], 'valid_ranges': e['valid_ranges']}

    def catalog(self):
        entries = [self.describe(key) for key in self.episodes]
        return sorted(entries, key=lambda e: (0 if 'PickApple' in e['dataset'] else 1, e['dataset'], e['source_episode']))

    def validate_frame(self, episode_id, frame):
        e = self.episode(episode_id)
        if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < e['n_frames']:
            raise ValueError(f"Kare 0–{e['n_frames'] - 1} aralığında olmalıdır; seçim değiştirilmedi.")
        if not any(lo <= frame < hi for lo, hi in e['valid_ranges']):
            raise ValueError('Seçilen kare geçerli veri aralığının dışında; seçim değiştirilmedi.')
        return e

    def reference_length(self, episode_id, frame, steps):
        e = self.validate_frame(episode_id, frame)
        hi = next(hi for lo, hi in e['valid_ranges'] if lo <= frame < hi)
        return max(0, min(steps, hi - frame - 1, e['n_frames'] - frame - 1))

    def validate_window(self, episode_id, frame):
        if self.reference_length(episode_id, frame, 32) < 32:
            raise ValueError('Tek chunk için aynı episode ve geçerli aralık içinde 32 gelecek kare gerekir; seçim değiştirilmedi.')

    def read_frames(self, episode_id, start, count, *, resized=False):
        e = self.episode(episode_id)
        if start < 0 or count < 1 or start + count > e['n_frames']:
            raise ValueError('Video penceresi episode sınırını aşıyor.')
        video = e['videos']['head']
        path = (self.root / video['file']).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError('Video path leaves dataset root')
        first = video['first_frame'] + start
        images = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            # MP4 seek lands on an earlier keyframe; PTS identifies exact source frames.
            container.seek(int(first / self.fps / stream.time_base), stream=stream, backward=True)
            for f in container.decode(stream):
                if f.pts is None:
                    raise ValueError('Dataset video frame has no timestamp')
                index = round(float(f.pts * stream.time_base) * self.fps)
                if index < first:
                    continue
                if index != first + len(images):
                    raise ValueError('Dataset video timeline has missing or duplicate frames')
                if resized:
                    f = f.reformat(width=256, height=192, format='rgb24')
                images.append(f.to_ndarray(format='rgb24'))
                if len(images) == count:
                    break
        if len(images) != count:
            raise ValueError('Dataset video ended before requested episode window')
        return np.stack(images)

    def observation(self, episode_id, frame):
        e = self.validate_frame(episode_id, frame)
        row = int(e['from_index']) + frame
        state = np.asarray(self.rows[row, :28]).copy()
        if not np.isfinite(state).all():
            raise ValueError('Measured state contains nonfinite values')
        return {'image': self.read_frames(episode_id, frame, 1)[0], 'state': state,
                'row_index': row, 'video_frame': int(e['videos']['head']['first_frame']) + frame,
                'previous_action': np.asarray(self.rows[row - 1, 28:]).copy() if frame > 0 else None}

    def reference(self, episode_id, frame, steps=32):
        count = self.reference_length(episode_id, frame, steps)
        e = self.episode(episode_id)
        row = int(e['from_index']) + frame
        actions = np.asarray(self.rows[row:row + count, 28:]).copy()
        if not np.isfinite(actions).all():
            raise ValueError('Reference actions contain nonfinite values')
        frames = self.read_frames(episode_id, frame + 1, count, resized=True) if count else np.empty((0,192,256,3), dtype=np.uint8)
        return {'frames': frames, 'actions': actions, 'steps': count}
