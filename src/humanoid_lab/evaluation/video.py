"""Video preparation with an explicit height-first resolution contract."""
from pathlib import Path

import av
import numpy as np


def padded_frames(count: int) -> int:
    # Decoder neighborhood attention requires at least five latent frames.
    length = max(17, count)
    return length + (1 - length) % 4


def prepare(path: Path, height: int, width: int, start_seconds: float, max_frames: int):
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30)
        skip = round(start_seconds * fps)
        for index, frame in enumerate(container.decode(stream)):
            if index < skip:
                continue
            frames.append(frame.reformat(width=width, height=height, format="rgb24").to_ndarray())
            if max_frames and len(frames) >= max_frames:
                break
            if len(frames) > 3600:
                raise ValueError("En fazla 3600 kare destekleniyor; başlangıç/kare sınırını ayarlayın.")
    if not frames:
        raise ValueError("Seçilen zaman aralığında video karesi yok.")
    return np.stack(frames), fps


def write_video(path: Path, frames, fps: float):
    from fractions import Fraction

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(10000))
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
