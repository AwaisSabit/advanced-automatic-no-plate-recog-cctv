"""Load videos and read their metadata with OpenCV."""

import os

import cv2

from plateforge.models import VideoInfo


def open_video(path):
    """Open a video file and return the capture object.

    Raises FileNotFoundError if the file does not exist and
    ValueError if OpenCV cannot open it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    return cap


def get_video_info(path):
    """Read duration, fps, frame count, resolution and codec."""
    cap = open_video(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
        codec = "".join(c for c in codec if c.isprintable()).strip() or "unknown"
        duration = frame_count / fps if fps else 0.0
        return VideoInfo(
            path=path,
            duration_seconds=round(duration, 2),
            fps=round(float(fps), 2),
            frame_count=frame_count,
            width=width,
            height=height,
            codec=codec,
        )
    finally:
        cap.release()


def iter_frames(path):
    """Yield (frame_index, timestamp, frame) for every frame."""
    cap = open_video(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = index / fps if fps else 0.0
            yield index, round(timestamp, 3), frame
            index += 1
    finally:
        cap.release()
