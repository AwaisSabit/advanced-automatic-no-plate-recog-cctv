"""Extract frames from video by interval, range or timestamps."""

from plateforge.video import iter_frames


def extract_by_interval(path, every_seconds=None, every_nth=None):
    """Extract one frame every N seconds, or every Nth frame.

    Returns a list of (frame_index, timestamp, frame). Exactly one of
    every_seconds and every_nth must be given.
    """
    if (every_seconds is None) == (every_nth is None):
        raise ValueError("Give exactly one of every_seconds and every_nth")
    picked = []
    for index, timestamp, frame in iter_frames(path):
        if every_seconds is not None:
            if timestamp % every_seconds < 1e-9 or index == 0 and timestamp == 0.0:
                picked.append((index, timestamp, frame))
        else:
            if index % every_nth == 0:
                picked.append((index, timestamp, frame))
    return picked
