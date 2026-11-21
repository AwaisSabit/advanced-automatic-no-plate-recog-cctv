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


def extract_by_range(path, start_time, end_time, every_seconds=None):
    """Extract frames between start_time and end_time (seconds).

    With every_seconds set, only one frame per interval inside the
    range is kept. Raises ValueError for an empty or reversed range.
    """
    if start_time < 0 or end_time <= start_time:
        raise ValueError("Need 0 <= start_time < end_time")
    picked = []
    last_kept = None
    for index, timestamp, frame in iter_frames(path):
        if timestamp < start_time or timestamp > end_time:
            continue
        if every_seconds is not None:
            if last_kept is not None and timestamp - last_kept < every_seconds - 1e-9:
                continue
            last_kept = timestamp
        picked.append((index, timestamp, frame))
    return picked
