"""Log detected plates to CSV and JSON, with duplicate suppression."""

import csv
import json
import os
from datetime import datetime, timezone

from plateforge.models import LogEntry

CSV_FIELDS = ["plate", "confidence", "timestamp", "video_time", "frame_index", "video", "snapshot"]


def entry_to_dict(entry):
    return {
        "plate": entry.plate,
        "confidence": entry.confidence,
        "timestamp": entry.timestamp,
        "video_time": entry.video_time,
        "frame_index": entry.frame_index,
        "video": entry.video,
        "snapshot": entry.snapshot,
    }


def entry_from_dict(data):
    return LogEntry(
        plate=data["plate"],
        confidence=float(data["confidence"]),
        timestamp=float(data["timestamp"]),
        video_time=float(data["video_time"]),
        frame_index=int(data["frame_index"]),
        video=data["video"],
        snapshot=data.get("snapshot", ""),
    )


def append_to_csv(entries, path):
    """Append entries to a CSV log, writing the header on a new file."""
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for entry in entries:
            writer.writerow(entry_to_dict(entry))
    return len(entries)


def append_to_json(entries, path):
    """Append entries to a JSON log (a list, created if missing)."""
    existing = [entry_to_dict(e) for e in load_json(path)] if os.path.exists(path) else []
    existing.extend(entry_to_dict(e) for e in entries)
    with open(path, "w") as handle:
        json.dump(existing, handle, indent=2)
    return len(entries)


def load_csv(path):
    """Read a CSV log back into LogEntry objects."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as handle:
        return [entry_from_dict(row) for row in csv.DictReader(handle)]


def load_json(path):
    """Read a JSON log back into LogEntry objects."""
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [entry_from_dict(row) for row in json.load(handle)]


def log_time_now():
    """Current UTC time as an ISO string, for the log timestamp."""
    return datetime.now(timezone.utc).isoformat()


def suppress_duplicates(entries, window_seconds=5.0):
    """Keep the first sighting of a plate, drop repeats nearby.

    The same car stays in view for many frames, so without this the
    log fills with the same plate. A repeat of a plate within the
    window (default 5 seconds of video time) is dropped; the same
    plate seen again after the window is logged as a new sighting.
    """
    kept = []
    last_seen = {}
    for entry in sorted(entries, key=lambda e: e.video_time):
        previous = last_seen.get(entry.plate)
        if previous is not None and entry.video_time - previous < window_seconds:
            continue
        last_seen[entry.plate] = entry.video_time
        kept.append(entry)
    return kept


def search_log(entries, plate=None, start_time=None, end_time=None):
    """Search log entries by plate number, time range, or both.

    The plate match ignores spaces and hyphens and is case
    insensitive, so searching for "abc123" finds "ABC-123".
    Times are video times in seconds. Returns matches in time order.
    """
    wanted = None
    if plate:
        wanted = "".join(ch for ch in plate.upper() if ch.isalnum())
    found = []
    for entry in entries:
        if wanted:
            entry_plate = "".join(ch for ch in entry.plate.upper() if ch.isalnum())
            if wanted not in entry_plate:
                continue
        if start_time is not None and entry.video_time < start_time:
            continue
        if end_time is not None and entry.video_time > end_time:
            continue
        found.append(entry)
    return sorted(found, key=lambda e: e.video_time)
