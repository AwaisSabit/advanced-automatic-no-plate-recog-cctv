"""Tests for plate logging, duplicate suppression and search helpers."""

from plateforge.log import append_to_csv, append_to_json, load_csv, load_json, suppress_duplicates
from plateforge.models import LogEntry


def _entry(plate="ABC-123", video_time=1.0, frame=10):
    return LogEntry(plate, 0.95, 1700000000.0, video_time, frame, "cctv.mp4", "snap.png")


def test_csv_round_trip(tmp_path):
    path = tmp_path / "plates.csv"
    append_to_csv([_entry(), _entry("LEK-17-1269", 2.0, 20)], str(path))
    rows = load_csv(str(path))
    assert [r.plate for r in rows] == ["ABC-123", "LEK-17-1269"]
    assert rows[0].confidence == 0.95
    assert rows[1].frame_index == 20


def test_json_round_trip(tmp_path):
    path = tmp_path / "plates.json"
    append_to_json([_entry()], str(path))
    append_to_json([_entry("LEK-17-1269", 2.0, 20)], str(path))
    rows = load_json(str(path))
    assert [r.plate for r in rows] == ["ABC-123", "LEK-17-1269"]


def test_duplicate_suppression_logs_same_plate_once_nearby():
    entries = [_entry(video_time=1.0), _entry(video_time=2.0), _entry(video_time=3.0)]
    assert len(suppress_duplicates(entries)) == 1


def test_same_plate_after_window_is_logged_again():
    entries = [_entry(video_time=1.0), _entry(video_time=10.0)]
    assert len(suppress_duplicates(entries)) == 2


def test_different_plates_are_all_kept():
    entries = [_entry(video_time=1.0), _entry("LEK-17-1269", 1.5, 15)]
    assert len(suppress_duplicates(entries)) == 2
