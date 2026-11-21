"""Tests for plate logging, duplicate suppression and search helpers."""

from plateforge.log import (
    append_to_csv,
    append_to_json,
    load_csv,
    load_json,
    search_log,
    suppress_duplicates,
)
from plateforge.models import LogEntry


def _entry(plate="ABC-123", video_time=1.0, frame=10):
    return LogEntry(plate, 0.95, 1700000000.0, video_time, frame, "cctv.mp4", "snap.png")


def test_csv_round_trip(tmp_path):
    path = tmp_path / "plates.csv"
    append_to_csv([_entry(), _entry("ABC-12-1234", 2.0, 20)], str(path))
    rows = load_csv(str(path))
    assert [r.plate for r in rows] == ["ABC-123", "ABC-12-1234"]
    assert rows[0].confidence == 0.95
    assert rows[1].frame_index == 20


def test_json_round_trip(tmp_path):
    path = tmp_path / "plates.json"
    append_to_json([_entry()], str(path))
    append_to_json([_entry("ABC-12-1234", 2.0, 20)], str(path))
    rows = load_json(str(path))
    assert [r.plate for r in rows] == ["ABC-123", "ABC-12-1234"]


def test_duplicate_suppression_logs_same_plate_once_nearby():
    entries = [_entry(video_time=1.0), _entry(video_time=2.0), _entry(video_time=3.0)]
    assert len(suppress_duplicates(entries)) == 1


def test_same_plate_after_window_is_logged_again():
    entries = [_entry(video_time=1.0), _entry(video_time=10.0)]
    assert len(suppress_duplicates(entries)) == 2


def test_different_plates_are_all_kept():
    entries = [_entry(video_time=1.0), _entry("ABC-12-1234", 1.5, 15)]
    assert len(suppress_duplicates(entries)) == 2


def test_search_by_plate_ignores_case_and_separators():
    entries = [_entry(video_time=1.0), _entry("ABC-12-1234", 2.0, 20)]
    assert [e.plate for e in search_log(entries, plate="abc123")] == ["ABC-123"]
    assert [e.plate for e in search_log(entries, plate="ABC 12-1234")] == ["ABC-12-1234"]


def test_search_by_time_range():
    entries = [_entry(video_time=1.0), _entry("ABC-12-1234", 5.0, 50), _entry("XYZ-999", 9.0, 90)]
    assert [e.plate for e in search_log(entries, start_time=2.0, end_time=6.0)] == ["ABC-12-1234"]
    assert len(search_log(entries, plate="XYZ", start_time=0.0, end_time=5.0)) == 0
