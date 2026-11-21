"""Tests for frame extraction by interval."""

from plateforge.extract import extract_at_timestamps, extract_by_interval, extract_by_range


def test_extract_every_second(video_path):
    frames = extract_by_interval(str(video_path), every_seconds=1)
    assert [f[0] for f in frames] == [0, 10, 20]
    assert [f[1] for f in frames] == [0.0, 1.0, 2.0]


def test_extract_every_nth_frame(video_path):
    frames = extract_by_interval(str(video_path), every_nth=15)
    assert [f[0] for f in frames] == [0, 15]


def test_extract_by_range(video_path):
    frames = extract_by_range(str(video_path), 1.0, 2.0)
    assert [f[1] for f in frames] == [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]


def test_extract_by_range_with_interval(video_path):
    frames = extract_by_range(str(video_path), 0.5, 2.5, every_seconds=1)
    assert [f[1] for f in frames] == [0.5, 1.5, 2.5]


def test_extract_at_timestamps(video_path):
    frames = extract_at_timestamps(str(video_path), [0.0, 1.55, 2.9])
    assert [f[0] for f in frames] == [0, 15, 29]
    assert [f[1] for f in frames] == [0.0, 1.5, 2.9]
