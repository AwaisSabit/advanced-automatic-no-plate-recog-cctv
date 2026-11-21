"""Tests for frame extraction by interval."""

from plateforge.extract import extract_by_interval


def test_extract_every_second(video_path):
    frames = extract_by_interval(str(video_path), every_seconds=1)
    assert [f[0] for f in frames] == [0, 10, 20]
    assert [f[1] for f in frames] == [0.0, 1.0, 2.0]


def test_extract_every_nth_frame(video_path):
    frames = extract_by_interval(str(video_path), every_nth=15)
    assert [f[0] for f in frames] == [0, 15]
