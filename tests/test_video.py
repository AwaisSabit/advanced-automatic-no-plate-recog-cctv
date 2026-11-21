"""Tests for video loading and metadata reading."""

import pytest

from plateforge.video import get_video_info, iter_frames, open_video


def test_video_info_reports_real_values(video_path):
    info = get_video_info(str(video_path))
    assert info.frame_count == 30
    assert info.fps == 10.0
    assert info.duration_seconds == 3.0
    assert info.width == 640
    assert info.height == 360
    assert info.codec


def test_missing_video_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        open_video("/tmp/does-not-exist-12345.mp4")


def test_iter_frames_yields_every_frame_with_timestamps(video_path):
    frames = list(iter_frames(str(video_path)))
    assert len(frames) == 30
    assert frames[0][0] == 0
    assert frames[0][1] == 0.0
    assert frames[10][1] == 1.0
    assert frames[0][2].shape == (360, 640, 3)
