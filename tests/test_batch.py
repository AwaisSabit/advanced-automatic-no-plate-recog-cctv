"""Tests for batch processing of videos."""

import os

from plateforge.batch import process_batch_file, process_video


def test_process_video_writes_log_and_snapshots(video_path, tmp_path):
    out = tmp_path / "results"
    entries = process_video(str(video_path), str(out), every_seconds=1)
    assert entries
    assert all(e.plate == "ABC-123" for e in entries)
    assert os.path.exists(out / "plates.csv")
    assert os.path.exists(out / "plates.json")
    assert os.path.exists(out / "report.md")
    assert list((out / "snapshots").glob("*.png"))


def test_process_batch_file_handles_two_videos(video_path, tmp_path):
    other = tmp_path / "second.mp4"
    import shutil

    shutil.copy(video_path, other)
    list_path = tmp_path / "videos.txt"
    list_path.write_text(f"{video_path}\n{other}\n")
    results = process_batch_file(str(list_path), str(tmp_path / "batch"))
    assert len(results) == 2
    assert all(results.values())
