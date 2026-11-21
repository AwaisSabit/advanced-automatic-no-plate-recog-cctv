"""Tests for the command line interface."""

import os

from plateforge.cli import main


def test_cli_info(video_path, capsys):
    assert main(["info", str(video_path)]) == 0
    out = capsys.readouterr().out
    assert "Frames: 30" in out
    assert "640x360" in out


def test_cli_detect(video_path, capsys):
    assert main(["detect", str(video_path), "--every-seconds", "1"]) == 0
    assert "Plate regions found:" in capsys.readouterr().out


def test_cli_read_writes_log(video_path, tmp_path, capsys):
    out = tmp_path / "readings"
    assert main(["read", str(video_path), "--output-dir", str(out)]) == 0
    assert "ABC-123" in capsys.readouterr().out
    assert os.path.exists(out / "plates.csv")


def test_cli_log_and_search(video_path, tmp_path, capsys):
    out = tmp_path / "readings"
    main(["read", str(video_path), "--output-dir", str(out)])
    assert main(["log", str(out / "plates.csv")]) == 0
    assert "ABC-123" in capsys.readouterr().out
    assert main(["search", str(out / "plates.csv"), "--plate", "ABC"]) == 0
    assert "Matches: 1" in capsys.readouterr().out
