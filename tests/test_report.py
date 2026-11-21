"""Tests for snapshot saving and the plate report."""

import cv2

from conftest import make_car_frame
from plateforge.models import LogEntry
from plateforge.report import build_report, save_snapshot


def test_snapshot_has_box_and_opens(tmp_path):
    frame = make_car_frame(car_x=100)
    path = save_snapshot(frame, (140, 210, 180, 40), "ABC-123", str(tmp_path / "snap.png"), 0.95)
    image = cv2.imread(path)
    assert image is not None
    assert image.shape == frame.shape
    assert image[210, 140].tolist() != frame[210, 140].tolist()


def test_report_lists_every_plate(tmp_path):
    entries = [
        LogEntry("ABC-123", 0.95, 1700000000.0, 1.0, 10, "cctv.mp4", "snap1.png"),
        LogEntry("ABC-12-1234", 0.9, 1700000000.0, 2.0, 20, "cctv.mp4", "snap2.png"),
    ]
    path = build_report(entries, str(tmp_path / "report.md"))
    text = open(path).read()
    assert "Plates detected: 2" in text
    assert "ABC-123" in text and "ABC-12-1234" in text
    assert "snap1.png" in text
