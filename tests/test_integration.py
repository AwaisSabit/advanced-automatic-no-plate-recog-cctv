"""End-to-end tests: video in, plate log out."""

import os

from plateforge.batch import process_video
from plateforge.detect import crop_plate, filter_candidates, find_plate_candidates, zoom_plate
from plateforge.ocr import clean_plate_text, is_valid_plate, read_plate_text


def test_full_pipeline_reads_plate_from_video(video_path, tmp_path):
    entries = process_video(str(video_path), str(tmp_path / "out"), every_seconds=1)
    assert len(entries) == 1
    assert entries[0].plate == "ABC-123"
    assert entries[0].confidence >= 0.8


def test_detection_ocr_and_cleaning_agree_on_one_frame(video_path):
    from plateforge.extract import extract_by_interval

    _, _, frame = extract_by_interval(str(video_path), every_seconds=1)[1]
    candidates = filter_candidates(find_plate_candidates(frame), frame.shape)
    assert candidates
    cropped = zoom_plate(crop_plate(frame, candidates[0]), scale=2.0)
    raw, conf = read_plate_text(cropped)
    cleaned = clean_plate_text(raw)
    assert cleaned == "ABC-123"
    assert is_valid_plate(cleaned)
    assert conf > 0.8


def test_log_files_match_pipeline_output(video_path, tmp_path):
    from plateforge.log import load_csv, load_json

    out = tmp_path / "out"
    entries = process_video(str(video_path), str(out), every_seconds=1)
    assert [e.plate for e in load_csv(str(out / "plates.csv"))] == [e.plate for e in entries]
    assert [e.plate for e in load_json(str(out / "plates.json"))] == [e.plate for e in entries]
    assert os.path.exists(out / "report.md")
