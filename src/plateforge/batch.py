"""Run the full ANPR pipeline on one video, or many videos at once."""

import os
import time

from plateforge.detect import crop_plate, filter_candidates, find_plate_candidates, zoom_plate
from plateforge.extract import extract_by_interval
from plateforge.log import append_to_csv, append_to_json, suppress_duplicates
from plateforge.models import LogEntry, PlateReading
from plateforge.motion import vehicle_regions
from plateforge.ocr import clean_plate_text, confidence_score, filter_readings, is_valid_plate, read_plate_text
from plateforge.report import build_report, save_snapshot


def process_video(path, output_dir, every_seconds=1, min_confidence=0.8):
    """Run detection, OCR and logging on one video.

    Returns the list of LogEntry objects written. Writes plates.csv,
    plates.json, snapshots and a report.md into output_dir. Snapshots
    are saved only for plates that survive duplicate suppression, so
    the snapshot folder matches the log.
    """
    os.makedirs(output_dir, exist_ok=True)
    snapshot_dir = os.path.join(output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    found = []  # (entry, frame, box) for every readable plate seen
    prev_frame = None
    for index, timestamp, frame in extract_by_interval(path, every_seconds=every_seconds):
        vehicles = vehicle_regions(prev_frame, frame) if prev_frame is not None else [(0, 0, frame.shape[1], frame.shape[0])]
        prev_frame = frame
        if not vehicles:
            continue
        candidates = filter_candidates(find_plate_candidates(frame), frame.shape)
        for cand in candidates:
            cropped = crop_plate(frame, cand)
            zoomed = zoom_plate(cropped, scale=2.0)
            raw_text, ocr_conf = read_plate_text(zoomed)
            cleaned = clean_plate_text(raw_text)
            if not cleaned:
                continue
            score = confidence_score(cleaned, ocr_conf)
            reading = PlateReading(raw_text, cleaned, ocr_conf, cand.box, index, timestamp, is_valid_plate(cleaned))
            if not filter_readings([reading], min_confidence=min_confidence):
                continue
            entry = LogEntry(
                plate=cleaned,
                confidence=score,
                timestamp=time.time(),
                video_time=timestamp,
                frame_index=index,
                video=os.path.basename(path),
                snapshot="",
            )
            found.append((entry, frame.copy(), cand.box))
    entries = suppress_duplicates([entry for entry, _, _ in found])
    kept_frames = {(entry.plate, entry.frame_index): (frame, box) for entry, frame, box in found}
    for entry in entries:
        frame, box = kept_frames[(entry.plate, entry.frame_index)]
        snap_name = f"frame-{entry.frame_index}-{entry.plate}.png"
        save_snapshot(frame, box, entry.plate, os.path.join(snapshot_dir, snap_name), entry.confidence)
        entry.snapshot = snap_name
    append_to_csv(entries, os.path.join(output_dir, "plates.csv"))
    append_to_json(entries, os.path.join(output_dir, "plates.json"))
    build_report(entries, os.path.join(output_dir, "report.md"))
    return entries


def process_videos(video_paths, output_dir, every_seconds=1, min_confidence=0.8):
    """Process each video into its own subfolder of output_dir.

    Returns a dict of video path -> list of LogEntry objects.
    """
    results = {}
    for video in video_paths:
        name = os.path.splitext(os.path.basename(video))[0]
        results[video] = process_video(
            video,
            os.path.join(output_dir, name),
            every_seconds=every_seconds,
            min_confidence=min_confidence,
        )
    return results


def process_batch_file(list_path, output_dir, every_seconds=1, min_confidence=0.8):
    """Process every video listed (one path per line) in a text file."""
    with open(list_path) as handle:
        videos = [line.strip() for line in handle if line.strip()]
    return process_videos(videos, output_dir, every_seconds, min_confidence)
