"""Save plate snapshots and build a report of detected plates."""

import os

import cv2

from plateforge.models import LogEntry


def save_snapshot(frame, box, plate_text, output_path, confidence=None):
    """Save a frame with a green box around the plate and its text.

    Returns the output path. The parent folder is created if needed.
    """
    x, y, w, h = box
    marked = frame.copy()
    cv2.rectangle(marked, (x, y), (x + w, y + h), (0, 200, 0), 3)
    label = plate_text
    if confidence is not None:
        label = f"{plate_text} ({confidence:.2f})"
    cv2.putText(
        marked,
        label,
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 0),
        2,
        cv2.LINE_AA,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, marked)
    return output_path


def build_report(entries, output_path, title="PlateForge report"):
    """Write a Markdown report listing every detected plate.

    Each row has the plate, confidence, video time, frame and the
    snapshot file name. Returns the output path.
    """
    lines = [f"# {title}", "", f"Plates detected: {len(entries)}", ""]
    lines.append("| Plate | Confidence | Video time | Frame | Snapshot |")
    lines.append("|---|---|---|---|---|")
    for entry in entries:
        lines.append(
            f"| {entry.plate} | {entry.confidence:.2f} | {entry.video_time:.2f}s "
            f"| {entry.frame_index} | {os.path.basename(entry.snapshot)} |"
        )
    lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write("\n".join(lines))
    return output_path
