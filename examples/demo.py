"""Demo: make a small synthetic CCTV video, then read its plates.

Run from the project root:  python examples/demo.py
Writes the video, log, snapshots and report to examples/output/.
No real footage or real plates are used: the car and its plate are
drawn with OpenCV.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from plateforge.batch import process_video  # noqa: E402
from plateforge.video import get_video_info  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
VIDEO = os.path.join(OUT, "cctv-demo.mp4")


def make_frame(width, height, car_x, plate_text="ABC-123"):
    frame = np.full((height, width, 3), 60, np.uint8)
    cv2.rectangle(frame, (0, height - 60), (width, height), (90, 90, 90), -1)
    cv2.rectangle(frame, (car_x, 150), (car_x + 260, 260), (30, 80, 160), -1)
    cv2.rectangle(frame, (car_x + 40, 210), (car_x + 220, 250), (255, 255, 255), -1)
    cv2.rectangle(frame, (car_x + 40, 210), (car_x + 220, 250), (0, 0, 0), 2)
    cv2.putText(
        frame, plate_text, (car_x + 48, 242),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA,
    )
    return frame


def main():
    os.makedirs(OUT, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO, fourcc, 10, (640, 360))
    for i in range(60):
        writer.write(make_frame(640, 360, 20 + i * 6))
    writer.release()
    info = get_video_info(VIDEO)
    print(f"Video: {info.duration_seconds}s, {info.fps} fps, {info.frame_count} frames")
    entries = process_video(VIDEO, OUT, every_seconds=1)
    for entry in entries:
        print(f"{entry.video_time:.2f}s  {entry.plate}  ({entry.confidence:.2f})")
    print(f"Logged {len(entries)} plate(s) in {OUT}")


if __name__ == "__main__":
    main()
