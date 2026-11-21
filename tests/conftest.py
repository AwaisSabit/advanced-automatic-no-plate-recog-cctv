"""Shared helpers for PlateForge tests: synthetic videos and plates."""

import cv2
import numpy as np
import pytest


def make_plate_image(text="ABC-123", width=480, height=120):
    """Draw a clear white plate with dark text, like a real plate."""
    img = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(img, (4, 4), (width - 5, height - 5), (0, 0, 0), 3)
    cv2.putText(
        img, text, (30, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4, cv2.LINE_AA
    )
    return img


def make_car_frame(width=640, height=360, car_x=100, plate_text="ABC-123"):
    """Draw one CCTV-style frame: plain background, a car rectangle,
    and a white plate with dark text on the car."""
    frame = np.full((height, width, 3), 60, np.uint8)
    cv2.rectangle(frame, (0, height - 60), (width, height), (90, 90, 90), -1)
    cv2.rectangle(frame, (car_x, 150), (car_x + 260, 260), (30, 80, 160), -1)
    cv2.rectangle(frame, (car_x + 40, 210), (car_x + 220, 250), (255, 255, 255), -1)
    cv2.rectangle(frame, (car_x + 40, 210), (car_x + 220, 250), (0, 0, 0), 2)
    cv2.putText(
        frame,
        plate_text,
        (car_x + 48, 242),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def make_video(path, frames=30, fps=10, width=640, height=360, plate_text="ABC-123"):
    """Write a small synthetic CCTV video: a car driving left to right."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(frames):
        car_x = 20 + i * 12
        writer.write(make_car_frame(width, height, car_x, plate_text))
    writer.release()
    return path


def make_still_video(path, frames=10, fps=10, width=640, height=360):
    """Write a video where nothing moves at all."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    frame = make_car_frame(width, height, car_x=100)
    for _ in range(frames):
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def video_path(tmp_path):
    return make_video(tmp_path / "cctv.mp4")


@pytest.fixture
def plate_image(tmp_path):
    path = tmp_path / "plate.png"
    cv2.imwrite(str(path), make_plate_image("ABC-123"))
    return path
