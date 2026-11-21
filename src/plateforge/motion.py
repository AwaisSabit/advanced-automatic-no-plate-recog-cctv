"""Motion detection: find what moved between two frames."""

import cv2
import numpy as np


def frame_difference(prev_frame, frame, threshold=25):
    """Return the binary mask of pixels that changed between frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def motion_boxes(prev_frame, frame, min_area=500, threshold=25):
    """Return bounding boxes (x, y, w, h) of regions that moved."""
    mask = frame_difference(prev_frame, frame, threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def has_motion(prev_frame, frame, min_area=500, threshold=25):
    """True when something big enough moved between the two frames."""
    return len(motion_boxes(prev_frame, frame, min_area, threshold)) > 0
