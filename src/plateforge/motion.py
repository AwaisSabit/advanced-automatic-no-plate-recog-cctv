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


def _boxes_overlap(a, b, gap=60):
    """True when boxes touch, or nearly touch (a car often moves in
    two separate difference regions with a gap between them)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (
        ax - gap < bx + bw
        and bx - gap < ax + aw
        and ay - gap < by + bh
        and by - gap < ay + ah
    )


def _merge_boxes(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x = min(ax, bx)
    y = min(ay, by)
    return (x, y, max(ax + aw, bx + bw) - x, max(ay + ah, by + bh) - y)


def vehicle_regions(prev_frame, frame, min_area=2000, threshold=25):
    """Return vehicle-sized regions from motion: wide, large boxes.

    Nearby motion boxes are merged first (a car often moves in pieces),
    then boxes too small or too tall and narrow to be a vehicle are
    dropped. Returns boxes sorted by area, largest first.
    """
    boxes = motion_boxes(prev_frame, frame, min_area=200, threshold=threshold)
    merged = []
    for box in boxes:
        for i, other in enumerate(merged):
            if _boxes_overlap(box, other):
                merged[i] = _merge_boxes(box, other)
                break
        else:
            merged.append(box)
    vehicles = []
    for x, y, w, h in merged:
        area = w * h
        aspect = w / h if h else 0
        if area >= min_area and 0.8 <= aspect <= 5.0:
            vehicles.append((x, y, w, h))
    vehicles.sort(key=lambda b: b[2] * b[3], reverse=True)
    return vehicles
