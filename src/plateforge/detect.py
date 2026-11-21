"""Detect number plate regions in a frame with edges and contours."""

import cv2
import numpy as np

from plateforge.models import PlateCandidate


def find_plate_candidates(frame, min_area=800, max_area=60000):
    """Find rectangular regions that could be a number plate.

    Plates are bright rectangles with a dark border and dark text, so
    they show up as strong edges. We look for four-sided contours
    whose shape is wide and flat, like a plate.
    Returns a list of PlateCandidate, largest area first.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue
        aspect = w / h
        if not 1.5 <= aspect <= 6.0:
            continue
        candidates.append(
            PlateCandidate(x=x, y=y, width=w, height=h, area=float(area), aspect_ratio=round(aspect, 2))
        )
    candidates.sort(key=lambda c: c.area, reverse=True)
    return _dedupe(candidates)


def _dedupe(candidates, overlap=0.5):
    """Drop candidates that mostly cover a bigger candidate."""
    kept = []
    for cand in candidates:
        duplicate = False
        for other in kept:
            ix = max(0, min(cand.x + cand.width, other.x + other.width) - max(cand.x, other.x))
            iy = max(0, min(cand.y + cand.height, other.y + other.height) - max(cand.y, other.y))
            inter = ix * iy
            if inter > overlap * min(cand.area, other.area):
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept
