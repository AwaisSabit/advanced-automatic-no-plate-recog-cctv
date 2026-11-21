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


def score_candidate(candidate, frame_shape):
    """Score how plate-like a candidate is, from 0 to 100.

    Real plates are wide and flat (aspect about 3–5), a sensible
    size for the frame, and sit in the lower half of the image where
    cars drive. Each of those earns points.
    """
    height, width = frame_shape[:2]
    score = 0.0
    aspect = candidate.aspect_ratio
    if 3.0 <= aspect <= 5.0:
        score += 40
    elif 2.0 <= aspect <= 6.0:
        score += 20
    frame_area = width * height
    share = candidate.area / frame_area if frame_area else 0
    if 0.005 <= share <= 0.15:
        score += 30
    elif 0.001 <= share <= 0.25:
        score += 15
    centre_y = candidate.y + candidate.height / 2
    if centre_y > height * 0.4:
        score += 30
    elif centre_y > height * 0.25:
        score += 15
    return round(min(score, 100.0), 1)


def score_candidates(candidates, frame_shape):
    """Score every candidate in place and return them best first."""
    for cand in candidates:
        cand.score = score_candidate(cand, frame_shape)
    return sorted(candidates, key=lambda c: c.score, reverse=True)


def filter_candidates(candidates, frame_shape, min_score=50):
    """Keep only candidates that score high enough to be a plate."""
    return [c for c in score_candidates(candidates, frame_shape) if c.score >= min_score]


def crop_plate(frame, candidate, padding=5):
    """Cut the plate region out of a frame, with a small border."""
    h, w = frame.shape[:2]
    x = max(0, candidate.x - padding)
    y = max(0, candidate.y - padding)
    x2 = min(w, candidate.x + candidate.width + padding)
    y2 = min(h, candidate.y + candidate.height + padding)
    return frame[y:y2, x:x2]


def zoom_plate(plate_image, scale=2.0):
    """Enlarge a cropped plate so the text is bigger for OCR.

    Uses cubic interpolation. Zooming cannot add detail that is not
    in the pixels, but a bigger plate reads better.
    """
    if scale <= 0:
        raise ValueError("Zoom scale must be positive")
    height, width = plate_image.shape[:2]
    return cv2.resize(
        plate_image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_CUBIC,
    )
