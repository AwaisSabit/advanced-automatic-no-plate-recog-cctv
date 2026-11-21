"""Read plate text with RapidOCR and clean it into a plate number.

OCR is done by the real RapidOCR library (CPU-only ONNX models, no
torch). Nothing here hardcodes a plate value: every result comes
from the image given.
"""

import cv2
import numpy as np

from plateforge.enhance import enhance_plate

_reader = None


def get_reader():
    """Create the RapidOCR engine once and reuse it (model load is slow)."""
    global _reader
    if _reader is None:
        from rapidocr_onnxruntime import RapidOCR

        _reader = RapidOCR()
    return _reader


def read_text(image):
    """Read all text lines in an image.

    Returns a list of (text, confidence) sorted by confidence,
    best first. Accepts a file path, a BGR image or a gray image.
    """
    if isinstance(image, str):
        result, _ = get_reader()(image)
    else:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result, _ = get_reader()(image)
    if not result:
        return []
    readings = [(text, float(conf)) for _, text, conf in result]
    readings.sort(key=lambda r: r[1], reverse=True)
    return readings


def read_plate_text(plate_image):
    """Read the plate text from a cropped plate image.

    Tries the enhanced gray image first, then the thresholded one,
    and returns the best reading found, or ("", 0.0) when nothing
    could be read.
    """
    gray, binary = enhance_plate(plate_image)
    for attempt in (gray, binary, plate_image):
        readings = read_text(attempt)
        if readings:
            return readings[0]
    return ("", 0.0)


import re

_LETTER_FIXES = str.maketrans({"0": "O", "1": "I", "5": "S"})


def _fix_letters(token):
    """Fix OCR confusions inside a letter group: 0 looks like O."""
    return token.translate(_LETTER_FIXES)


def _fix_digits(token):
    """Fix OCR confusions inside a digit group: O looks like 0.

    A letter that belongs in the group (like the B in 12B) is kept:
    only characters that are common digit confusions are replaced.
    """
    fixed = []
    for ch in token:
        if ch.isdigit():
            fixed.append(ch)
        elif ch in "Oo":
            fixed.append("0")
        elif ch in "Il":
            fixed.append("1")
        else:
            fixed.append(ch)
    return "".join(fixed)


def clean_plate_text(raw):
    """Normalise OCR output into one canonical plate form.

    Uppercase, collapse separators, and treat spaces and hyphens as
    the same separator, so "ABC 123", "ABC-123" and "ABC 123" all
    become "ABC-123". Letter groups get letter fixes, digit groups
    get digit fixes, so a misread O in the digits becomes a 0.
    Returns "" when nothing usable is left.
    """
    if not raw:
        return ""
    upper = raw.upper().strip()
    tokens = [t for t in re.split(r"[^A-Z0-9]+", upper) if t]
    if not tokens:
        return ""
    fixed = [_fix_letters(tokens[0])]
    for token in tokens[1:]:
        fixed.append(_fix_digits(token))
    return "-".join(fixed)


_GROUP_RE = re.compile(r"^\d{1,4}[A-Z]?$")


def is_valid_plate(text):
    """Permissive plate check, not a strict one.

    Accepts 2-3 letters at the start, then one or two groups of
    digits with an optional letter in the middle group, with spaces
    or hyphens as separators in any position. Examples that pass:
    ABC-123, ABC 123, ABC-1234, AB-123, ABC-12-133, ABC-12-1234
    (ABC-12-1234), ABC-12B-123, ABC 12B 123. A plate with no digits
    is rejected.
    """
    cleaned = clean_plate_text(text)
    if not cleaned:
        return False
    groups = cleaned.split("-")
    if len(groups) < 2 or len(groups) > 3:
        return False
    if not re.fullmatch(r"[A-Z]{2,3}", groups[0]):
        return False
    if not any(ch.isdigit() for ch in cleaned):
        return False
    return all(_GROUP_RE.fullmatch(g) for g in groups[1:])


def confidence_score(cleaned_text, ocr_confidence):
    """Combine the OCR confidence with the plate-format check.

    A reading that looks like a real plate keeps its OCR score
    (with a small bonus, capped at 1.0). A reading that fails the
    format check is probably a misread, so its score is cut by
    almost a third. The result is always between 0 and 1.
    """
    if not cleaned_text:
        return 0.0
    if is_valid_plate(cleaned_text):
        return round(min(1.0, ocr_confidence + 0.05), 3)
    return round(ocr_confidence * 0.7, 3)


def filter_readings(readings, min_confidence=0.8):
    """Keep only readable plates from a list of PlateReading objects.

    A reading is kept when its cleaned text is a valid plate and
    its combined confidence reaches the threshold. Blurry, partial
    and misread plates are dropped, so the log only holds plates
    worth acting on.
    """
    kept = []
    for reading in readings:
        score = confidence_score(reading.cleaned_text, reading.confidence)
        if reading.is_valid and score >= min_confidence:
            kept.append(reading)
    return kept
