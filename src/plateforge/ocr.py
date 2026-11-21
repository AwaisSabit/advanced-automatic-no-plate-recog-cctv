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
