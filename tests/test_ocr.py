"""Tests for OCR reading of plate text (real RapidOCR, no stubs)."""

from conftest import make_plate_image
from plateforge.ocr import read_plate_text, read_text


def test_ocr_reads_clear_synthetic_plate():
    img = make_plate_image("ABC-123")
    text, confidence = read_plate_text(img)
    assert text == "ABC-123"
    assert confidence > 0.9


def test_ocr_reads_pakistani_style_plate():
    img = make_plate_image("LEK-17-1269")
    text, confidence = read_plate_text(img)
    assert text == "LEK-17-1269"
    assert confidence > 0.9


def test_ocr_returns_nothing_for_blank_image():
    import numpy as np

    blank = np.full((120, 480, 3), 255, np.uint8)
    assert read_text(blank) == []
