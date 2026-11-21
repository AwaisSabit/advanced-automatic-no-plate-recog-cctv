"""Tests for OCR reading of plate text (real RapidOCR, no stubs)."""

from conftest import make_plate_image
from plateforge.ocr import (
    clean_plate_text,
    confidence_score,
    filter_readings,
    is_valid_plate,
    read_plate_text,
    read_text,
)


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


def test_cleaning_normalises_spaces_and_hyphens():
    assert clean_plate_text("abc 123") == "ABC-123"
    assert clean_plate_text("ABC-123") == "ABC-123"
    assert clean_plate_text("  ABC   12B   123  ") == "ABC-12B-123"


def test_all_plate_formats_validate():
    for plate in (
        "ABC 123",
        "ABC-12-133",
        "ABC 12B 123",
        "ABC-123",
        "ABC 1234",
        "AB-123",
        "ABC-12-1234",
        "LEK-17-1269",
        "ABC-12B-123",
    ):
        assert is_valid_plate(plate), plate


def test_plate_with_no_digits_is_rejected():
    assert not is_valid_plate("ABC-DEF")
    assert not is_valid_plate("ABC")
    assert not is_valid_plate("")


def test_ocr_confusion_fixed_by_position():
    assert clean_plate_text("ABC-1O3") == "ABC-103"
    assert clean_plate_text("AB0-123") == "ABO-123"


def test_confidence_score_rewards_valid_plates():
    assert confidence_score("ABC-123", 0.9) == 0.95
    assert confidence_score("ABC-123", 0.99) == 1.0
    assert confidence_score("ABC-DEF", 0.9) == 0.63
    assert confidence_score("", 0.9) == 0.0


def test_clear_filter_keeps_only_readable_plates():
    from plateforge.models import PlateReading

    good = PlateReading("ABC-123", "ABC-123", 0.95, (0, 0, 10, 10), 0, 0.0, True)
    weak = PlateReading("ABC-123", "ABC-123", 0.5, (0, 0, 10, 10), 1, 0.1, True)
    bad = PlateReading("ABC-DEF", "ABC-DEF", 0.95, (0, 0, 10, 10), 2, 0.2, False)
    assert filter_readings([good, weak, bad]) == [good]
    assert filter_readings([good, weak, bad], min_confidence=0.5) == [good, weak]
