"""Tests for plate image enhancement."""

import numpy as np

from conftest import make_plate_image
from plateforge.enhance import apply_threshold, denoise, enhance_plate, sharpen, to_grayscale


def test_grayscale_and_threshold_keep_size():
    img = make_plate_image("ABC-123")
    gray = to_grayscale(img)
    assert len(gray.shape) == 2
    binary = apply_threshold(gray)
    assert binary.shape == gray.shape
    assert set(np.unique(binary)).issubset({0, 255})


def test_sharpen_and_denoise_change_pixels_not_size():
    img = make_plate_image("ABC-123")
    gray = to_grayscale(img)
    assert sharpen(gray).shape == gray.shape
    assert denoise(gray).shape == gray.shape
    assert not np.array_equal(sharpen(gray), gray)


def test_enhance_plate_returns_gray_and_binary():
    img = make_plate_image("ABC-123")
    gray, binary = enhance_plate(img)
    assert gray.shape == img.shape[:2]
    assert binary.shape == img.shape[:2]
