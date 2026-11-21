"""Enhance a cropped plate image so OCR can read it better."""

import cv2
import numpy as np


def to_grayscale(image):
    """Return a grayscale copy, or the image itself if already gray."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_threshold(gray_image):
    """Binarise with Otsu's method: dark text on a white plate."""
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def sharpen(image):
    """Unsharp mask: make edges crisper without changing the size."""
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def denoise(image):
    """Remove small specks while keeping the text edges."""
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def enhance_plate(plate_image):
    """Full pipeline for a cropped plate: gray, denoise, sharpen,
    and return both the enhanced gray image and its thresholded
    version, so the reader can try whichever reads better."""
    gray = to_grayscale(plate_image)
    gray = denoise(gray)
    gray = sharpen(gray)
    return gray, apply_threshold(gray)
