"""Tests for plate region detection."""

from conftest import make_car_frame
from plateforge.detect import find_plate_candidates


def test_plate_region_found_in_car_frame():
    frame = make_car_frame(car_x=100)
    candidates = find_plate_candidates(frame)
    assert len(candidates) >= 1
    cand = candidates[0]
    # The plate in the test frame is at x=140..320, y=210..250.
    assert 120 <= cand.x <= 160
    assert 195 <= cand.y <= 225
    assert 150 <= cand.width <= 220
    assert 25 <= cand.height <= 60
    assert 2.0 <= cand.aspect_ratio <= 6.0


def test_no_plate_in_plain_frame():
    import numpy as np

    frame = np.full((360, 640, 3), 60, np.uint8)
    assert find_plate_candidates(frame) == []
