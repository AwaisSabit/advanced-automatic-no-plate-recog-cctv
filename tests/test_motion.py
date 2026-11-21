"""Tests for motion detection."""

from conftest import make_car_frame, make_still_video
from plateforge.motion import has_motion, motion_boxes


def test_motion_found_when_car_moves(tmp_path):
    still = make_car_frame(car_x=100)
    moved = make_car_frame(car_x=220)
    boxes = motion_boxes(still, moved)
    assert len(boxes) >= 1
    assert has_motion(still, moved)


def test_no_motion_in_identical_frames():
    frame = make_car_frame(car_x=100)
    assert motion_boxes(frame, frame.copy()) == []
    assert not has_motion(frame, frame.copy())


def test_no_motion_in_still_video(tmp_path):
    path = make_still_video(tmp_path / "still.mp4")
    from plateforge.video import iter_frames

    frames = [f for _, _, f in iter_frames(str(path))]
    assert not has_motion(frames[0], frames[5])
