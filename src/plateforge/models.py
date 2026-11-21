"""Shared data types for PlateForge."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VideoInfo:
    """Metadata about a video file."""

    path: str
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str


@dataclass
class PlateCandidate:
    """One rectangular region that might contain a number plate."""

    x: int
    y: int
    width: int
    height: int
    area: float
    aspect_ratio: float
    score: float = 0.0

    @property
    def box(self):
        return (self.x, self.y, self.width, self.height)


@dataclass
class PlateReading:
    """One OCR reading of a number plate."""

    raw_text: str
    cleaned_text: str
    confidence: float
    box: tuple
    frame_index: int
    timestamp: float
    is_valid: bool = False


@dataclass
class FrameDetection:
    """Everything found in one video frame."""

    frame_index: int
    timestamp: float
    vehicle_boxes: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    readings: list = field(default_factory=list)


@dataclass
class LogEntry:
    """One plate logged with its time and source."""

    plate: str
    confidence: float
    timestamp: float
    video_time: float
    frame_index: int
    video: str
    snapshot: str = ""
