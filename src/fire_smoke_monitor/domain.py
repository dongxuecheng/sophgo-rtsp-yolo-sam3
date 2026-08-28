from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Detection:
    """A pixel-coordinate detection on the original frame."""

    label: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class UploadFrame:
    """Only compressed bytes cross into the HTTP thread; no device memory."""

    camera_id: str
    stream_name: str
    captured_at: datetime
    created_monotonic: float
    jpeg: bytes = field(repr=False)
