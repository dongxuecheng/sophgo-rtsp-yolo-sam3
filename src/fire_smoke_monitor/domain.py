from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
class ModelRuntime:
    """Detector plus the small amount of scheduling policy it needs."""

    detector: Any
    interval_sec: float
    save_cooldown_sec: float

