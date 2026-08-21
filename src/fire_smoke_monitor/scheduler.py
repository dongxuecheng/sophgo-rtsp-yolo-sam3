from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence

from .domain import ModelRuntime
from .streams import FrameUnavailableError


logger = logging.getLogger(__name__)


@dataclass
class _Job:
    model_name: str
    camera_id: str
    next_due: float


class InferenceScheduler:
    """Serial TPU scheduler: drain due YOLO work, then run one SAM3 job."""

    def __init__(
        self,
        camera_ids: Sequence[str],
        models: Mapping[str, ModelRuntime],
        source: Any,
        writer: Any,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._models: Dict[str, ModelRuntime] = dict(models)
        self._source = source
        self._writer = writer
        self._clock = clock
        start = clock()
        self._jobs: List[_Job] = [
            _Job(model_name=model_name, camera_id=camera_id, next_due=start)
            for model_name in self._models
            for camera_id in camera_ids
        ]

    def run(self, stop_event: threading.Event) -> None:
        logger.info("推理调度已启动 models=%s", ",".join(self._models))
        while not stop_event.is_set():
            work_count = self.run_iteration()
            if work_count:
                continue
            wait_sec = self._seconds_until_next_job()
            stop_event.wait(min(max(wait_sec, 0.01), 0.5))
        logger.info("推理调度已停止")

    def run_iteration(self) -> int:
        work_count = 0
        now = self._clock()

        # YOLO is short and interval-sensitive: clear all due YOLO jobs first.
        due_yolo = sorted(
            (
                job
                for job in self._jobs
                if job.model_name == "yolo26" and job.next_due <= now
            ),
            key=lambda job: job.next_due,
        )
        for job in due_yolo:
            self._run_job(job)
            work_count += 1

        # SAM3 is much heavier. One job per iteration lets newly due YOLO work
        # regain priority between two cameras' SAM3 runs.
        now = self._clock()
        due_sam = [
            job
            for job in self._jobs
            if job.model_name == "sam3" and job.next_due <= now
        ]
        if due_sam:
            self._run_job(min(due_sam, key=lambda job: job.next_due))
            work_count += 1
        return work_count

    def _run_job(self, job: _Job) -> None:
        runtime = self._models[job.model_name]
        started = self._clock()
        try:
            image = self._source.read(job.camera_id)
            detections = runtime.detector.detect(image)
            saved = self._writer.save(
                job.model_name,
                job.camera_id,
                image,
                detections,
                runtime.save_cooldown_sec,
            )
            if detections:
                logger.info(
                    "检测命中 model=%s camera=%s count=%d saved=%s elapsed=%.3fs",
                    job.model_name,
                    job.camera_id,
                    len(detections),
                    str(saved) if saved is not None else "cooldown/disk-guard",
                    self._clock() - started,
                )
            else:
                logger.debug(
                    "未命中 model=%s camera=%s elapsed=%.3fs",
                    job.model_name,
                    job.camera_id,
                    self._clock() - started,
                )
        except FrameUnavailableError as exc:
            logger.debug(
                "跳过当前无帧任务 model=%s camera=%s reason=%s",
                job.model_name,
                job.camera_id,
                exc,
            )
        except Exception:
            logger.exception("推理任务失败 model=%s camera=%s", job.model_name, job.camera_id)
        finally:
            # Fixed-delay scheduling intentionally avoids catch-up bursts after
            # a slow SAM3 inference or a temporarily unavailable stream.
            job.next_due = self._clock() + runtime.interval_sec

    def _seconds_until_next_job(self) -> float:
        if not self._jobs:
            return 0.5
        return min(job.next_due for job in self._jobs) - self._clock()
