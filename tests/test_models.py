import numpy as np

from fire_smoke_monitor.models import Yolo26Detector


class FakeImage:
    def width(self):
        return 100

    def height(self):
        return 80


class FakeModel:
    def __call__(self, images):
        assert len(images) == 1
        return [
            np.asarray(
                [
                    [-5, 2, 40, 60, 0.9, 1],
                    [10, 10, 20, 20, 0.8, 0],
                    [30, 30, 30, 50, 0.7, 2],
                    [0, 0, np.nan, 10, 0.6, 2],
                    [50, 20, 120, 90, 0.5, 2],
                ],
                dtype=np.float32,
            )
        ]


def test_yolo_filters_classes_invalid_rows_and_clamps_boxes():
    detector = object.__new__(Yolo26Detector)
    detector._model = FakeModel()
    detector._class_names = ("person", "fire", "smoke")
    detector._target_ids = {1, 2}

    detections = detector.detect(FakeImage())

    assert [item.label for item in detections] == ["fire", "smoke"]
    assert [(item.x1, item.y1, item.x2, item.y2) for item in detections] == [
        (0, 2, 40, 60),
        (50, 20, 99, 79),
    ]
