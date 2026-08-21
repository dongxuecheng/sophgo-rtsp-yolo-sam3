import numpy as np

from fire_smoke_monitor.models import Sam3Detector


def make_detector(threshold=0.3, nms_threshold=0.5, max_detections=100):
    detector = object.__new__(Sam3Detector)
    detector._threshold = threshold
    detector._nms_threshold = nms_threshold
    detector._max_detections_per_prompt = max_detections
    return detector


def test_sam3_keeps_multiple_boxes_and_suppresses_duplicates():
    detector = make_detector()
    result = {
        "pred_scores": np.asarray([[[0.90], [0.80], [0.70], [0.29]]]),
        "pred_boxes": np.asarray(
            [
                [
                    [0.20, 0.20, 0.20, 0.20],
                    [0.21, 0.21, 0.20, 0.20],
                    [0.70, 0.70, 0.20, 0.20],
                    [0.50, 0.50, 0.20, 0.20],
                ]
            ]
        ),
    }

    detections = detector._detections_from_result(
        result, "person", width=100, height=100
    )

    assert len(detections) == 2
    assert [item.score for item in detections] == [0.9, 0.7]
    assert [(item.x1, item.y1, item.x2, item.y2) for item in detections] == [
        (10, 10, 30, 30),
        (60, 60, 80, 80),
    ]


def test_sam3_limits_detections_per_prompt_after_sorting():
    detector = make_detector(max_detections=2)
    result = {
        "pred_scores": np.asarray([[[0.70], [0.95], [0.80]]]),
        "pred_boxes": np.asarray(
            [
                [
                    [0.20, 0.20, 0.10, 0.10],
                    [0.50, 0.50, 0.10, 0.10],
                    [0.80, 0.80, 0.10, 0.10],
                ]
            ]
        ),
    }

    detections = detector._detections_from_result(
        result, "person", width=100, height=100
    )

    assert [item.score for item in detections] == [0.95, 0.8]


def test_sam3_ignores_non_finite_and_invalid_boxes():
    detector = make_detector()
    result = {
        "pred_scores": np.asarray([[[0.90], [np.nan], [0.80]]]),
        "pred_boxes": np.asarray(
            [
                [
                    [np.inf, 0.20, 0.20, 0.20],
                    [0.50, 0.50, 0.20, 0.20],
                    [0.70, 0.70, -0.20, 0.20],
                ]
            ]
        ),
    }

    detections = detector._detections_from_result(
        result, "person", width=100, height=100
    )

    assert detections == []
