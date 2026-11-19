"""CPU-only synthetic tests for frozen FLIR failure-evidence matching."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "code" / "scripts" / "export_flir_val_failure_evidence.py"
SPEC = importlib.util.spec_from_file_location("failure_export", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IoUTests(unittest.TestCase):
    def test_iou_native_xyxy_and_degenerate_box(self) -> None:
        box = np.array([0.0, 0.0, 10.0, 10.0])
        others = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 15.0],
            [20.0, 20.0, 20.0, 25.0],
        ])
        actual = MODULE.iou_one_to_many(box, others)
        np.testing.assert_allclose(actual, [1.0, 25.0 / 175.0, 0.0])

    def test_empty_candidates(self) -> None:
        actual = MODULE.iou_one_to_many(np.zeros(4), np.empty((0, 4)))
        self.assertEqual(actual.shape, (0,))


class MatchingTests(unittest.TestCase):
    def test_confidence_priority_same_class_one_to_one(self) -> None:
        gt_boxes = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
        gt_classes = np.array([0, 0])
        pred_boxes = np.array([
            [0, 0, 10, 10],       # lower confidence duplicate -> FP
            [0, 0, 10, 10],       # higher confidence -> GT 0
            [20, 20, 30, 30],     # -> GT 1
            [20, 20, 30, 30],     # wrong class -> FP
            [40, 40, 50, 50],     # below operating threshold, not FP
        ], dtype=float)
        pred_classes = np.array([0, 0, 0, 1, 0])
        pred_conf = np.array([0.8, 0.9, 0.7, 0.6, 0.249])

        matched, unmatched, operating = MODULE.match_predictions_to_gt(
            pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, 0.25, 0.50
        )

        self.assertEqual(matched[0][0], 1)
        self.assertEqual(matched[1][0], 2)
        self.assertEqual(unmatched, [0, 3])
        self.assertEqual(operating.tolist(), [1, 0, 2, 3])

    def test_threshold_is_inclusive_and_ties_are_stable(self) -> None:
        gt_boxes = np.array([[0, 0, 10, 10]], dtype=float)
        gt_classes = np.array([0])
        pred_boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
        pred_classes = np.array([0, 0])
        pred_conf = np.array([0.25, 0.25])

        matched, unmatched, operating = MODULE.match_predictions_to_gt(
            pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, 0.25, 0.50
        )

        self.assertEqual(matched[0][0], 0)
        self.assertEqual(unmatched, [1])
        self.assertEqual(operating.tolist(), [0, 1])

    def test_duplicate_gt_requires_distinct_predictions(self) -> None:
        # If duplicated labels survive upstream deduplication, a single
        # detection can match only one GT. This is the intended one-to-one
        # evaluation semantics, not an implicit duplicate-label collapse.
        gt_boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
        gt_classes = np.array([0, 0])
        pred_boxes = np.array([[0, 0, 10, 10]], dtype=float)
        pred_classes = np.array([0])
        pred_conf = np.array([0.9])

        matched, unmatched, _ = MODULE.match_predictions_to_gt(
            pred_boxes, pred_classes, pred_conf, gt_boxes, gt_classes, 0.25, 0.50
        )

        self.assertEqual(len(matched), 1)
        self.assertEqual(unmatched, [])

    def test_mismatched_array_lengths_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.match_predictions_to_gt(
                np.empty((1, 4)), np.empty(0), np.empty(1),
                np.empty((0, 4)), np.empty(0), 0.25, 0.50
            )


class PredictionValidationTests(unittest.TestCase):
    def test_valid_native_coordinate_predictions(self) -> None:
        MODULE.validate_prediction_arrays(
            np.array([[0.0, 0.0, 640.0, 512.0], [10.0, 20.0, 30.0, 40.0]]),
            np.array([0, 8]),
            np.array([0.25, 1.0]),
            (512, 640),
            {0, 8},
        )

    def test_resized_or_out_of_bounds_coordinates_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.validate_prediction_arrays(
                np.array([[0.0, 0.0, 640.0, 640.0]]),
                np.array([0]),
                np.array([0.9]),
                (512, 640),
                {0},
            )

    def test_explicit_native_boundary_clipping(self) -> None:
        boxes = np.array([[-2.0, -3.0, 645.0, 520.0], [10.0, 20.0, 30.0, 40.0]])
        overflow = MODULE.boundary_overflow(boxes, (512, 640))
        clipped = MODULE.clip_prediction_boxes(boxes, (512, 640))

        np.testing.assert_allclose(overflow, [[2, 3, 5, 8], [0, 0, 0, 0]])
        np.testing.assert_allclose(clipped, [[0, 0, 640, 512], [10, 20, 30, 40]])
        self.assertEqual(boxes[0, 0], -2.0, "clipping must not mutate the raw audit array")

    def test_unknown_class_and_nonfinite_values_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.validate_prediction_arrays(
                np.array([[0.0, 0.0, np.nan, 10.0]]),
                np.array([9]),
                np.array([0.9]),
                (512, 640),
                set(range(9)),
            )


class PredictionSourceTests(unittest.TestCase):
    def test_source_is_disk_backed_glob_not_in_memory_list(self) -> None:
        source = MODULE.build_prediction_source(Path("validation-images"))

        self.assertIsInstance(source, str)
        self.assertEqual(Path(source).name, "*.jpg")
        self.assertEqual(Path(source).parent, Path("validation-images"))


class PredictionSemanticsTests(unittest.TestCase):
    def test_yolo_and_rtdetr_postprocessing_are_explicit(self) -> None:
        yolo = MODULE.prediction_semantics("yolo")
        rtdetr = MODULE.prediction_semantics("rtdetr")

        self.assertTrue(yolo["nms_applied"])
        self.assertTrue(yolo["nms_iou_effective"])
        self.assertFalse(rtdetr["nms_applied"])
        self.assertFalse(rtdetr["nms_iou_effective"])

    def test_unknown_family_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.prediction_semantics("unknown")


if __name__ == "__main__":
    unittest.main()
