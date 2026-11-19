"""Export fixed-protocol FLIR validation predictions and TP/FP/FN evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty(0, dtype=float)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(0.0, bottom_right - top_left), axis=1)
    area_a = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_confidences: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    operating_conf: float,
    match_iou: float,
) -> tuple[dict[int, tuple[int, float]], list[int], np.ndarray]:
    """Confidence-priority, same-class, one-to-one greedy matching.

    Predictions below ``operating_conf`` are retained by the caller in the
    low-threshold JSONL evidence, but they are intentionally excluded from the
    fixed-operating-point TP/FP/FN table.
    """
    lengths = {len(pred_boxes), len(pred_classes), len(pred_confidences)}
    if len(lengths) != 1:
        raise ValueError("Prediction boxes, classes and confidences have different lengths")
    if len(gt_boxes) != len(gt_classes):
        raise ValueError("GT boxes and classes have different lengths")

    operating_indices = np.flatnonzero(pred_confidences >= operating_conf)
    # A stable sort makes exact confidence ties deterministic using detector
    # output order as the secondary key.
    operating_indices = operating_indices[
        np.argsort(-pred_confidences[operating_indices], kind="stable")
    ]
    matched_gt: dict[int, tuple[int, float]] = {}
    unmatched_predictions: list[int] = []
    available_gt = np.ones(len(gt_boxes), dtype=bool)

    for pred_index_raw in operating_indices:
        pred_index = int(pred_index_raw)
        candidates = np.flatnonzero(
            (gt_classes == pred_classes[pred_index]) & available_gt
        )
        overlaps = iou_one_to_many(pred_boxes[pred_index], gt_boxes[candidates])
        if overlaps.size and overlaps.max() >= match_iou:
            local_best = int(overlaps.argmax())
            gt_index = int(candidates[local_best])
            matched_gt[gt_index] = (pred_index, float(overlaps[local_best]))
            available_gt[gt_index] = False
        else:
            unmatched_predictions.append(pred_index)

    return matched_gt, unmatched_predictions, operating_indices


def _name_map(names: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def build_prediction_source(images_dir: Path) -> str:
    """Return a disk-backed source so Ultralytics honors the requested batch.

    Passing a Python list of filenames to Ultralytics 8.4.138 selects
    ``LoadPilAndNumpy``, which loads the whole list as one in-memory batch and
    ignores the predictor's ``batch`` argument. A glob string selects
    ``LoadImagesAndVideos`` and preserves its deterministic sorted order.
    """
    return str(images_dir / "*.jpg")


def prediction_semantics(model_family: str) -> dict[str, object]:
    """Describe architecture-native prediction filtering without changing it."""
    if model_family == "yolo":
        return {
            "model_family": "YOLO",
            "postprocessing": "Ultralytics native post-NMS predictions",
            "nms_applied": True,
            "nms_iou_effective": True,
        }
    if model_family == "rtdetr":
        return {
            "model_family": "RT-DETR",
            "postprocessing": "Ultralytics native NMS-free query predictions filtered by confidence and max_det",
            "nms_applied": False,
            "nms_iou_effective": False,
        }
    raise ValueError(f"Unsupported model family: {model_family}")


def boundary_overflow(boxes: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """Return left, top, right and bottom excess in native-image pixels."""
    height, width = original_shape
    return np.column_stack((
        np.maximum(0.0, -boxes[:, 0]),
        np.maximum(0.0, -boxes[:, 1]),
        np.maximum(0.0, boxes[:, 2] - width),
        np.maximum(0.0, boxes[:, 3] - height),
    ))


def clip_prediction_boxes(boxes: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """Clip xyxy boxes to the native image boundary, returning a copy."""
    height, width = original_shape
    clipped = boxes.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0.0, float(width))
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0.0, float(height))
    return clipped


def validate_prediction_arrays(
    boxes: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    original_shape: tuple[int, int],
    valid_class_ids: set[int],
) -> None:
    """Fail closed if detector outputs cannot be native-image xyxy evidence."""
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"Prediction boxes must have shape (N, 4), got {boxes.shape}")
    if len({len(boxes), len(classes), len(confidences)}) != 1:
        raise ValueError("Prediction boxes, classes and confidences have different lengths")
    if not (np.isfinite(boxes).all() and np.isfinite(confidences).all()):
        raise ValueError("Prediction boxes or confidences contain non-finite values")
    if np.any((confidences < 0.0) | (confidences > 1.0)):
        raise ValueError("Prediction confidence falls outside [0, 1]")
    if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
        raise ValueError("Prediction xyxy coordinates are inverted")
    height, width = original_shape
    tolerance = 1e-3
    if (
        np.any(boxes[:, [0, 2]] < -tolerance)
        or np.any(boxes[:, [0, 2]] > width + tolerance)
        or np.any(boxes[:, [1, 3]] < -tolerance)
        or np.any(boxes[:, [1, 3]] > height + tolerance)
    ):
        raise ValueError("Prediction boxes are not in the native original-image coordinate space")
    unknown_classes = set(np.asarray(classes, dtype=int).tolist()) - valid_class_ids
    if unknown_classes:
        raise ValueError(f"Predictions contain unknown class IDs: {sorted(unknown_classes)}")


def main() -> None:
    # Delayed import keeps pure matching functions testable in a CPU-only
    # environment without initializing Ultralytics or CUDA.
    from ultralytics import RTDETR, YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-family", choices=("yolo", "rtdetr"), default="yolo")
    parser.add_argument(
        "--clip-to-image", action="store_true",
        help="Explicitly clip native-scaled boxes to image bounds before frozen matching",
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--gt-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--prediction-conf", type=float, default=0.001)
    parser.add_argument("--operating-conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    if args.clip_to_image and args.model_family != "rtdetr":
        raise ValueError("--clip-to-image is reserved for the audited RT-DETR interface adaptation")

    args.output.mkdir(parents=True, exist_ok=False)
    incomplete_marker = args.output / "_INCOMPLETE"
    incomplete_marker.write_text(
        "Evidence export has not completed; do not use this directory for analysis.\n",
        encoding="utf-8",
    )
    metadata = pd.read_csv(args.gt_metadata)
    required_columns = {
        "image", "object_index", "class_id", "class_name", "x1", "y1", "x2", "y2",
        "size_bin", "ltc", "contrast_bin", "tiny_proxy_flag", "background_mode",
    }
    missing_columns = sorted(required_columns - set(metadata.columns))
    if missing_columns:
        raise ValueError(f"GT metadata is missing columns: {missing_columns}")
    if metadata.duplicated(["image", "object_index"]).any():
        raise ValueError("GT metadata contains duplicate (image, object_index) identifiers")
    grouped = {name: group.sort_values("object_index") for name, group in metadata.groupby("image")}
    # Ultralytics sorts globbed filenames as strings (case-sensitive). Windows
    # Path ordering is case-insensitive, so sort by the string to keep the
    # expected image sequence aligned with the disk-backed prediction stream.
    images = sorted(args.images.glob("*.jpg"), key=lambda path: str(path))
    if {path.name for path in images} != set(grouped):
        raise ValueError("Validation image names and GT metadata image names differ")

    model = (YOLO if args.model_family == "yolo" else RTDETR)(str(args.model))
    model_names = _name_map(model.names)
    metadata_names = (
        metadata[["class_id", "class_name"]]
        .drop_duplicates()
        .sort_values("class_id")
    )
    if metadata_names["class_id"].duplicated().any():
        raise ValueError("GT metadata maps one class ID to multiple class names")
    expected_names = dict(zip(metadata_names["class_id"].astype(int), metadata_names["class_name"].astype(str)))
    if any(model_names.get(class_id) != name for class_id, name in expected_names.items()):
        raise ValueError(
            f"Model and GT class mappings differ: model={model_names}, gt={expected_names}"
        )
    results = model.predict(
        source=build_prediction_source(args.images),
        stream=True,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.prediction_conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        save=False,
        verbose=False,
    )

    fieldnames = [
        "row_type", "outcome", "image", "object_index", "class_id", "class_name",
        "x1", "y1", "x2", "y2", "confidence", "matched_iou", "size_bin", "ltc",
        "contrast_bin", "tiny_proxy_flag", "background_mode",
    ]
    totals = {
        "gt": 0, "tp": 0, "fn": 0, "fp": 0,
        "all_saved_predictions": 0, "operating_predictions": 0,
        "raw_out_of_bounds_predictions": 0,
        "raw_operating_out_of_bounds_predictions": 0,
    }
    max_raw_overflow_ltrb_px = np.zeros(4, dtype=float)
    predictions_path = args.output / "predictions_all.jsonl"
    outcomes_path = args.output / "fixed_threshold_outcomes.csv"
    image_summary_path = args.output / "image_level_summary.csv"
    image_summary_fields = [
        "image", "gt", "tp", "fn", "fp", "predictions_low_threshold",
        "predictions_operating_threshold",
    ]
    with (
        predictions_path.open("w", encoding="utf-8") as prediction_handle,
        outcomes_path.open("w", encoding="utf-8", newline="") as outcome_handle,
        image_summary_path.open("w", encoding="utf-8", newline="") as image_summary_handle,
    ):
        writer = csv.DictWriter(outcome_handle, fieldnames=fieldnames)
        writer.writeheader()
        image_writer = csv.DictWriter(image_summary_handle, fieldnames=image_summary_fields)
        image_writer.writeheader()
        for image_path, result in zip(images, results, strict=True):
            result_path = Path(result.path)
            if result_path.name != image_path.name:
                raise ValueError(
                    f"Prediction order/path mismatch: expected {image_path.name}, got {result_path.name}"
                )
            if tuple(result.orig_shape) != (512, 640):
                raise ValueError(
                    f"Unexpected original image shape for {image_path.name}: {result.orig_shape}"
                )
            pred_boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
            pred_classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            pred_confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
            raw_overflow = boundary_overflow(pred_boxes, tuple(result.orig_shape))
            raw_oob = raw_overflow.max(axis=1) > 0 if len(pred_boxes) else np.zeros(0, dtype=bool)
            totals["raw_out_of_bounds_predictions"] += int(raw_oob.sum())
            totals["raw_operating_out_of_bounds_predictions"] += int(
                (raw_oob & (pred_confidences >= args.operating_conf)).sum()
            )
            if len(pred_boxes):
                max_raw_overflow_ltrb_px = np.maximum(max_raw_overflow_ltrb_px, raw_overflow.max(axis=0))
            if args.clip_to_image:
                pred_boxes = clip_prediction_boxes(pred_boxes, tuple(result.orig_shape))
            validate_prediction_arrays(
                pred_boxes,
                pred_classes,
                pred_confidences,
                tuple(result.orig_shape),
                set(model_names),
            )
            totals["all_saved_predictions"] += len(pred_boxes)
            prediction_handle.write(
                json.dumps(
                    {
                        "image": image_path.name,
                        "boxes": pred_boxes.tolist(),
                        "classes": pred_classes.tolist(),
                        "confidences": pred_confidences.tolist(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

            gt = grouped[image_path.name].reset_index(drop=True)
            gt_boxes = gt[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            gt_classes = gt["class_id"].to_numpy(dtype=int)
            matched_gt, unmatched_predictions, operating_indices = match_predictions_to_gt(
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                pred_confidences=pred_confidences,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                operating_conf=args.operating_conf,
                match_iou=args.match_iou,
            )
            totals["operating_predictions"] += len(operating_indices)

            for gt_index, row in gt.iterrows():
                match = matched_gt.get(int(gt_index))
                totals["gt"] += 1
                totals["tp" if match else "fn"] += 1
                writer.writerow(
                    {
                        "row_type": "gt",
                        "outcome": "TP" if match else "FN",
                        "image": image_path.name,
                        "object_index": int(row.object_index),
                        "class_id": int(row.class_id),
                        "class_name": row.class_name,
                        "x1": row.x1,
                        "y1": row.y1,
                        "x2": row.x2,
                        "y2": row.y2,
                        "confidence": "" if match is None else pred_confidences[match[0]],
                        "matched_iou": "" if match is None else match[1],
                        "size_bin": row.size_bin,
                        "ltc": row.ltc,
                        "contrast_bin": row.contrast_bin,
                        "tiny_proxy_flag": row.tiny_proxy_flag,
                        "background_mode": row.background_mode,
                    }
                )
            for pred_index in unmatched_predictions:
                totals["fp"] += 1
                writer.writerow(
                    {
                        "row_type": "prediction",
                        "outcome": "FP",
                        "image": image_path.name,
                        "object_index": "",
                        "class_id": pred_classes[pred_index],
                        "class_name": result.names[int(pred_classes[pred_index])],
                        "x1": pred_boxes[pred_index, 0],
                        "y1": pred_boxes[pred_index, 1],
                        "x2": pred_boxes[pred_index, 2],
                        "y2": pred_boxes[pred_index, 3],
                        "confidence": pred_confidences[pred_index],
                        "matched_iou": "",
                        "size_bin": "",
                        "ltc": "",
                        "contrast_bin": "",
                        "tiny_proxy_flag": "",
                        "background_mode": "",
                    }
                )
            image_tp = len(matched_gt)
            image_fn = len(gt) - image_tp
            image_fp = len(unmatched_predictions)
            image_writer.writerow(
                {
                    "image": image_path.name,
                    "gt": len(gt),
                    "tp": image_tp,
                    "fn": image_fn,
                    "fp": image_fp,
                    "predictions_low_threshold": len(pred_boxes),
                    "predictions_operating_threshold": len(operating_indices),
                }
            )

    if totals["tp"] + totals["fn"] != totals["gt"]:
        raise RuntimeError("GT accounting invariant failed")
    if totals["tp"] + totals["fp"] != totals["operating_predictions"]:
        raise RuntimeError("Operating-prediction accounting invariant failed")
    if totals["operating_predictions"] > totals["all_saved_predictions"]:
        raise RuntimeError("Operating predictions exceed saved low-threshold predictions")
    if totals["gt"] != len(metadata):
        raise RuntimeError("Exported GT count differs from metadata row count")

    semantics = prediction_semantics(args.model_family)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Frozen FLIR val prediction and fixed-threshold matching evidence",
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "images": len(images),
        "gt_metadata": str(args.gt_metadata.resolve()),
        "gt_metadata_sha256": sha256(args.gt_metadata),
        "script_sha256": sha256(Path(__file__)),
        "coordinate_space": "native original-image xyxy pixels (Ultralytics Results.boxes.xyxy)",
        "boundary_clipping_applied": args.clip_to_image,
        "boundary_clipping_reason": (
            "Protocol amendment after audited RT-DETR native predictor returned partly out-of-bounds boxes; "
            "clipping aligns boundary semantics with YOLO scale_boxes without changing scores, classes, order, "
            "max_det, confidence thresholds, matching IoU, or NMS-free behavior"
            if args.clip_to_image else None
        ),
        "max_raw_overflow_ltrb_px": {
            key: float(value)
            for key, value in zip(("left", "top", "right", "bottom"), max_raw_overflow_ltrb_px, strict=True)
        },
        "prediction_evidence_semantics": semantics["postprocessing"],
        "model_family": semantics["model_family"],
        "nms_applied": semantics["nms_applied"],
        "nms_iou_effective": semantics["nms_iou_effective"],
        "prediction_conf": args.prediction_conf,
        "operating_conf": args.operating_conf,
        "nms_iou": args.nms_iou,
        "match_iou": args.match_iou,
        "max_det": args.max_det,
        "totals": totals,
        "fixed_threshold_precision": (
            totals["tp"] / (totals["tp"] + totals["fp"])
            if totals["tp"] + totals["fp"] else None
        ),
        "fixed_threshold_recall": totals["tp"] / totals["gt"] if totals["gt"] else None,
    }
    (args.output / "evidence_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    incomplete_marker.unlink()
    (args.output / "_SUCCESS").write_text(
        "Evidence export completed and accounting invariants passed.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
