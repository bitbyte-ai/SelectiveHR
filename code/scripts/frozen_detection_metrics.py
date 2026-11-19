"""Frozen saved-output evaluator used by the paper; not pycocotools."""
from __future__ import annotations
import numpy as np

IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)

def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Pairwise IoU for xyxy arrays."""
    if not len(boxes1) or not len(boxes2):
        return np.zeros((len(boxes1), len(boxes2)), dtype=float)
    tl = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    br = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = np.maximum(br - tl, 0.0).prod(axis=2)
    area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0.0) * np.maximum(boxes1[:, 3] - boxes1[:, 1], 0.0)
    area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0.0) * np.maximum(boxes2[:, 3] - boxes2[:, 1], 0.0)
    return inter / np.maximum(area1[:, None] + area2[None, :] - inter, 1e-12)

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Ultralytics/COCO-style 101-point interpolated AP."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    # NumPy introduced ``trapezoid`` in 2.0; the validated cloud environment
    # uses NumPy 1.x, where the numerically equivalent API is ``trapz``.
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(np.interp(x, mrec, mpre), x=x))

def match_correct(
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
) -> np.ndarray:
    """Reproduce Ultralytics greedy matching at 0.50:0.95 IoU."""
    correct = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(pred_boxes) or not len(gt_boxes):
        return correct
    ious = box_iou(gt_boxes, pred_boxes)
    ious *= gt_classes[:, None] == pred_classes[None, :]
    for column, threshold in enumerate(IOU_THRESHOLDS):
        label_idx, pred_idx = np.nonzero(ious >= threshold)
        if not len(label_idx):
            continue
        matches = np.column_stack((label_idx, pred_idx, ious[label_idx, pred_idx]))
        matches = matches[np.argsort(-matches[:, 2], kind="stable")]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), column] = True
    return correct

def average_precision(
    correct: np.ndarray,
    scores: np.ndarray,
    pred_classes: np.ndarray,
    target_classes: np.ndarray,
    class_count: int = 9,
) -> tuple[float, float, np.ndarray]:
    """Return mAP50-95, mAP75, and per-class AP matrix."""
    order = np.argsort(-scores, kind="stable")
    correct, pred_classes = correct[order], pred_classes[order]
    ap = np.full((class_count, len(IOU_THRESHOLDS)), np.nan, dtype=float)
    for class_id in range(class_count):
        n_labels = int(np.sum(target_classes == class_id))
        mask = pred_classes == class_id
        if not n_labels or not np.any(mask):
            continue
        tp = correct[mask].astype(float).cumsum(axis=0)
        fp = (~correct[mask]).astype(float).cumsum(axis=0)
        recall = tp / (n_labels + 1e-16)
        precision = tp / (tp + fp + 1e-16)
        for index in range(len(IOU_THRESHOLDS)):
            ap[class_id, index] = compute_ap(recall[:, index], precision[:, index])
    return float(np.nanmean(ap)), float(np.nanmean(ap[:, 5])), ap
