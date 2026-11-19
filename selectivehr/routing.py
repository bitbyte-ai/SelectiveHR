"""Portable extraction of the paper's frozen S4 rule.

Only first-pass predictions and native image dimensions enter selection.
Evaluation labels and higher-resolution predictions are not accepted here.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

FEATURES = ("low_conf_small_count", "count_area_lt16", "candidate_density")


def features(record: Mapping) -> dict[str, float]:
    """Use native-pixel xyxy post-NMS boxes; confidence floor must be 0.001."""
    width, height = float(record["width"]), float(record["height"])
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Native image dimensions must be positive and finite")
    boxes = np.asarray(record["boxes"], dtype=float)
    if boxes.size == 0:
        boxes = np.empty((0, 4), dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    confidence = np.asarray(record["confidences"], dtype=float)
    if confidence.ndim != 1 or len(confidence) != len(boxes):
        raise ValueError("Each box must have one confidence")
    if not np.isfinite(boxes).all() or not np.isfinite(confidence).all():
        raise ValueError("Predictions must be finite")
    if np.any((confidence < 0.001) | (confidence > 1)):
        raise ValueError("Expected retained predictions with confidence in [0.001, 1]")
    if len(boxes) > 300:
        raise ValueError("Paper protocol retains at most 300 post-NMS boxes")
    # The frozen implementation clamps negative side lengths to zero.
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return {
        "low_conf_small_count": int(((areas < 32**2) & (confidence < .25)).sum()),
        "count_area_lt16": int((areas < 16**2).sum()),
        "candidate_density": len(confidence) / (width * height),
    }


def score_features(values: Sequence[Mapping], normalization: Mapping) -> np.ndarray:
    """Source mean and population SD stay frozen for every target batch."""
    scores = np.zeros(len(values), dtype=float)
    for name in FEATURES:
        mean = float(normalization[name]["mean"])
        std = float(normalization[name]["std_ddof0"])
        if not math.isfinite(mean) or not math.isfinite(std) or std < 0:
            raise ValueError("Normalization must be finite with nonnegative SD")
        column = np.asarray([row[name] for row in values], dtype=float)
        if not np.isfinite(column).all():
            raise ValueError("Features must be finite")
        scores += (column - mean) / (std or 1.0)
    return scores


def select_batch(records: Sequence[Mapping], normalization: Mapping, budget: float = .20) -> dict:
    """Select round(budget * N) images, breaking score ties by image ID.

    Uses Python round (ties to even), exactly as the frozen experiment scripts.
    Small batches may select zero images. Input order does not break score ties.
    """
    if not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("budget must lie in [0, 1]")
    names = [record["image"] for record in records]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Every image needs a nonempty string ID")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate image IDs are not allowed")
    scores = score_features([features(record) for record in records], normalization)
    order = sorted(range(len(records)), key=lambda i: (-scores[i], names[i]))
    count = int(round(budget * len(records)))
    return {"budget": budget, "images": len(records), "selected_images": count,
            "selected": [names[i] for i in order[:count]],
            "scores": {name: float(score) for name, score in zip(names, scores)}}


def reroute(low_records: Sequence[Mapping], normalization: Mapping,
            predict_high: Callable[[str], Mapping], budget: float = .20) -> tuple[list[Mapping], dict]:
    """Call the high-resolution predictor only after selection; replace outputs."""
    selection = select_batch(low_records, normalization, budget)
    selected = set(selection["selected"])
    outputs = []
    for record in low_records:
        if record["image"] in selected:
            high = predict_high(record["image"])
            if high.get("image") != record["image"]:
                raise ValueError("High-resolution output has the wrong image ID")
            outputs.append(high)
        else:
            outputs.append(record)
    return outputs, selection
