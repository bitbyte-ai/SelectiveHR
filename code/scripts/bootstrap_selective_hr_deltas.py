#!/usr/bin/env python3
"""Paired image-bootstrap intervals for frozen SelectiveHR deltas."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))

from frozen_detection_metrics import average_precision  # noqa: E402
from audit_selective_hr_zero_training import load_gt, load_predictions, prepare_source  # noqa: E402


def matched_recalls(events: dict[str, np.ndarray], fp_budget: int, totals: dict[str, int]) -> dict[str, float]:
    order = np.argsort(-events["score"], kind="stable")
    cumulative_fp = np.cumsum(events["fp"][order])
    point = int(np.abs(cumulative_fp - fp_budget).argmin())
    prefix = order[: point + 1]
    return {
        name: float(events[name][prefix].sum() / max(totals[name], 1))
        for name in ("lt16", "lt32", "medium", "large")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--budget", type=float, default=0.20)
    parser.add_argument("--score-column", default="equal_z_composite")
    args = parser.parse_args()

    low_gt, high_gt = load_gt(args.low), load_gt(args.high)
    identity = ["image", "object_index", "class_id", "x1", "y1", "x2", "y2"]
    if not low_gt[identity].equals(high_gt[identity]):
        raise RuntimeError("640/960 GT mismatch")
    images = sorted(low_gt.image.unique())
    groups = {name: frame.reset_index(drop=True) for name, frame in low_gt.groupby("image", sort=True)}
    low = prepare_source(load_predictions(args.low / "predictions_all.jsonl"), groups, images)
    high = prepare_source(load_predictions(args.high / "predictions_all.jsonl"), groups, images)

    selection = pd.read_csv(args.selection)
    if set(selection.image) != set(images):
        raise RuntimeError("Selection/image population mismatch")
    if "selected_s4_20pct" in selection:
        selected = set(selection.loc[selection.selected_s4_20pct.astype(bool), "image"])
    else:
        count = int(round(len(images) * args.budget))
        selected = set(
            selection.sort_values([args.score_column, "image"], ascending=[False, True], kind="stable")
            .head(count)
            .image
        )
    if len(selected) != int(round(len(images) * args.budget)):
        raise RuntimeError("Frozen selected-image count mismatch")

    gt_classes_by_image = {name: groups[name].class_id.to_numpy(int) for name in images}
    gt_areas_by_image = {name: groups[name].area_px2.to_numpy(float) for name in images}
    class_count = int(low_gt.class_id.max()) + 1

    rng = np.random.default_rng(args.seed)
    samples = [rng.integers(0, len(images), size=len(images)) for _ in range(args.replicates)]

    def evaluate(sample: np.ndarray) -> dict[str, float]:
        names = [images[int(index)] for index in sample]
        target_classes = np.concatenate([gt_classes_by_image[name] for name in names])
        areas = np.concatenate([gt_areas_by_image[name] for name in names])
        totals = {
            "lt16": int((areas < 16**2).sum()),
            "lt32": int((areas < 32**2).sum()),
            "medium": int(((areas >= 32**2) & (areas < 96**2)).sum()),
            "large": int((areas >= 96**2).sum()),
        }
        low_chosen = [low[name] for name in names]
        hybrid_chosen = [high[name] if name in selected else low[name] for name in names]

        def arrays(chosen: list[dict]) -> tuple[float, dict[str, np.ndarray]]:
            correct = np.concatenate([item["correct"] for item in chosen])
            scores = np.concatenate([item["scores"] for item in chosen])
            classes = np.concatenate([item["classes"] for item in chosen])
            ap, _, _ = average_precision(correct, scores, classes, target_classes, class_count)
            events = {
                key: np.concatenate([item["events"][key] for item in chosen])
                for key in chosen[0]["events"]
            }
            return float(ap), events

        low_ap, low_events = arrays(low_chosen)
        hybrid_ap, hybrid_events = arrays(hybrid_chosen)
        fp_budget = int(low_events["fp"][low_events["score"] >= 0.25].sum())
        low_recall = matched_recalls(low_events, fp_budget, totals)
        hybrid_recall = matched_recalls(hybrid_events, fp_budget, totals)
        return {
            "ap50_95_delta_pp": 100 * (hybrid_ap - low_ap),
            **{
                f"matched_{name}_recall_delta_pp": 100 * (hybrid_recall[name] - low_recall[name])
                for name in low_recall
            },
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, samples))
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    summary = {
        "schema": "selectivehr-paired-image-bootstrap-v1",
        "replicates": args.replicates,
        "seed": args.seed,
        "images": len(images),
        "selected_images": len(selected),
        "sampling_unit": "image with replacement",
        "interval": "percentile 95%",
        "metrics": {
            column: {
                "mean": float(frame[column].mean()),
                "lower": float(frame[column].quantile(0.025)),
                "upper": float(frame[column].quantile(0.975)),
            }
            for column in frame
        },
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
