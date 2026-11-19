"""Frozen transfer of the YOLO11n SelectiveHR S4 router to another setting."""

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

from audit_selective_hr_zero_training import (  # noqa: E402
    evaluate_selection,
    image_features,
    load_gt,
    load_predictions,
    prepare_source,
)

FEATURES = ("low_conf_small_count", "count_area_lt16", "candidate_density")


def source_normalization(source: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        name: {"mean": float(source[name].mean()), "std_ddof0": float(source[name].std(ddof=0))}
        for name in FEATURES
    }


def fixed_s4_scores(target: pd.DataFrame, normalization: dict[str, dict[str, float]]) -> np.ndarray:
    score = np.zeros(len(target), dtype=float)
    for name in FEATURES:
        scale = normalization[name]["std_ddof0"] or 1.0
        score += (target[name].to_numpy(float) - normalization[name]["mean"]) / scale
    return score


def delta_row(strategy: str, metrics: dict, baseline: dict, cost: float, budget: float) -> dict:
    row = {"strategy": strategy, "budget": budget, "cost_proxy": cost, **metrics}
    for metric in ("ap50_95", "matched_lt32_recall", "matched_lt16_recall", "matched_medium_recall", "matched_large_recall"):
        row[f"{metric}_delta_pp"] = 100 * (metrics[metric] - baseline[metric])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    normalization_inputs = parser.add_mutually_exclusive_group()
    normalization_inputs.add_argument("--source-features", type=Path)
    normalization_inputs.add_argument("--normalization", type=Path, default=Path("configs/source_normalization.json"))
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="78_yolov8n")
    parser.add_argument("--target-model", default="YOLOv8n")
    parser.add_argument(
        "--expected-images",
        type=int,
        default=None,
        help="Optional frozen image-count invariant (for example 1127 for FLIR or 840 for M3FD).",
    )
    parser.add_argument("--random-replicates", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    low_manifest = json.loads((args.low / "evidence_manifest.json").read_text(encoding="utf-8"))
    high_manifest = json.loads((args.high / "evidence_manifest.json").read_text(encoding="utf-8"))
    expected_contract = {
        "prediction_conf": 0.001,
        "operating_conf": 0.25,
        "nms_iou": 0.7,
        "match_iou": 0.5,
        "max_det": 300,
    }
    for manifest in (low_manifest, high_manifest):
        for key, expected in expected_contract.items():
            if manifest[key] != expected:
                raise RuntimeError(f"manifest mismatch: {key}={manifest[key]} expected {expected}")
    if low_manifest["images"] != high_manifest["images"]:
        raise RuntimeError("640/960 image-count mismatch")
    if args.expected_images is not None and low_manifest["images"] != args.expected_images:
        raise RuntimeError(
            f"manifest mismatch: images={low_manifest['images']} expected {args.expected_images}"
        )
    if low_manifest["model_sha256"] != high_manifest["model_sha256"]:
        raise RuntimeError("YOLOv8n 640/960 checkpoint mismatch")

    low_gt, high_gt = load_gt(args.low), load_gt(args.high)
    identity = ["image", "object_index", "class_id", "x1", "y1", "x2", "y2"]
    if not low_gt[identity].equals(high_gt[identity]):
        raise RuntimeError("YOLOv8n 640/960 GT mismatch")
    images = sorted(low_gt.image.unique())
    low_pred = load_predictions(args.low / "predictions_all.jsonl")
    high_pred = load_predictions(args.high / "predictions_all.jsonl")
    gt_groups = {name: frame.reset_index(drop=True) for name, frame in low_gt.groupby("image", sort=True)}
    low_source = prepare_source(low_pred, gt_groups, images)
    high_source = prepare_source(high_pred, gt_groups, images)
    target_classes = low_gt.class_id.to_numpy(int)
    areas = low_gt.area_px2.to_numpy(float)
    group_totals = {
        "lt16": int((areas < 16**2).sum()),
        "lt32": int((areas < 32**2).sum()),
        "medium": int(((areas >= 32**2) & (areas < 96**2)).sum()),
        "large": int((areas >= 96**2).sum()),
    }
    baseline_fp = int(low_manifest["totals"]["fp"])

    normalization = (source_normalization(pd.read_csv(args.source_features)) if args.source_features
                     else json.loads(args.normalization.read_text(encoding="utf-8")))
    target_rows = [{"image": image, **image_features(low_pred[image], args.images / image)} for image in images]
    target = pd.DataFrame(target_rows)
    target["s4_frozen_source_normalized"] = fixed_s4_scores(target, normalization)
    count = int(round(len(images) * 0.20))
    selected = set(
        target.sort_values(["s4_frozen_source_normalized", "image"], ascending=[False, True], kind="stable")
        .head(count).image
    )
    selected_mask = np.asarray([image in selected for image in images], dtype=bool)
    target["selected_s4_20pct"] = target.image.isin(selected).astype(int)
    target.to_csv(args.output / f"{args.tag}_frozen_s4_image_scores.csv", index=False)

    none = np.zeros(len(images), dtype=bool)
    all_high = np.ones(len(images), dtype=bool)
    baseline = evaluate_selection(none, images, low_source, high_source, target_classes, group_totals, baseline_fp)
    s4 = evaluate_selection(selected_mask, images, low_source, high_source, target_classes, group_totals, baseline_fp)
    full960 = evaluate_selection(all_high, images, low_source, high_source, target_classes, group_totals, baseline_fp)

    rng = np.random.default_rng(20260927)
    permutations = [rng.permutation(len(images)) for _ in range(args.random_replicates)]

    def run_random(permutation: np.ndarray) -> dict[str, float]:
        mask = np.zeros(len(images), dtype=bool)
        mask[permutation[:count]] = True
        value = evaluate_selection(mask, images, low_source, high_source, target_classes, group_totals, baseline_fp)
        return {
            "ap50_95": value["ap50_95"],
            "matched_lt32_recall": value["matched_lt32_recall"],
            "matched_lt16_recall": value["matched_lt16_recall"],
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        random_rows = list(pool.map(run_random, permutations))
    random = pd.DataFrame(random_rows)
    random.to_csv(args.output / f"{args.tag}_random_20pct_replicates.csv", index=False)
    random_summary = {
        metric: {
            "mean": float(random[metric].mean()),
            "p025": float(random[metric].quantile(0.025)),
            "p975": float(random[metric].quantile(0.975)),
        }
        for metric in random.columns
    }
    random_metrics = {
        **baseline,
        "ap50_95": random_summary["ap50_95"]["mean"],
        "matched_lt32_recall": random_summary["matched_lt32_recall"]["mean"],
        "matched_lt16_recall": random_summary["matched_lt16_recall"]["mean"],
    }
    table = pd.DataFrame([
        delta_row("full_640", baseline, baseline, 1.0, 0.0),
        delta_row("random_20pct_mean", random_metrics, baseline, 1.45, 0.20),
        delta_row("frozen_S4_20pct_replace", s4, baseline, 1.45, 0.20),
        delta_row("full_960", full960, baseline, 3.25, 1.0),
    ])
    table.to_csv(args.output / f"{args.tag}_frozen_transfer_metrics.csv", index=False)

    gates = {
        "ap_delta_ge_0_5pp": 100 * (s4["ap50_95"] - baseline["ap50_95"]) >= 0.5,
        "small_delta_ge_1pp": 100 * (s4["matched_lt32_recall"] - baseline["matched_lt32_recall"]) >= 1.0,
        "medium_delta_ge_minus_1pp": 100 * (s4["matched_medium_recall"] - baseline["matched_medium_recall"]) >= -1.0,
        "large_delta_ge_minus_1pp": 100 * (s4["matched_large_recall"] - baseline["matched_large_recall"]) >= -1.0,
        "ap_above_random_mean": s4["ap50_95"] > random_summary["ap50_95"]["mean"],
        "small_above_random_mean": s4["matched_lt32_recall"] > random_summary["matched_lt32_recall"]["mean"],
    }
    base_intervention_positive = (
        full960["ap50_95"] > baseline["ap50_95"]
        and full960["matched_lt32_recall"] > baseline["matched_lt32_recall"]
    )
    selective_direction_positive = (
        s4["ap50_95"] > baseline["ap50_95"]
        and s4["matched_lt32_recall"] > baseline["matched_lt32_recall"]
    )
    if all(gates.values()):
        paper_interpretation = "TRANSFER_EFFICACY"
    elif not base_intervention_positive and selective_direction_positive:
        paper_interpretation = "SELECTIVE_RECOVERY_FROM_NEGATIVE_AVERAGE_INTERVENTION"
    elif not base_intervention_positive:
        paper_interpretation = "ARCHITECTURE_BOUNDARY_BASE_INTERVENTION_NEGATIVE"
    else:
        paper_interpretation = "SELECTOR_TRANSFER_LIMITATION"
    result = {
        "schema": "selectivehr-frozen-cross-model-transfer-v1",
        "source_model": "YOLO11n",
        "target_model": args.target_model,
        "normalization": normalization,
        "formula": "sum source-normalized z for low_conf_small_count, count_area_lt16, candidate_density",
        "budget": 0.20,
        "images": len(images),
        "selected_images": count,
        "routing": "selected=960 replacement; unselected=640",
        "random_seed": 20260927,
        "random_replicates": args.random_replicates,
        "metrics": {"baseline": baseline, "s4": s4, "full960": full960, "random": random_summary},
        "gates": {key: bool(value) for key, value in gates.items()},
        "decision": "TRANSFER_PASS" if all(gates.values()) else "TRANSFER_PARTIAL" if gates["ap_delta_ge_0_5pp"] and gates["small_delta_ge_1pp"] else "TRANSFER_FAIL",
        "base_intervention_positive": bool(base_intervention_positive),
        "selective_direction_positive": bool(selective_direction_positive),
        "paper_interpretation": paper_interpretation,
        "detector_training_performed": False,
        "target_labels_used_for_selection": False,
        "test_accessed": False,
    }
    (args.output / f"{args.tag}_frozen_transfer_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
