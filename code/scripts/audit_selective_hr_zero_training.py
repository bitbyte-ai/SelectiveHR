"""Zero-training selective 640/960 inference feasibility audit on frozen FLIR val exports."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))

from frozen_detection_metrics import average_precision, box_iou, match_correct  # noqa: E402


BUDGETS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
RANDOM_SEED = 20260927
RANDOM_REPLICATES = 1000
CONF = 0.25
CLASS_COUNT = 9


def load_predictions(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            records[row["image"]] = row
    return records


def load_gt(export: Path) -> pd.DataFrame:
    frame = pd.read_csv(export / "fixed_threshold_outcomes.csv")
    gt = frame.loc[frame.row_type.eq("gt")].copy()
    gt["area_px2"] = (gt.x2 - gt.x1) * (gt.y2 - gt.y1)
    return gt.sort_values(["image", "object_index"], kind="stable").reset_index(drop=True)


def event_arrays(record: dict, gt: pd.DataFrame) -> dict[str, np.ndarray]:
    boxes = np.asarray(record["boxes"], dtype=float).reshape(-1, 4)
    classes = np.asarray(record["classes"], dtype=int)
    scores = np.asarray(record["confidences"], dtype=float)
    order = np.argsort(-scores, kind="stable")
    boxes, classes, scores = boxes[order], classes[order], scores[order]
    gt_boxes = gt[["x1", "y1", "x2", "y2"]].to_numpy(float)
    gt_classes = gt.class_id.to_numpy(int)
    areas = gt.area_px2.to_numpy(float)
    available = np.ones(len(gt), dtype=bool)
    tp = np.zeros(len(scores), dtype=np.int8)
    lt16 = np.zeros(len(scores), dtype=np.int8)
    lt32 = np.zeros(len(scores), dtype=np.int8)
    medium = np.zeros(len(scores), dtype=np.int8)
    large = np.zeros(len(scores), dtype=np.int8)
    for j in range(len(scores)):
        candidates = np.flatnonzero((gt_classes == classes[j]) & available)
        if not len(candidates):
            continue
        overlaps = box_iou(boxes[j : j + 1], gt_boxes[candidates]).reshape(-1)
        if len(overlaps) and overlaps.max() >= 0.50:
            match = int(candidates[int(overlaps.argmax())])
            available[match] = False
            tp[j] = 1
            lt16[j] = areas[match] < 16**2
            lt32[j] = areas[match] < 32**2
            medium[j] = 32**2 <= areas[match] < 96**2
            large[j] = areas[match] >= 96**2
    return {
        "score": scores,
        "tp": tp,
        "fp": 1 - tp,
        "lt16": lt16,
        "lt32": lt32,
        "medium": medium,
        "large": large,
    }


def prepare_source(predictions: dict[str, dict], gt_groups: dict[str, pd.DataFrame], images: list[str]) -> dict[str, dict]:
    source: dict[str, dict] = {}
    for image in images:
        record = predictions[image]
        gt = gt_groups[image]
        boxes = np.asarray(record["boxes"], dtype=float).reshape(-1, 4)
        classes = np.asarray(record["classes"], dtype=int)
        scores = np.asarray(record["confidences"], dtype=float)
        correct = match_correct(
            boxes,
            classes,
            gt[["x1", "y1", "x2", "y2"]].to_numpy(float),
            gt.class_id.to_numpy(int),
        )
        source[image] = {
            "scores": scores,
            "classes": classes,
            "correct": correct,
            "events": event_arrays(record, gt),
        }
    return source


def evaluate_selection(
    selected: np.ndarray,
    images: list[str],
    low: dict[str, dict],
    high: dict[str, dict],
    target_classes: np.ndarray,
    group_totals: dict[str, int],
    baseline_fp: int,
) -> dict[str, float]:
    chosen = [high[name] if selected[i] else low[name] for i, name in enumerate(images)]
    correct = np.concatenate([x["correct"] for x in chosen])
    scores = np.concatenate([x["scores"] for x in chosen])
    classes = np.concatenate([x["classes"] for x in chosen])
    map5095, _map75, ap = average_precision(correct, scores, classes, target_classes, CLASS_COUNT)

    events = {key: np.concatenate([x["events"][key] for x in chosen]) for key in chosen[0]["events"]}
    fixed = events["score"] >= CONF
    tp_fixed = int(events["tp"][fixed].sum())
    fp_fixed = int(events["fp"][fixed].sum())
    order = np.argsort(-events["score"], kind="stable")
    cumulative_fp = np.cumsum(events["fp"][order])
    point = int(np.abs(cumulative_fp - baseline_fp).argmin())
    prefix = order[: point + 1]
    matched_fp = int(cumulative_fp[point])
    return {
        "ap50_95": map5095,
        "ap50": float(np.nanmean(ap[:, 0])),
        "precision_conf025": tp_fixed / max(tp_fixed + fp_fixed, 1),
        "recall_conf025": tp_fixed / len(target_classes),
        "matched_threshold": float(events["score"][order[point]]),
        "matched_fppi": matched_fp / len(images),
        "matched_lt32_recall": float(events["lt32"][prefix].sum() / group_totals["lt32"]),
        "matched_lt16_recall": float(events["lt16"][prefix].sum() / group_totals["lt16"]),
        "matched_medium_recall": float(events["medium"][prefix].sum() / group_totals["medium"]),
        "matched_large_recall": float(events["large"][prefix].sum() / group_totals["large"]),
    }


def image_features(record: dict, image_path: Path) -> dict[str, float]:
    boxes = np.asarray(record["boxes"], dtype=float).reshape(-1, 4)
    scores = np.asarray(record["confidences"], dtype=float)
    classes = np.asarray(record["classes"], dtype=int)
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    with Image.open(image_path) as im:
        width, height = im.size
    values: dict[str, float] = {
        "total_detection_count": len(scores),
        "count_area_lt8": int((areas < 8**2).sum()),
        "count_area_lt16": int((areas < 16**2).sum()),
        "count_area_lt32": int((areas < 32**2).sum()),
        "min_area": float(areas.min()) if len(areas) else 0.0,
        "median_area": float(np.median(areas)) if len(areas) else 0.0,
        "min_confidence": float(scores.min()) if len(scores) else 0.0,
        "mean_confidence": float(scores.mean()) if len(scores) else 0.0,
        "median_confidence": float(np.median(scores)) if len(scores) else 0.0,
        "lower_quartile_confidence": float(np.quantile(scores, 0.25)) if len(scores) else 0.0,
        "conf_0001_005_count": int(((scores >= 0.001) & (scores < 0.05)).sum()),
        "conf_005_010_count": int(((scores >= 0.05) & (scores < 0.10)).sum()),
        "conf_010_020_count": int(((scores >= 0.10) & (scores < 0.20)).sum()),
        "conf_020_030_count": int(((scores >= 0.20) & (scores < 0.30)).sum()),
        "candidate_density": len(scores) / float(width * height),
    }
    low_small = (areas < 32**2) & (scores < CONF)
    values["low_conf_small_count"] = int(low_small.sum())
    values["low_conf_small_conf_sum"] = float(scores[low_small].sum())
    values["low_conf_small_conf_max"] = float(scores[low_small].max()) if low_small.any() else 0.0
    if len(scores) and scores.sum() > 0:
        probs = scores / scores.sum()
        values["score_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum() / max(math.log(len(probs)), 1.0))
        values["score_concentration"] = float(scores.max() / scores.sum())
    else:
        values["score_entropy"] = values["score_concentration"] = 0.0
    if len(boxes) > 1:
        centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / (2 * width), (boxes[:, 1] + boxes[:, 3]) / (2 * height)))
        tree = cKDTree(centers)
        nearest = tree.query(centers, k=2)[0][:, 1]
        local = tree.query_ball_point(centers, r=0.05)
        values["median_nearest_neighbor_distance"] = float(np.median(nearest))
        values["max_local_candidate_density"] = int(max(len(x) for x in local))
    else:
        values["median_nearest_neighbor_distance"] = 1.0
        values["max_local_candidate_density"] = len(boxes)
    for class_id in range(CLASS_COUNT):
        values[f"class_{class_id}_count"] = int((classes == class_id).sum())
    return values


def benefit_table(low_gt: pd.DataFrame, high_gt: pd.DataFrame, low_export: Path, high_export: Path) -> pd.DataFrame:
    keys = ["image", "object_index"]
    joined = low_gt[keys + ["area_px2", "outcome"]].merge(
        high_gt[keys + ["outcome"]], on=keys, suffixes=("_640", "_960"), validate="one_to_one"
    )
    small = joined.area_px2 < 32**2
    tiny = joined.area_px2 < 16**2
    joined["recovered"] = joined.outcome_640.eq("FN") & joined.outcome_960.eq("TP")
    joined["lost"] = joined.outcome_640.eq("TP") & joined.outcome_960.eq("FN")
    rows = []
    fp_frames = {}
    for label, export in (("640", low_export), ("960", high_export)):
        frame = pd.read_csv(export / "fixed_threshold_outcomes.csv")
        fp_frames[label] = frame.loc[frame.row_type.eq("prediction") & frame.outcome.eq("FP")].groupby("image").size()
    for image, group in joined.groupby("image", sort=True):
        is_small = group.area_px2 < 32**2
        is_tiny = group.area_px2 < 16**2
        recovered_small = int((group.recovered & is_small).sum())
        lost_small = int((group.lost & is_small).sum())
        recovered_tiny = int((group.recovered & is_tiny).sum())
        lost_tiny = int((group.lost & is_tiny).sum())
        fp_change = int(fp_frames["960"].get(image, 0) - fp_frames["640"].get(image, 0))
        net = recovered_small - lost_small
        rows.append({
            "image": image,
            "recovered_lt32": recovered_small,
            "lost_lt32": lost_small,
            "net_lt32_tp_gain": net,
            "recovered_lt16": recovered_tiny,
            "lost_lt16": lost_tiny,
            "net_lt16_tp_gain": recovered_tiny - lost_tiny,
            "fp_change_conf025": fp_change,
            "risk_aware_benefit": net - 0.25 * fp_change,
            "positive_recoverability": int(net > 0),
            "positive_lt16_recoverability": int(recovered_tiny - lost_tiny > 0),
        })
    return pd.DataFrame(rows)


def stable_top(frame: pd.DataFrame, score: str, count: int, extra: list[str] | None = None) -> np.ndarray:
    columns = [score, *(extra or []), "image"]
    ascending = [False] * (len(columns) - 1) + [True]
    names = set(frame.sort_values(columns, ascending=ascending, kind="stable").head(count).image)
    return frame.image.isin(names).to_numpy()


def main() -> None:
    # These metrics are needed only by the standalone development audit.  Keep
    # the optional sklearn dependency out of the reusable frozen-transfer
    # helpers imported by transfer_selective_hr_s4.py and bootstrap scripts.
    from sklearn.metrics import average_precision_score, roc_auc_score

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-replicates", type=int, default=RANDOM_REPLICATES)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for frozen_output in (
        "oracle_budget_curve.csv", "random_budget_baseline.csv", "recoverability_feature_audit.csv",
        "heuristic_selector_results.csv", "feasibility_summary.json",
    ):
        if (args.output / frozen_output).exists():
            raise FileExistsError(args.output / frozen_output)

    for directory in (args.low, args.high):
        manifest = json.loads((directory / "evidence_manifest.json").read_text(encoding="utf-8"))
        required = {"images": 1127, "prediction_conf": 0.001, "operating_conf": 0.25, "nms_iou": 0.7, "match_iou": 0.5, "max_det": 300}
        for key, expected in required.items():
            if manifest[key] != expected:
                raise ValueError(f"{directory}: {key} mismatch")
    low_manifest = json.loads((args.low / "evidence_manifest.json").read_text(encoding="utf-8"))
    high_manifest = json.loads((args.high / "evidence_manifest.json").read_text(encoding="utf-8"))
    if low_manifest["model_sha256"] != high_manifest["model_sha256"]:
        raise ValueError("640 and 960 checkpoint hashes differ")

    low_gt, high_gt = load_gt(args.low), load_gt(args.high)
    identity = ["image", "object_index", "class_id", "x1", "y1", "x2", "y2"]
    if not low_gt[identity].equals(high_gt[identity]):
        raise ValueError("GT identity differs across resolutions")
    images = sorted(low_gt.image.unique())
    low_pred = load_predictions(args.low / "predictions_all.jsonl")
    high_pred = load_predictions(args.high / "predictions_all.jsonl")
    if set(images) != set(low_pred) or set(images) != set(high_pred):
        raise ValueError("Prediction image inventory mismatch")
    gt_groups = {name: frame.reset_index(drop=True) for name, frame in low_gt.groupby("image", sort=True)}
    low = prepare_source(low_pred, gt_groups, images)
    high = prepare_source(high_pred, gt_groups, images)
    target_classes = low_gt.class_id.to_numpy(int)
    group_totals = {
        "lt16": int((low_gt.area_px2 < 16**2).sum()),
        "lt32": int((low_gt.area_px2 < 32**2).sum()),
        "medium": int(((low_gt.area_px2 >= 32**2) & (low_gt.area_px2 < 96**2)).sum()),
        "large": int((low_gt.area_px2 >= 96**2).sum()),
    }
    baseline_fp = int(low_manifest["totals"]["fp"])
    none = np.zeros(len(images), dtype=bool)
    all_high = np.ones(len(images), dtype=bool)
    baseline = evaluate_selection(none, images, low, high, target_classes, group_totals, baseline_fp)
    full960 = evaluate_selection(all_high, images, low, high, target_classes, group_totals, baseline_fp)

    benefits = benefit_table(low_gt, high_gt, args.low, args.high)
    feature_rows = []
    for image in images:
        feature_rows.append({"image": image, **image_features(low_pred[image], args.images / image)})
    features = pd.DataFrame(feature_rows)
    audit = features.merge(benefits, on="image", validate="one_to_one")
    feature_names = [c for c in features.columns if c != "image"]
    assoc_rows = []
    y = audit.positive_recoverability.to_numpy(int)
    y16 = audit.positive_lt16_recoverability.to_numpy(int)
    for feature in feature_names:
        x = audit[feature].to_numpy(float)
        rho, pvalue = spearmanr(x, audit.risk_aware_benefit.to_numpy(float))
        assoc_rows.append({
            "feature": feature,
            "spearman_risk_benefit": rho,
            "spearman_p": pvalue,
            "auroc_positive_lt32": roc_auc_score(y, x) if len(np.unique(y)) == 2 else np.nan,
            "prauc_positive_lt32": average_precision_score(y, x) if y.sum() else np.nan,
            "auroc_positive_lt16": roc_auc_score(y16, x) if len(np.unique(y16)) == 2 else np.nan,
            "prauc_positive_lt16": average_precision_score(y16, x) if y16.sum() else np.nan,
        })
    association = pd.DataFrame(assoc_rows).sort_values("auroc_positive_lt32", ascending=False)
    association.to_csv(args.output / "recoverability_feature_audit.csv", index=False)

    for name in ("low_conf_small_count", "count_area_lt16", "candidate_density"):
        std = audit[name].std(ddof=0)
        audit[f"z_{name}"] = (audit[name] - audit[name].mean()) / (std if std else 1.0)
    audit["equal_z_composite"] = audit[["z_low_conf_small_count", "z_count_area_lt16", "z_candidate_density"]].sum(axis=1)
    audit["negative_fp_change_conf025"] = -audit["fp_change_conf025"]
    audit.to_csv(args.output / "image_recoverability_and_features.csv", index=False)

    oracle_rows, selector_rows = [], []
    exact_cache: dict[tuple[str, float], dict] = {}
    for budget in BUDGETS:
        count = int(round(len(images) * budget))
        selected = stable_top(
            audit, "risk_aware_benefit", count,
            ["net_lt32_tp_gain", "recovered_lt16", "negative_fp_change_conf025"],
        )
        result = evaluate_selection(selected, images, low, high, target_classes, group_totals, baseline_fp)
        oracle_rows.append({"selector": "ORACLE", "budget": budget, "selected_images": count, "cost_proxy": 1 + 2.25 * budget, **result})
        for label, column in {
            "S1_low_conf_small_count": "low_conf_small_count",
            "S2_ultra_small_count": "count_area_lt16",
            "S3_candidate_density": "candidate_density",
            "S4_equal_z_composite": "equal_z_composite",
        }.items():
            mask = stable_top(audit, column, count)
            value = evaluate_selection(mask, images, low, high, target_classes, group_totals, baseline_fp)
            exact_cache[(label, budget)] = value
            selector_rows.append({"selector": label, "budget": budget, "selected_images": count, "cost_proxy": 1 + 2.25 * budget, **value})
    oracle = pd.DataFrame(oracle_rows)
    selectors = pd.DataFrame(selector_rows)
    oracle.to_csv(args.output / "oracle_budget_curve.csv", index=False)

    rng = np.random.default_rng(RANDOM_SEED)
    permutations = [rng.permutation(len(images)) for _ in range(args.random_replicates)]
    random_rows = []
    for budget in BUDGETS:
        count = int(round(len(images) * budget))
        selections = []
        for permutation in permutations:
            mask = np.zeros(len(images), dtype=bool)
            mask[permutation[:count]] = True
            selections.append(mask)
        def run(mask: np.ndarray) -> tuple[float, float, float]:
            result = evaluate_selection(mask, images, low, high, target_classes, group_totals, baseline_fp)
            return result["ap50_95"], result["matched_lt32_recall"], result["matched_lt16_recall"]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            values = np.asarray(list(pool.map(run, selections)), dtype=float)
        random_rows.append({
            "budget": budget,
            "selected_images": count,
            "replicates": args.random_replicates,
            "seed": RANDOM_SEED,
            "ap50_95_mean": values[:, 0].mean(),
            "ap50_95_p025": np.quantile(values[:, 0], 0.025),
            "ap50_95_p975": np.quantile(values[:, 0], 0.975),
            "matched_lt32_recall_mean": values[:, 1].mean(),
            "matched_lt32_recall_p025": np.quantile(values[:, 1], 0.025),
            "matched_lt32_recall_p975": np.quantile(values[:, 1], 0.975),
            "matched_lt16_recall_mean": values[:, 2].mean(),
            "matched_lt16_recall_p025": np.quantile(values[:, 2], 0.025),
            "matched_lt16_recall_p975": np.quantile(values[:, 2], 0.975),
        })
        print(f"random budget {budget:.0%} complete", flush=True)
    random_frame = pd.DataFrame(random_rows)
    random_frame.to_csv(args.output / "random_budget_baseline.csv", index=False)

    selectors = selectors.merge(random_frame, on=["budget", "selected_images"], how="left")
    for metric in ("ap50_95", "matched_lt32_recall", "matched_lt16_recall", "matched_medium_recall", "matched_large_recall"):
        selectors[f"{metric}_delta_pp"] = 100 * (selectors[metric] - baseline[metric])
    selectors["ap_gain_efficiency"] = (selectors.ap50_95 - baseline["ap50_95"]) / (full960["ap50_95"] - baseline["ap50_95"])
    selectors["small_gain_efficiency"] = (selectors.matched_lt32_recall - baseline["matched_lt32_recall"]) / (full960["matched_lt32_recall"] - baseline["matched_lt32_recall"])
    selectors["beats_random_ap_p975"] = selectors.ap50_95 > selectors.ap50_95_p975
    selectors["beats_random_small_p975"] = selectors.matched_lt32_recall > selectors.matched_lt32_recall_p975
    selectors.to_csv(args.output / "heuristic_selector_results.csv", index=False)

    eligible = selectors[selectors.budget <= 0.20].copy()
    eligible["primary_gate"] = (
        (eligible.ap50_95_delta_pp >= 1.0)
        & (eligible.matched_lt32_recall_delta_pp >= 2.0)
        & (eligible.matched_medium_recall_delta_pp >= -0.2)
        & (eligible.matched_large_recall_delta_pp >= -0.2)
        & eligible.beats_random_ap_p975
        & eligible.beats_random_small_p975
    )
    eligible["secondary_gate"] = (
        (eligible.ap50_95_delta_pp >= 0.5)
        & (eligible.matched_lt32_recall_delta_pp >= 1.0)
        & eligible.beats_random_ap_p975
        & eligible.beats_random_small_p975
    )
    decision = "GO" if eligible.primary_gate.any() else "PROMISING" if eligible.secondary_gate.any() else "NO-GO"
    best = eligible.sort_values(["primary_gate", "secondary_gate", "ap50_95_delta_pp", "matched_lt32_recall_delta_pp"], ascending=False).iloc[0]
    best_feature = association.iloc[0]
    summary = {
        "metric_scope": "saved-post-NMS export evaluator; paired deltas authoritative",
        "baseline_640": baseline,
        "full_960": full960,
        "full_960_delta": {key: full960[key] - baseline[key] for key in full960},
        "best_selector": best.to_dict(),
        "best_single_feature": best_feature.to_dict(),
        "decision": decision,
        "new_detector_training_performed": False,
        "flir_test_accessed": False,
    }
    (args.output / "feasibility_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
