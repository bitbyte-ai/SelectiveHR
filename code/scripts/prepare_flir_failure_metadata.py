"""Prepare model-independent FLIR GT size and local-contrast metadata.

The proxy is frozen in 20_FLIR_Failure_Analysis_Protocol.md. This script reads
only processed train/val images and labels; it never reads model predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


NAMES = ("person", "bike", "car", "motor", "bus", "truck", "light", "hydrant", "sign")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_unique_labels(path: Path) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """Read YOLO rows using Ultralytics-equivalent exact float32-row deduplication.

    The rule is generic across train/val/test.  It removes only exactly equal
    ``(class_id, x, y, w, h)`` rows after numeric float32 parsing, mirroring
    ``ultralytics.data.utils.verify_image_label``.  It never applies an epsilon,
    IoU, class-specific, image-specific, or split-specific merge.
    """
    seen: set[tuple[float, float, float, float, float]] = set()
    rows: list[tuple[int, float, float, float, float]] = []
    duplicates = 0
    for line_index, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Expected five YOLO detection fields: {path}:{line_index}")
        values = np.asarray(fields, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite YOLO label row: {path}:{line_index}")
        if not float(values[0]).is_integer():
            raise ValueError(f"Non-integer YOLO class id: {path}:{line_index}")
        key = tuple(float(value) for value in values.tolist())
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        rows.append((int(values[0]), float(values[1]), float(values[2]),
                     float(values[3]), float(values[4])))
    return rows, duplicates


def pixel_box(row: tuple[int, float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    _, xc, yc, bw, bh = row
    x1 = max(0, min(width - 1, int(np.floor((xc - bw / 2) * width))))
    y1 = max(0, min(height - 1, int(np.floor((yc - bh / 2) * height))))
    x2 = max(x1 + 1, min(width, int(np.ceil((xc + bw / 2) * width))))
    y2 = max(y1 + 1, min(height, int(np.ceil((yc + bh / 2) * height))))
    return x1, y1, x2, y2


def expanded_box(box: tuple[int, int, int, int], factor: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    out_w, out_h = (x2 - x1) * factor, (y2 - y1) * factor
    return (
        max(0, int(np.floor(cx - out_w / 2))),
        max(0, int(np.floor(cy - out_h / 2))),
        min(width, int(np.ceil(cx + out_w / 2))),
        min(height, int(np.ceil(cy + out_h / 2))),
    )


def contrast_for_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    occupied: np.ndarray,
) -> dict[str, object]:
    height, width = image.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    ix, iy = int(np.floor(0.1 * bw)), int(np.floor(0.1 * bh))
    core = (x1 + ix, y1 + iy, x2 - ix, y2 - iy)
    if core[2] - core[0] < 3 or core[3] - core[1] < 3:
        core = box
    cx1, cy1, cx2, cy2 = core
    object_pixels = image[cy1:cy2, cx1:cx2].reshape(-1)
    required_background = max(32, int(object_pixels.size))

    background_pixels: np.ndarray | None = None
    mode = "unavailable"
    for factor in (2.0, 3.0):
        ox1, oy1, ox2, oy2 = expanded_box(box, factor, width, height)
        crop = image[oy1:oy2, ox1:ox2]
        available = ~occupied[oy1:oy2, ox1:ox2]
        candidate = crop[available]
        if candidate.size >= required_background:
            background_pixels = candidate
            mode = f"local_{factor:.0f}x"
            break
    if background_pixels is None:
        candidate = image[~occupied]
        if candidate.size:
            background_pixels = candidate
            mode = "global_outside_gt_fallback"

    if object_pixels.size == 0 or background_pixels is None or background_pixels.size == 0:
        return {
            "ltc": np.nan,
            "absolute_median_contrast": np.nan,
            "object_median": np.nan,
            "background_median": np.nan,
            "background_robust_scale": np.nan,
            "object_pixels": int(object_pixels.size),
            "background_pixels": 0 if background_pixels is None else int(background_pixels.size),
            "background_mode": mode,
            "tiny_proxy_flag": bool(object_pixels.size < 9),
        }

    object_median = float(np.median(object_pixels))
    background_median = float(np.median(background_pixels))
    robust_scale = float(1.4826 * np.median(np.abs(background_pixels - background_median)))
    absolute_contrast = abs(object_median - background_median)
    return {
        "ltc": float(absolute_contrast / (robust_scale + 1 / 255)),
        "absolute_median_contrast": float(absolute_contrast),
        "object_median": object_median,
        "background_median": background_median,
        "background_robust_scale": robust_scale,
        "object_pixels": int(object_pixels.size),
        "background_pixels": int(background_pixels.size),
        "background_mode": mode,
        "tiny_proxy_flag": bool(object_pixels.size < 9),
    }


def process_split(root: Path, split: str) -> tuple[pd.DataFrame, dict[str, int]]:
    records: list[dict[str, object]] = []
    duplicate_rows = 0
    images = sorted((root / "images" / split).glob("*.jpg"))
    for index, image_path in enumerate(images, start=1):
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
        height, width = gray.shape
        labels, duplicates = read_unique_labels(root / "labels" / split / f"{image_path.stem}.txt")
        duplicate_rows += duplicates
        boxes = [pixel_box(row, width, height) for row in labels]
        occupied = np.zeros((height, width), dtype=bool)
        for x1, y1, x2, y2 in boxes:
            occupied[y1:y2, x1:x2] = True
        for object_index, (row, box) in enumerate(zip(labels, boxes, strict=True)):
            cls = row[0]
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            size_bin = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
            records.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "object_index": object_index,
                    "class_id": cls,
                    "class_name": NAMES[cls],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "area_px2": area,
                    "size_bin": size_bin,
                    **contrast_for_box(gray, box, occupied),
                }
            )
        if index % 500 == 0 or index == len(images):
            print(f"{split}: {index}/{len(images)} images", flush=True)
    return pd.DataFrame.from_records(records), {"images": len(images), "duplicate_rows_removed": duplicate_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    audit: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        frame, split_audit = process_split(args.dataset_root, split)
        frames[split] = frame
        audit[split] = split_audit

    valid_train = frames["train"]["ltc"].dropna().to_numpy()
    q33, q67 = np.percentile(valid_train, [100 / 3, 200 / 3], method="linear")
    for split, frame in frames.items():
        frame["contrast_bin"] = pd.cut(
            frame["ltc"],
            bins=[-np.inf, q33, q67, np.inf],
            labels=["low", "medium", "high"],
            right=True,
        )
        frame.to_csv(args.output_root / f"gt_metadata_{split}.csv", index=False, float_format="%.8f")

    script_path = Path(__file__).resolve()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Frozen model-independent LTC bin thresholds from deduplicated training GT",
        "dataset_root": str(args.dataset_root.resolve()),
        "formula": "abs(median(object)-median(background))/(1.4826*MAD(background)+1/255)",
        "percentile_method": "numpy linear",
        "q33_train": float(q33),
        "q67_train": float(q67),
        "script": str(script_path),
        "script_sha256": sha256(script_path),
        "audit": audit,
        "valid_ltc": {split: int(frame["ltc"].notna().sum()) for split, frame in frames.items()},
        "invalid_ltc": {split: int(frame["ltc"].isna().sum()) for split, frame in frames.items()},
        "tiny_proxy": {split: int(frame["tiny_proxy_flag"].sum()) for split, frame in frames.items()},
        "background_mode_counts": {
            split: {str(k): int(v) for k, v in frame["background_mode"].value_counts().items()}
            for split, frame in frames.items()
        },
        "contrast_bin_counts": {
            split: {str(k): int(v) for k, v in frame["contrast_bin"].value_counts().items()}
            for split, frame in frames.items()
        },
    }
    (args.output_root / "contrast_thresholds.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"q33_train": q33, "q67_train": q67, "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
