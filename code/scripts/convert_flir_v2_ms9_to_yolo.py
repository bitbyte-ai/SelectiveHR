"""Convert FLIR ADAS V2 thermal JPEG data to the MS-YOLO nine-class protocol.

Raw inputs are read-only. The output is built in a temporary sibling directory
and renamed into place only after all checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


CLASS_NAMES = (
    "person",
    "bike",
    "car",
    "motor",
    "bus",
    "truck",
    "light",
    "hydrant",
    "sign",
)

SPLITS = {
    "train": "images_thermal_train",
    "val": "images_thermal_val",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_name(frame: dict[str, object]) -> str:
    video = frame["videoMetadata"]
    return (
        f"video-{video['videoId']}-frame-{int(video['frameIndex']):06d}-"
        f"{frame['datasetFrameId']}.jpg"
    )


def convert_split(
    raw_root: Path,
    build_root: Path,
    split: str,
    source_dir_name: str,
) -> dict[str, object]:
    source_root = raw_root / source_dir_name
    index_path = source_root / "index.json"
    with index_path.open(encoding="utf-8") as handle:
        frames = json.load(handle)["frames"]

    image_out = build_root / "images" / split
    label_out = build_root / "labels" / split
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)

    selected = {name: index for index, name in enumerate(CLASS_NAMES)}
    selected_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    boundary_crossing_count = 0
    duplicate_annotation_count = 0
    kept_names: list[str] = []

    for frame in frames:
        width = float(frame["width"])
        height = float(frame["height"])
        rows: list[str] = []
        row_keys: set[tuple[object, ...]] = set()

        for annotation in frame.get("annotations", []):
            labels = annotation.get("labels") or []
            if not labels:
                continue
            label = labels[0]
            if label not in selected:
                excluded_counts[label] += 1
                continue

            box = annotation["boundingBox"]
            x = float(box["x"])
            y = float(box["y"])
            w = float(box["w"])
            h = float(box["h"])
            yolo_values = (
                (x + w / 2) / width,
                (y + h / 2) / height,
                w / width,
                h / height,
            )
            if w <= 0 or h <= 0 or any(value < 0 or value > 1 for value in yolo_values):
                raise ValueError(
                    f"Box cannot be represented safely in YOLO format in "
                    f"{source_dir_name}/{image_name(frame)}: "
                    f"label={label}, box={(x, y, w, h)}, image={(width, height)}"
                )
            boundary_crossing_count += int(
                x < 0 or y < 0 or x + w > width or y + h > height
            )
            row_key = (selected[label], x, y, w, h)
            duplicate_annotation_count += int(row_key in row_keys)
            row_keys.add(row_key)

            rows.append(
                f"{selected[label]} {yolo_values[0]:.6f} {yolo_values[1]:.6f} "
                f"{yolo_values[2]:.6f} {yolo_values[3]:.6f}"
            )
            selected_counts[label] += 1

        # Match the official MS-YOLO converter: retain only frames containing
        # at least one selected annotation.
        if not rows:
            continue

        name = image_name(frame)
        source_image = source_root / "data" / name
        if not source_image.is_file():
            raise FileNotFoundError(source_image)

        shutil.copy2(source_image, image_out / name)
        (label_out / f"{Path(name).stem}.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        kept_names.append(name)

    (build_root / f"{split}.txt").write_text(
        "\n".join(f"images/{split}/{name}" for name in kept_names) + "\n",
        encoding="utf-8",
    )

    return {
        "source_directory": str(source_root),
        "source_index": str(index_path),
        "source_index_sha256": sha256(index_path),
        "raw_frame_count": len(frames),
        "retained_frame_count": len(kept_names),
        "excluded_frame_count": len(frames) - len(kept_names),
        "selected_annotation_count": sum(selected_counts.values()),
        "selected_exact_duplicate_annotations": duplicate_annotation_count,
        "ultralytics_effective_annotation_count": sum(selected_counts.values()) - duplicate_annotation_count,
        "selected_annotation_counts": dict(sorted(selected_counts.items())),
        "selected_boxes_crossing_image_boundary_preserved": boundary_crossing_count,
        "selected_boxes_with_yolo_value_outside_0_1": 0,
        "excluded_annotation_count": sum(excluded_counts.values()),
        "excluded_annotation_counts": dict(sorted(excluded_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    build_root = output_root.with_name(output_root.name + "._building")
    if output_root.exists() or build_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output/staging directory: "
            f"{output_root} or {build_root}"
        )

    build_root.mkdir(parents=True)
    try:
        split_reports = {
            split: convert_split(raw_root, build_root, split, source_dir)
            for split, source_dir in SPLITS.items()
        }
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_root": str(raw_root),
            "output_root": str(output_root),
            "protocol": "MS-YOLO official nine-class index.json selection",
            "class_names_in_order": list(CLASS_NAMES),
            "class_semantics_note": (
                "Source index.json labels are preserved. The MS-YOLO paper displays "
                "bike/motor/light/hydrant/sign as bicycle/motorcycle/traffic light/"
                "fire hydrant/stop sign; source label 'sign' is not silently narrowed."
            ),
            "frame_policy": (
                "Only frames containing at least one selected annotation are retained, "
                "matching the official MS-YOLO conversion script and Table I counts."
            ),
            "boundary_box_policy": (
                "Source boxes that cross an image edge are preserved without clipping "
                "when their normalized YOLO center/width/height values remain within "
                "[0, 1], matching the official converter. Counts are reported per split."
            ),
            "raw_data_modified": False,
            "splits": split_reports,
        }
        (build_root / "conversion_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        yaml_lines = [
            f"path: {output_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            *(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)),
            "",
        ]
        (build_root / "flir_v2_ms9.yaml").write_text(
            "\n".join(yaml_lines), encoding="utf-8"
        )
        build_root.rename(output_root)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except Exception:
        print(f"Conversion failed; partial staging preserved at: {build_root}")
        raise


if __name__ == "__main__":
    main()
