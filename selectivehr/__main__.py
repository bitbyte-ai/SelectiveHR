"""CPU routing demo and optional local-checkpoint Ultralytics inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .routing import reroute, select_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    route = commands.add_parser("route", help="Select IDs using only saved low-resolution outputs")
    route.add_argument("--predictions", type=Path, required=True)
    infer = commands.add_parser("infer", help="Run a local YOLO checkpoint with selected reruns")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--images", type=Path, required=True, help="Flat directory of input images")
    infer.add_argument("--device", default="cpu")
    for command in (route, infer):
        command.add_argument("--normalization", type=Path, default=Path("configs/source_normalization.json"))
        command.add_argument("--budget", type=float, default=.20)
        command.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")

    if args.command == "route":
        records = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
        selection = select_batch(records, normalization, args.budget)
        outputs = None
    else:
        if not args.model.is_file() or not args.images.is_dir():
            parser.error("Use an existing local checkpoint and image directory")
        # Optional dependency is loaded only for real detector inference.
        from ultralytics import YOLO
        model = YOLO(str(args.model))
        paths = sorted((p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}), key=lambda p: p.name)
        if not paths:
            parser.error("No input images found")
        by_name = {p.name: p for p in paths}

        def predict(name, resolution):
            result = model.predict(source=str(by_name[name]), imgsz=resolution, device=args.device,
                                   batch=1, conf=.001, iou=.70, max_det=300, save=False, verbose=False)[0]
            height, width = result.orig_shape
            return {"image": name, "width": width, "height": height,
                    "boxes": result.boxes.xyxy.cpu().numpy().tolist(),
                    "classes": result.boxes.cls.cpu().numpy().astype(int).tolist(),
                    "confidences": result.boxes.conf.cpu().numpy().tolist(), "resolution": resolution}

        records = [predict(p.name, 640) for p in paths]
        outputs, selection = reroute(records, normalization, lambda name: predict(name, 960), args.budget)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if outputs is not None:
        (args.output / "predictions.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in outputs), encoding="utf-8")
    print(f"Selected {selection['selected_images']} of {selection['images']} images")


if __name__ == "__main__":
    main()
