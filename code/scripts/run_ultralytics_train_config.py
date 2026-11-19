#!/usr/bin/env python3
"""Run one frozen Ultralytics training YAML without command-line overrides."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Training config must be a mapping")
    model_path = config.pop("model")
    task = config.pop("task", "detect")
    mode = config.pop("mode", "train")
    if task != "detect" or mode != "train":
        raise ValueError(f"Unexpected task/mode: {task}/{mode}")
    from ultralytics import RTDETR, YOLO

    model_class = RTDETR if str(model_path).lower().startswith("rtdetr") else YOLO
    print(json.dumps({"config": str(args.config.resolve()), "config_sha256": sha256(args.config), "model": model_path}, indent=2))
    model_class(model_path).train(**config)


if __name__ == "__main__":
    main()
