# Reproducibility guide

## Frozen routing contract

The paper runs a fixed checkpoint at input size 640, with prediction confidence 0.001, NMS IoU 0.70 and max detections 300. For each image, it counts boxes with native area < 32² and confidence < 0.25, counts boxes with native area < 16², and divides the total candidate count by native image area. It sums three z-scores using source means and population standard deviations (`ddof=0`). Target batches do not refit these statistics.

Rank descending by score, then ascending by image ID to break ties. Select `round(budget * N)` images using Python's ties-to-even rounding. At N=1127 and budget=0.20, this selects 225 images. Rerun only selected images at size 960 with the same checkpoint and replace their outputs; do not merge predictions across resolutions. The retained normalization is in `configs/source_normalization.json`. The transfer evaluator reads this JSON directly by default.

## What can be reproduced without data or weights

The README's toy example and unit tests use only included files. Figures, plotting scripts and aggregate experiment archives are omitted from this compact release.

## Detector experiments: prerequisites

The commands below document the historical workflow. They are not part of the CPU tests. They need the licensed dataset, local checkpoints, suitable hardware and separately installed detector dependencies. Use new output directories, preserve raw files and retain each export manifest.

The FLIR converter expects the downloaded source layout used by the original conversion script, including its `index.json` files. This public copy processes only train and validation; the historical test mapping is omitted and the adaptation is recorded in `PROVENANCE.json`. Read the script's source-directory mapping before conversion:

```sh
python code/scripts/convert_flir_v2_ms9_to_yolo.py /path/to/raw_flir /path/to/new_yolo_dataset
python code/scripts/prepare_flir_failure_metadata.py /path/to/new_yolo_dataset /path/to/new_metadata
```

For retraining, set the explicit local path in `code/configs/flir_adas_v2.yaml` and use one of the six frozen configurations:

```sh
python code/scripts/run_ultralytics_train_config.py --config experiments/configs/P1-Y11N-FLIR-BASELINE-S42.yaml
```

Keep a record of this path adaptation and your environment. Retraining is not necessary to use a supplied checkpoint. Original checkpoint files are not included in this release.

Export each checkpoint's low- and high-resolution predictions against the same validation metadata:

```sh
python code/scripts/export_flir_val_failure_evidence.py --model /path/to/best.pt --model-family yolo --images /path/to/new_yolo_dataset/images/val --gt-metadata /path/to/new_metadata/gt_metadata_val.csv --output outputs/low --imgsz 640 --batch 8 --device 0 --prediction-conf 0.001 --operating-conf 0.25 --nms-iou 0.70 --match-iou 0.50 --max-det 300
python code/scripts/export_flir_val_failure_evidence.py --model /path/to/best.pt --model-family yolo --images /path/to/new_yolo_dataset/images/val --gt-metadata /path/to/new_metadata/gt_metadata_val.csv --output outputs/high --imgsz 960 --batch 8 --device 0 --prediction-conf 0.001 --operating-conf 0.25 --nms-iou 0.70 --match-iou 0.50 --max-det 300
```

These full high-resolution exports support offline comparisons and oracle analysis. Deployment selection only uses the first pass; the inference API does not run the high-resolution detector on every image.

Run the frozen transfer evaluator, retaining the source normalization:

```sh
python code/scripts/transfer_selective_hr_s4.py --normalization configs/source_normalization.json --low outputs/low --high outputs/high --images /path/to/new_yolo_dataset/images/val --output outputs/transfer --tag y11_s42 --target-model YOLO11n --expected-images 1127 --random-replicates 1000 --workers 4
python code/scripts/bootstrap_selective_hr_deltas.py --low outputs/low --high outputs/high --selection outputs/transfer/y11_s42_frozen_s4_image_scores.csv --output outputs/bootstrap.json --replicates 500 --workers 4 --seed 20261002 --budget 0.20 --score-column s4_frozen_source_normalized
```

`audit_selective_hr_zero_training.py` is retained because the evaluator imports its prediction, feature and metric helpers. Its historical development diagnostics are not required to run the frozen method.

## Metrics and uncertainty

The archived scripts preserve the experiment's custom matching and AP computation. The AP helper uses 101-point interpolation; classes without predictions are left undefined and omitted by its nan-mean reduction. This definition is retained for traceability and is not guaranteed equivalent to COCO or native Ultralytics metrics. If changing the evaluator, recompute all baselines and comparisons and label the new protocol separately.

Matched-FP recall is an evaluation-time risk-control comparison, not an inference threshold that the router learns from labels. Use the exporter and evaluator together; do not replace their matching rules silently. Paired bootstrap intervals quantify image-resampling variability for fixed predictions. Seed standard deviations describe training variation over the same validation images. Neither establishes universal cross-domain performance.

The compact release does not include the original latency benchmark or timing archives. The inference wrapper is a usage example; its runtime should not be presented as a reproduction of the paper's latency measurements.
