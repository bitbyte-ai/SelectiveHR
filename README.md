# SelectiveHR

**Output-Guided Resolution Allocation for Thermal Small-Object Detection**

SelectiveHR uses a detector's low-resolution outputs to select images for a higher-resolution second pass. It requires no trained routing network. The same checkpoint runs both passes; the selected high-resolution outputs replace the corresponding low-resolution outputs.

This repository provides the method implementation, frozen settings, training configurations and evaluation tools. It does not include datasets, checkpoints, paper figures or experiment archives. Code is licensed under [AGPL-3.0](LICENSE).

## Install

Use Python 3.10 or newer in a separate environment. Run the following commands from this repository root:

```sh
python -m pip install -e .
```

## CPU example

```sh
python -m selectivehr route --predictions examples/low_predictions.jsonl --budget 0.20 --output outputs/toy
```

This synthetic example selects `example_c.png` from five records and writes `outputs/toy/selection.json`. It needs no GPU, dataset or weights. Choose a new output directory for each run.

Each input record contains an image ID, native image width and height, xyxy boxes in native coordinates, and confidence scores. Use post-NMS predictions retained at confidence 0.001, NMS IoU 0.70 and at most 300 detections per image.

## Detector inference

Install the optional dependency with a suitable PyTorch build, then supply your local detection checkpoint and a flat image directory:

```sh
python -m pip install -e ".[detector]"
python -m selectivehr infer --model /path/to/best.pt --images /path/to/images --device 0 --budget 0.20 --output outputs/inference
```

The implementation first processes all images at low resolution, then reruns the selected images at high resolution. It saves final replacement predictions and selection scores. The paper uses input sizes 640/960 and a 20% budget. This is batch-level selection, not a validated streaming scheduler.

## Frozen selection rule

The router sums source-standardized values of three cues: low-confidence small-box count, ultra-tiny-box count and candidate density. Box areas use native image coordinates. The source mean and population standard deviation are fixed in `configs/source_normalization.json`; they are not refitted on target images. Selection uses no labels or high-resolution outputs.

Select `round(budget * N)` images, sorting by descending score and then ascending image ID. The original 1,127-image setting therefore selects 225 images at 20%.

## Evaluation and tests

```sh
python -m pip install -e ".[analysis,test]"
python -m pytest -q
```

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for data preparation, training and saved-prediction evaluation. [DATA.md](docs/DATA.md) describes data access and weight availability. [THIRD_PARTY.md](docs/THIRD_PARTY.md) describes dependency licenses. `PROVENANCE.json` records the origins and adaptations of imported project files.

The evaluation tools retain the paper's custom metrics. They should not be assumed identical to native Ultralytics or COCO metrics. Higher-resolution rerunning does not help every detector, and reduced pixel work does not imply lower measured latency than direct high-resolution inference. Check performance on the intended setting with the same evaluation protocol.

Original checkpoints are not distributed here, so this is not a one-command reproduction of every reported experiment. Manuscript authorship and citation metadata will be added when finalized.
