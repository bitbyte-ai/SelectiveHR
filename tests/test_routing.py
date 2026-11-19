import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from selectivehr import features, score_features, select_batch, reroute
from transfer_selective_hr_s4 import fixed_s4_scores


def test_release_converter_exposes_only_train_and_validation():
    from convert_flir_v2_ms9_to_yolo import SPLITS
    assert set(SPLITS) == {"train", "val"}

ROOT = Path(__file__).resolve().parents[1]
NORM = json.loads((ROOT / "configs/source_normalization.json").read_text())


def record(name, boxes=(), scores=()):
    return {"image": name, "width": 640, "height": 512, "boxes": list(boxes), "confidences": list(scores)}


def test_strict_native_area_and_confidence_thresholds():
    row = record("a", [[0, 0, 15, 15], [0, 0, 16, 16], [0, 0, 32, 32]], [.25, .249, .1])
    f = features(row)
    assert f["low_conf_small_count"] == 1
    assert f["count_area_lt16"] == 1
    assert f["candidate_density"] == 3 / (640 * 512)


def test_empty_prediction_image_is_kept():
    result = select_batch([record(str(i)) for i in range(5)], NORM)
    assert result["selected"] == ["0"]
    assert len(result["scores"]) == 5


def test_frozen_score_matches_original_transfer_function():
    rows = [features(record("a")), features(record("b", [[1, 2, 8, 9]], [.2]))]
    np.testing.assert_array_equal(score_features(rows, NORM), fixed_s4_scores(pd.DataFrame(rows), NORM))


def test_ties_use_image_ids_not_input_order():
    a = select_batch([record(n) for n in ["z", "b", "a", "d", "c"]], NORM, .4)
    assert a["selected"] == ["a", "b"]


@pytest.mark.parametrize("budget", [-.01, 1.01, float("nan")])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        select_batch([], NORM, budget)


def test_duplicate_ids_fail():
    with pytest.raises(ValueError):
        select_batch([record("a"), record("a")], NORM)


def test_budget_rounding_and_zero_budget():
    rows = [record(str(i)) for i in range(5)]
    assert select_batch(rows, NORM, .5)["selected_images"] == 2
    assert select_batch(rows, NORM, 0)["selected"] == []
    assert select_batch(rows, NORM, 1)["selected_images"] == 5
    assert select_batch([], NORM)["selected_images"] == 0


def test_high_resolution_is_called_only_for_selected_images_and_replaces():
    calls = []
    low = [record(str(i)) for i in range(5)]
    def high(name):
        calls.append(name)
        return {"image": name, "sentinel": "higher-resolution-only"}
    output, selection = reroute(low, NORM, high)
    assert calls == selection["selected"] == ["0"]
    assert output[0] == {"image": "0", "sentinel": "higher-resolution-only"}
    assert output[1] is low[1]


def test_nonfinite_predictions_fail():
    with pytest.raises(ValueError):
        features(record("a", [[0, 0, float("nan"), 1]], [.1]))


def test_demo_selects_expected_image():
    rows = [json.loads(line) for line in (ROOT / "examples/low_predictions.jsonl").read_text().splitlines()]
    assert select_batch(rows, NORM)["selected"] == ["example_c.png"]
