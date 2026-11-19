import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/scripts"))

from transfer_selective_hr_s4 import fixed_s4_scores, source_normalization


def test_fixed_transfer_uses_source_not_target_normalization():
    source = pd.DataFrame({
        "low_conf_small_count": [0.0, 2.0],
        "count_area_lt16": [10.0, 14.0],
        "candidate_density": [1.0, 3.0],
    })
    target = pd.DataFrame({
        "low_conf_small_count": [2.0],
        "count_area_lt16": [14.0],
        "candidate_density": [3.0],
    })
    norm = source_normalization(source)
    assert np.allclose(fixed_s4_scores(target, norm), [3.0])


def test_source_normalization_uses_population_std():
    source = pd.DataFrame({name: [0.0, 2.0] for name in ("low_conf_small_count", "count_area_lt16", "candidate_density")})
    norm = source_normalization(source)
    assert all(value["std_ddof0"] == 1.0 for value in norm.values())
