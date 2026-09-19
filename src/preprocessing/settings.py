"""Shared constants and pipeline configuration.

Import these rather than redefining encodings, categories, or paths elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

RESPONSE_ENCODING = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly Agree": 2,
}

CATEGORIES = np.array([-2, -1, 0, 1, 2], dtype=int)
MISSING_NC = "NC"
MISSING_STRUCTURAL = "STRUCTURAL_BLANK"


@dataclass
class PipelineConfig:
    raw_csv: Path = Path("data/Survey_Results_UC.csv")
    output_dir: Path = Path("outputs/sanitised_data")
    min_pairwise_n: int = 50
    edge_threshold: float = 0.30
    test_size: float = 0.20
    random_state: int = 42
    validation_repeats: int = 5
    validation_mask_rate: float = 0.12
    structural_validation_rows: int = 40
    ordinal_l2: float = 1.0
    max_iter: int = 500
    row_centre: bool = True
