"""Shared constants for the Shreyash analysis layer.

Every module imports seed, paths, and palette from here rather than redefining
them. Kept separate from ``src/preprocessing/settings.py`` (Arijeet's ``PipelineConfig``, which
configures the sanitisation/imputation stage) so the two owners do not edit the
same file.

The palette is not invented here: it is the one already used by Arijeet's
committed figures 1-3 (``src/viz/diagnostics.py``), adopted verbatim so that no
existing figure has to be regenerated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# --- reproducibility -------------------------------------------------------
# One seed for every stochastic call in the repo: permutation, rewiring, ER,
# Louvain, and layouts. Matches PipelineConfig.random_state so the two stages
# cannot silently disagree.
SEED = 42

# --- paths -----------------------------------------------------------------
# parents[2] == repo root: src/common/config.py -> src/common -> src -> root.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "Survey_Results_UC.csv"
ARTIFACTS = ROOT / "artifacts"
FIGDIR = ROOT / "figures"
REPORT_FIGDIR = ROOT / "report" / "figures"

# --- topic blocks ----------------------------------------------------------
BLOCKS = ["T", "E", "S", "V"]
BLOCK_NAMES = {
    "T": "Technology",
    "E": "Education",
    "S": "Society / Ethics",
    "V": "Environment",
}
BLOCK_COLORS = {"T": "#2E86AB", "E": "#7B9E4A", "S": "#8E5EA2", "V": "#D17A22"}

# Signed correlations are plotted on a diverging map centred at zero.
DIVERGING_CMAP = "RdBu_r"

# Neutral accents reused across the Shreyash figures.
REAL_COLOR = "#1B998B"
NULL_COLOR = "#E4572E"
MARK_COLOR = "#2B3A55"

# --- pipeline constants ----------------------------------------------------
# Respondents with more than this many missing items are dropped: 495 of the
# 542 missing cells sit in the 10 respondents above the cut, leaving 86
# respondents with 47 missing cells (0.9%).
MAX_MISSING_ITEMS = 12
N_ITEMS = 60

# --- analysis constants ----------------------------------------------------
N_NULL_REPS = 200
THRESHOLD_GRID = np.arange(0.05, 0.501, 0.01)
REPORT_THRESHOLDS = (0.15, 0.20, 0.25, 0.30)

# --- figure output ---------------------------------------------------------
DPI = 200  # matches Arijeet's committed figures


def ensure_dirs() -> None:
    """Create output directories if absent.

    Deliberately a function rather than import-time code: the project convention
    requires modules to be side-effect free at import. Each entry point calls
    this explicitly.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORT_FIGDIR.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
