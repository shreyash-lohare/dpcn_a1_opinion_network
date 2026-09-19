"""Analysis matrix for the Shreyash stages.

**Team decision (2026-09-17):** the project keeps all 91 non-empty respondents
and recovers missing cells with Arijeet's question-wise regularised
proportional-odds ordinal regression (``src/preprocessing/imputation.py``), rather than
dropping the ten respondents with more than twelve missing items and filling
the remainder with a column statistic. This module therefore consumes Arijeet's
imputed matrix directly; it does not re-impute anything.

Two matrices come out, and the distinction matters:

``encoded``  91 x 60, imputed, **not** centred. This is what the permutation
             null shuffles -- centring is part of the pipeline under test, so it
             has to happen *after* the shuffle, inside ``pipeline_fn``.
``centred``  91 x 60, row-centred. The basis for every correlation.

``build_frozen_variant()`` rebuilds the superseded 86-respondent / column-mean
pipeline. It is retained solely so the report can quantify what the respondent
filter changes (Section ``sec:rejected-p2``), not as an analysis path.

Correlations are Pearson throughout the Shreyash stages, per the frozen decision
for this project. Arijeet's own network uses Spearman; the difference is
deliberate and noted in the report, because Marchenko-Pastur theory is derived
for Pearson-type correlation matrices and the eigenvalue band has no equivalent
closed form for rank correlations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.common.config import ARTIFACTS, DATA, MAX_MISSING_ITEMS, N_ITEMS, ROOT
from src.preprocessing.loader import encode_responses, load_data, parse_question_columns

IMPUTED_CSV = ROOT / "outputs" / "sanitised_data" / "dataset_imputed.csv"


@dataclass
class AnalysisMatrix:
    """Everything downstream needs, in one object."""

    encoded: pd.DataFrame        # 91 x 60, ordinal-regression imputed, pre-centring
    centred: pd.DataFrame        # 91 x 60, row-centred
    encoded_all: pd.DataFrame    # 96 x 60, raw encoding with NaN (diagnostics only)
    codes: List[str]             # item codes in column order, e.g. "T01"
    question_text: Dict[str, str]
    respondent_ids: pd.Series
    dropped_ids: pd.Series
    n_imputed: int
    source: str

    @property
    def blocks(self) -> List[str]:
        return [c[0] for c in self.codes]


def _raw_encoded():
    """96 x 60 encoded matrix with NaN for blanks and 'No Comments'."""
    raw = load_data(DATA)
    question_cols, question_codes, question_text = parse_question_columns(raw)
    if len(question_cols) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} item columns, parsed {len(question_cols)}")
    return raw, question_cols, question_codes, question_text, encode_responses(
        raw, question_cols, question_codes
    )


def build_matrices(imputed_csv: Path = IMPUTED_CSV) -> AnalysisMatrix:
    """Primary path: Arijeet's 91-respondent ordinal-imputed matrix, row-centred."""
    if not imputed_csv.exists():
        raise FileNotFoundError(
            f"{imputed_csv.relative_to(ROOT)} is missing. It is produced by Arijeet's "
            "stage 1 (src/preprocessing/pipeline.py). Run `python run_all.py --only Arijeet` first, or "
            "`python run_all.py` to run everything in dependency order."
        )
    raw, _, _, question_text, encoded_all = _raw_encoded()
    id_col = raw.columns[0]

    frame = pd.read_csv(imputed_csv, encoding="utf-8-sig")
    item_cols = [c for c in frame.columns if re.match(r"^[TESV]\d\d", c)]
    if len(item_cols) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} item columns in {imputed_csv.name}, got {len(item_cols)}")
    encoded = frame[item_cols].astype(float)
    encoded.columns = [c.split(".", 1)[0].strip() for c in item_cols]

    # Row-centring: subtract each respondent's own mean across all 60 items.
    centred = encoded.sub(encoded.mean(axis=1), axis=0)

    kept_ids = set(frame[frame.columns[0]].astype(str))
    all_ids = raw[id_col].astype(str)
    # How many cells the imputer actually filled, measured rather than assumed.
    retained_raw = encoded_all.loc[all_ids.isin(kept_ids).to_numpy()]
    n_imputed = int(retained_raw.isna().to_numpy().sum())

    return AnalysisMatrix(
        encoded=encoded,
        centred=centred,
        encoded_all=encoded_all,
        codes=list(encoded.columns),
        question_text=question_text,
        respondent_ids=frame[frame.columns[0]].reset_index(drop=True),
        dropped_ids=all_ids[~all_ids.isin(kept_ids)].reset_index(drop=True),
        n_imputed=n_imputed,
        source="Arijeet ordinal-regression imputation (91 respondents)",
    )


def build_frozen_variant(max_missing: int = MAX_MISSING_ITEMS) -> AnalysisMatrix:
    """Superseded pipeline: drop >12 missing, column-mean fill. Sensitivity only."""
    raw, _, _, question_text, encoded_all = _raw_encoded()
    id_col = raw.columns[0]
    missing_per_row = encoded_all.isna().sum(axis=1)
    keep = missing_per_row <= max_missing
    kept = encoded_all.loc[keep]
    n_imputed = int(kept.isna().to_numpy().sum())
    encoded = kept.fillna(kept.mean())
    centred = encoded.sub(encoded.mean(axis=1), axis=0)
    return AnalysisMatrix(
        encoded=encoded,
        centred=centred,
        encoded_all=encoded_all,
        codes=list(encoded.columns),
        question_text=question_text,
        respondent_ids=raw.loc[keep, id_col].reset_index(drop=True),
        dropped_ids=raw.loc[~keep, id_col].reset_index(drop=True),
        n_imputed=n_imputed,
        source=f"frozen variant (drop >{max_missing} missing, column-mean, 86 respondents)",
    )


def correlation(centred: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Pearson item-item correlation, signs retained. Returns a 60x60 array."""
    values = centred.to_numpy() if isinstance(centred, pd.DataFrame) else np.asarray(centred)
    return np.corrcoef(values.T)


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """The n(n-1)/2 unique off-diagonal entries, as a flat array."""
    n = matrix.shape[0]
    return matrix[np.triu_indices(n, 1)]


def save_matrices(am: AnalysisMatrix) -> None:
    """Persist the handoff artefacts Arijeet and Dev also read."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS / "X_encoded.npy", am.encoded.to_numpy())
    np.save(ARTIFACTS / "X_centred.npy", am.centred.to_numpy())
    payload = {
        "source": am.source,
        "codes": am.codes,
        "blocks": am.blocks,
        "question_text": {c: am.question_text[c] for c in am.codes},
        "n_respondents": int(len(am.encoded)),
        "n_items": int(am.encoded.shape[1]),
        "n_imputed_cells": am.n_imputed,
        "dropped_respondents": [str(x) for x in am.dropped_ids.tolist()],
    }
    (ARTIFACTS / "items.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
