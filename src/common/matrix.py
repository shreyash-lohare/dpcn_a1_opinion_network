"""Spec-conformant analysis matrix (CLAUDE.md section 2).

Person 1 owns parsing and encoding; this module imports
``load_data`` / ``parse_question_columns`` / ``encode_responses`` from
``src.loader`` unchanged and never edits them. What it adds is the part of the
frozen pipeline that ``src/loader.py`` deliberately does not apply (see its
module docstring): the >12-missing respondent filter, column-mean imputation,
and row-centring.

Two matrices come out, and the distinction matters:

``encoded``  86 x 60, imputed, **not** centred. This is what the permutation
             null must shuffle -- centring is part of the pipeline under test,
             so it has to happen *after* the shuffle, inside ``pipeline_fn``.
``centred``  86 x 60, row-centred. The basis for every correlation.

Column-mean imputation commutes with column permutation (permuting a column
leaves its mean unchanged), so imputing once here rather than inside every
null replicate is exact, not an approximation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.common.config import ARTIFACTS, DATA, MAX_MISSING_ITEMS, N_ITEMS
from src.loader import encode_responses, load_data, parse_question_columns


@dataclass
class AnalysisMatrix:
    """Everything downstream needs, in one object."""

    encoded: pd.DataFrame        # 86 x 60, imputed, pre-centring
    centred: pd.DataFrame        # 86 x 60, row-centred
    encoded_all: pd.DataFrame    # 96 x 60, raw encoding with NaN (for diagnostics)
    codes: List[str]             # item codes in column order, e.g. "T01"
    question_text: Dict[str, str]
    respondent_ids: pd.Series
    dropped_ids: pd.Series
    n_imputed: int

    @property
    def blocks(self) -> List[str]:
        return [c[0] for c in self.codes]


def build_matrices(
    csv_path: Path = DATA, max_missing: int = MAX_MISSING_ITEMS
) -> AnalysisMatrix:
    """Load, filter, impute, and row-centre per the frozen decisions."""
    raw = load_data(csv_path)
    question_cols, question_codes, question_text = parse_question_columns(raw)
    if len(question_cols) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} item columns, parsed {len(question_cols)}")

    encoded_all = encode_responses(raw, question_cols, question_codes)
    id_col = raw.columns[0]

    # Frozen filter: drop respondents with more than `max_missing` missing items.
    missing_per_row = encoded_all.isna().sum(axis=1)
    keep = missing_per_row <= max_missing
    kept = encoded_all.loc[keep]

    # Column-mean imputation, computed on the *retained* respondents only.
    n_imputed = int(kept.isna().to_numpy().sum())
    encoded = kept.fillna(kept.mean())

    # Row-centring: subtract each respondent's own mean across all 60 items.
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
    """Persist the handoff artefacts Person 1's downstream graphs also read."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS / "X_encoded.npy", am.encoded.to_numpy())
    np.save(ARTIFACTS / "X_centred.npy", am.centred.to_numpy())
    payload = {
        "codes": am.codes,
        "blocks": am.blocks,
        "question_text": {c: am.question_text[c] for c in am.codes},
        "n_respondents": int(len(am.encoded)),
        "n_items": int(am.encoded.shape[1]),
        "n_imputed_cells": am.n_imputed,
        "dropped_respondents": [str(x) for x in am.dropped_ids.tolist()],
    }
    (ARTIFACTS / "items.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_person1_variant() -> pd.DataFrame | None:
    """Person 1's own row-centred export (91 respondents, ordinal-imputed).

    Used only by the sensitivity annex. Returns None if absent so the main
    pipeline never hard-depends on it.
    """
    path = ARTIFACTS.parent / "outputs" / "sanitised_data" / "row_centred_data.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame.iloc[:, 1:]  # drop the respondent-ID column
