"""Parsing, encoding, missingness identification, and dataset assembly.

Respondents are only dropped when entirely empty. Neither this module nor the
rest of the pipeline drops respondents above a missing-item threshold (e.g.
">12 missing") -- that filter appears in draft spec documents but is not
applied here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.config import RESPONSE_ENCODING


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_question_columns(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    question_cols = [
        col
        for col in df.columns
        if len(col) >= 3 and col[0] in {"T", "E", "S", "V"} and col[1:3].isdigit()
    ]
    question_codes = {col: col.split(".", 1)[0].strip() for col in question_cols}
    question_text = {
        question_codes[col]: col.split(".", 1)[1].strip() if "." in col else col
        for col in question_cols
    }
    return question_cols, question_codes, question_text


def identify_missingness(df: pd.DataFrame, question_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    stripped = df.loc[:, question_cols].map(lambda value: value.strip() if isinstance(value, str) else value)
    structural_blank = stripped.isna() | stripped.eq("")
    nc = stripped.eq("No Comments")
    return structural_blank, nc


def encode_responses(df: pd.DataFrame, question_cols: Sequence[str], question_codes: Dict[str, str]) -> pd.DataFrame:
    encoded = df.loc[:, question_cols].replace(RESPONSE_ENCODING)
    encoded = encoded.mask(encoded.eq("No Comments"))
    encoded = encoded.apply(pd.to_numeric, errors="coerce")
    encoded.columns = [question_codes[col] for col in question_cols]
    return encoded.astype(float)


def completion_summary(
    encoded: pd.DataFrame, structural_blank: pd.DataFrame, nc: pd.DataFrame
) -> Dict[str, int]:
    answered_likert = encoded.notna().sum(axis=1)
    no_structural_blanks = (structural_blank.sum(axis=1) == 0).sum()
    return {
        "respondents": int(len(encoded)),
        "questions": int(encoded.shape[1]),
        "no_structural_blank_rows": int(no_structural_blanks),
        "fully_observed_likert_rows": int((answered_likert == encoded.shape[1]).sum()),
        "stopped_after_technology_rows": int((answered_likert == 15).sum()),
        "stopped_partway_environment_rows": int(((answered_likert >= 45) & (answered_likert <= 50)).sum()),
        "entirely_empty_rows": int((answered_likert == 0).sum()),
        "structural_blank_cells": int(structural_blank.to_numpy().sum()),
        "nc_cells": int(nc.to_numpy().sum()),
        "nc_users": int((nc.sum(axis=1) > 0).sum()),
        "usable_respondents": int((answered_likert > 0).sum()),
    }


def drop_entirely_empty_respondents(
    raw_df: pd.DataFrame,
    encoded: pd.DataFrame,
    structural_blank: pd.DataFrame,
    nc: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    keep_mask = encoded.notna().sum(axis=1) > 0
    return (
        raw_df.loc[keep_mask].reset_index(drop=True),
        encoded.loc[keep_mask].reset_index(drop=True),
        structural_blank.loc[keep_mask].reset_index(drop=True),
        nc.loc[keep_mask].reset_index(drop=True),
        keep_mask,
    )


def decode_response(value: int) -> str:
    reverse = {v: k for k, v in RESPONSE_ENCODING.items()}
    return reverse[int(value)]


def build_numeric_dataset(
    raw_df: pd.DataFrame,
    question_cols: Sequence[str],
    question_codes: Dict[str, str],
    imputed_numeric: pd.DataFrame,
) -> pd.DataFrame:
    cleaned = raw_df.copy()
    for original_col in question_cols:
        code = question_codes[original_col]
        cleaned[original_col] = imputed_numeric[code].astype(int)
    return cleaned


def build_label_dataset(
    raw_df: pd.DataFrame,
    question_cols: Sequence[str],
    question_codes: Dict[str, str],
    imputed_numeric: pd.DataFrame,
) -> pd.DataFrame:
    cleaned = raw_df.copy()
    for original_col in question_cols:
        code = question_codes[original_col]
        cleaned[original_col] = imputed_numeric[code].astype(int).map(decode_response)
    return cleaned
