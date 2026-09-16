"""Row-centring and correlation computation for the item-item network.

Why row-centring is mandatory
------------------------------
The grand mean of the raw responses is +1.14 on a -2..+2 scale: respondents
agree with nearly everything (acquiescence bias). Left uncentred, that shared
bias inflates correlation between *every* pair of items at once -- a
respondent who agrees with everything pushes all 60 columns up together for
that row, which looks like item-item association but is really just one
person's general agreeableness. Spearman correlation does not remove this: it
is invariant to a monotonic transform of a single column, not to an additive
shift that differs row-by-row.

Row-centring subtracts each respondent's own mean across all 60 items from
that respondent's row before any correlation is computed. This leaves, per
respondent, only the *relative* pattern of which items they lean toward or
away from relative to their own baseline -- the part of the signal that can
actually indicate item-to-item association rather than shared response style.

This module operates on the fully-imputed matrix (see imputation.py), which
already sits on the original -2..+2 scale. Centring is a network-construction
step and must not be applied before or during imputation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def row_centre(matrix: pd.DataFrame) -> pd.DataFrame:
    """Subtract each respondent's own mean across all items (row-wise)."""
    return matrix.sub(matrix.mean(axis=1), axis=0)


def compute_spearman_matrix(
    matrix: pd.DataFrame, min_pairwise_n: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Spearman is the primary ordinal/rank-based correlation used for the network.

    Pearson is intentionally not used as the primary edge definition because the
    questionnaire responses are ordered Likert categories, not continuous measurements.
    The function is isolated so a future polychoric-correlation implementation can
    replace this association step without changing graph construction.

    Callers are responsible for passing an already row-centred matrix when that
    is the desired basis for the network (see `row_centre`).
    """
    cols = list(matrix.columns)
    corr = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    pvals = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    ns = pd.DataFrame(0, index=cols, columns=cols, dtype=int)

    for i, left in enumerate(cols):
        corr.loc[left, left] = 1.0
        pvals.loc[left, left] = 0.0
        ns.loc[left, left] = int(matrix[left].notna().sum())
        for right in cols[i + 1 :]:
            pair = matrix[[left, right]].dropna()
            n = len(pair)
            ns.loc[left, right] = ns.loc[right, left] = n
            if n < min_pairwise_n:
                continue
            if pair[left].nunique() < 2 or pair[right].nunique() < 2:
                continue
            value, pvalue = spearmanr(pair[left], pair[right])
            if np.isfinite(value):
                corr.loc[left, right] = corr.loc[right, left] = float(value)
                pvals.loc[left, right] = pvals.loc[right, left] = float(pvalue)
    return corr, ns, pvals


def summarize_correlation_matrix(corr: pd.DataFrame, edge_threshold: float) -> Dict[str, float]:
    """Cheap summary stats used to compare raw vs. row-centred correlation matrices."""
    off_diagonal = corr.where(~np.eye(len(corr), dtype=bool)).to_numpy(dtype=float)
    values = off_diagonal[np.isfinite(off_diagonal)]
    if len(values) == 0:
        return {"mean_abs_corr": np.nan, "median_abs_corr": np.nan, "edge_count_at_threshold": 0, "possible_edges": 0}
    abs_values = np.abs(values)
    return {
        "mean_abs_corr": float(np.mean(abs_values)),
        "median_abs_corr": float(np.median(abs_values)),
        "edge_count_at_threshold": int(np.sum(abs_values >= edge_threshold) // 2),
        "possible_edges": int(len(values) // 2),
    }
