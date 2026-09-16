"""Markdown/JSON summary report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_markdown_report(
    path: Path,
    summary: dict,
    performance: pd.DataFrame,
    imputed_cells: int,
) -> None:
    grouped = performance.groupby("scenario")[["accuracy", "mae"]].mean().reset_index()
    perf_lines = "\n".join(
        f"- {row.scenario}: accuracy={row.accuracy:.3f}, MAE={row.mae:.3f}"
        for row in grouped.itertuples(index=False)
    )
    text = f"""# Sanitised Questionnaire Data Report

## Missing Response Treatment

`No Comments` was treated as missing (`NC`), not as Neutral. Structural blanks were also
treated as missing (`STRUCTURAL_BLANK`). Respondents who used `No Comments` were kept in
the analysis. No respondent is dropped for having many missing items; only entirely empty
submissions are excluded.

Confirmed raw-data counts and analysis exclusions:

- Raw respondents: {summary['raw_respondents']}
- Excluded entirely empty respondents: {summary['excluded_entirely_empty_rows']}
- Respondents analysed/exported: {summary['respondents']}
- Questions: {summary['questions']}
- Rows with no structural blanks: {summary['no_structural_blank_rows']}
- Fully observed Likert rows: {summary['fully_observed_likert_rows']}
- Stopped after technology block: {summary['stopped_after_technology_rows']}
- Stopped partway through environment: {summary['stopped_partway_environment_rows']}
- Entirely empty rows retained for imputation: {summary['entirely_empty_rows']}
- Structural blank cells: {summary['structural_blank_cells']}
- NC cells: {summary['nc_cells']}
- NC users: {summary['nc_users']}

## Row-Centring

The grand mean of raw responses is well above zero (acquiescence bias): respondents agree
with nearly everything, so raw correlation between two items is inflated by shared
agreeableness rather than genuine item-to-item association. Each respondent's own mean
across all 60 imputed items is therefore subtracted from their row before the correlation
step. Centring is applied strictly after imputation (imputation still predicts on the
original -2..+2 scale) and strictly before correlation.

Effect of centring on the correlation matrix, measured at the same edge threshold used for
the network below:

| Matrix | Mean \\|r\\| | Median \\|r\\| | Edges at threshold {summary['edge_threshold']:.2f} |
|---|---|---|---|
| Raw (uncentred) | {summary['raw_mean_abs_corr']:.3f} | {summary['raw_median_abs_corr']:.3f} | {summary['raw_edge_count_at_threshold']} |
| Row-centred | {summary['centred_mean_abs_corr']:.3f} | {summary['centred_median_abs_corr']:.3f} | {summary['centred_edge_count_at_threshold']} |

## Correlation Network

The network has one node per questionnaire item. Edges are Spearman rank correlations
computed pairwise using available observations from the row-centred matrix above. Spearman
is the primary ordinal/rank-based correlation; Pearson is not used as the primary network
edge because the responses are ordered Likert categories. The implementation keeps the
correlation step modular so a future polychoric association estimator can replace it.

## Ordinal Regression Imputation

Each question is predicted by a separate regularized proportional-odds ordinal logistic
regression model using the other 59 questions. Missing predictors are imputed with
training-column medians before standardization, consistently during validation and final
prediction. Missing predictors are not encoded as Neutral.

## Validation

Validation uses artificial masking rather than genuine missing cells, whose true values
are unknown. Two mechanisms are evaluated separately: scattered NC-like masking and
contiguous structural suffix masking. In structural validation, answers after the
simulated stopping point are hidden from the predictor row before prediction.

{perf_lines}

## Final Imputation

The selected ordinal regression procedure imputed {imputed_cells} genuinely missing
questionnaire cells after excluding the entirely empty submissions. `dataset_imputed.csv`
and `sanitised_data.csv` store the final questionnaire responses numerically as `-2`,
`-1`, `0`, `1`, and `2`, on the original (uncentred) scale. `dataset_imputed_labels.csv` is
provided only as a human-readable label copy. `row_centred_data.csv` stores the same
respondents and items after row-centring (each respondent's own mean across all 60 items
subtracted) -- this is the matrix the correlation network is actually built from.

## NC Robustness Notes

Prior NC diagnostics recorded in the project context were: 37 NC cells, 17 NC users,
0.6% NC cells, dispersion index 1.10, chi-square uniformity p=0.27, correlation with
item SD r=-0.02, mean score NC users=4.01, mean score non-NC users=4.15, mean-comparison
p=0.27, neutral rate NC users=0.123, neutral rate non-NC users=0.142, and neutral-rate
comparison p=0.82. These diagnostics support treating NC as missing rather than Neutral.
"""
    path.write_text(text, encoding="utf-8")
