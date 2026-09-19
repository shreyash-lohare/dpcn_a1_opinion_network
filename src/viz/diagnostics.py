"""Diagnostic figures documenting preprocessing decisions.

These figures characterise the dataset (missingness structure, item
variance, the effect of row-centring) so that decisions made or rejected in
the main pipeline (src/preprocessing/pipeline.py) are demonstrated, not just asserted.
Graph 1 operates on the raw 96-respondent CSV directly; Graph 2 reads the
already-imputed 91-respondent matrix produced by src/preprocessing/pipeline.py, since item
variance is a property of the analysis-ready data, not the raw file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.preprocessing.loader import encode_responses, identify_missingness, load_data, parse_question_columns
from src.preprocessing.similarity import compute_spearman_matrix, row_centre

FIGURES_DIR = Path("figures")
RAW_CSV = Path("data/Survey_Results_UC.csv")
IMPUTED_CSV = Path("outputs/sanitised_data/dataset_imputed.csv")

PRESENT_COLOR = "#EDEFF2"
MISSING_COLOR = "#2B3A55"
CUTOFF_COLOR = "#E4572E"
HIST_COLOR = "#1B998B"
RAW_CORR_COLOR = "#E4572E"
CENTRED_CORR_COLOR = "#1B998B"

BLOCK_COLORS = {"T": "#2E86AB", "E": "#7B9E4A", "S": "#8E5EA2", "V": "#D17A22"}
BLOCK_NAMES = {"T": "Technology", "E": "Education", "S": "Society/Ethics", "V": "Environment"}


def _load_raw_encoded() -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = load_data(RAW_CSV)
    question_cols, question_codes, _ = parse_question_columns(raw_df)
    _structural_blank, _nc = identify_missingness(raw_df, question_cols)
    encoded = encode_responses(raw_df, question_cols, question_codes)
    return raw_df, encoded


def plot_missingness_and_response_distribution(
    output_path: Path = FIGURES_DIR / "fig01_missingness.png",
) -> Dict[str, object]:
    """Cutoff marks what this project's pipeline actually does: drop only the
    entirely empty respondents (all 60 items missing); everyone else is kept
    and their missing cells are filled by the ordinal regression imputer in
    src/preprocessing/imputation.py, not by a missing-item-count threshold.
    """
    raw_df, encoded = _load_raw_encoded()
    respondent_ids = raw_df.iloc[:, 0].astype(str)
    n_items = encoded.shape[1]

    missing_mask = encoded.isna()  # True where structural blank OR "No Comments"
    missing_per_row = missing_mask.sum(axis=1)

    order = missing_per_row.sort_values(ascending=False).index
    sorted_mask = missing_mask.loc[order]
    sorted_counts = missing_per_row.loc[order]
    sorted_ids = respondent_ids.loc[order]

    is_fully_empty = sorted_counts == n_items
    n_dropped = int(is_fully_empty.sum())
    n_retained = len(sorted_counts) - n_dropped
    cells_dropped = int(sorted_counts[is_fully_empty].sum())
    cells_to_impute = int(sorted_counts[~is_fully_empty].sum())
    cells_possible_after_drop = n_retained * n_items

    values = encoded.to_numpy(dtype=float)
    grand_mean = float(np.nanmean(values))

    respondent_means = encoded.mean(axis=1, skipna=True)  # NaN for zero-answered rows

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    cmap = ListedColormap([PRESENT_COLOR, MISSING_COLOR])
    ax.imshow(sorted_mask.to_numpy(dtype=float), aspect="auto", cmap=cmap, interpolation="none")
    for boundary in (15, 30, 45):
        ax.axvline(boundary - 0.5, color="#FFFFFF", linewidth=0.8)
    ax.axhline(n_dropped - 0.5, color=CUTOFF_COLOR, linewidth=2.2, linestyle="--")
    ax.set_xlabel("Item (T01-T15, E01-E15, S01-S15, V01-V15)")
    ax.set_ylabel("Respondent, sorted by missing-item count (descending)")
    ax.set_title("Missingness map (dark = missing)")
    ax.text(
        n_items - 0.5,
        n_dropped - 0.5,
        f"  dropped: fully empty (n={n_dropped})\n  kept + imputed (n={n_retained})",
        color=CUTOFF_COLOR,
        fontsize=9,
        ha="left",
        va="center",
        fontweight="bold",
    )

    ax2 = axes[1]
    ax2.hist(respondent_means.dropna(), bins=20, color=HIST_COLOR, edgecolor="white")
    ax2.axvline(grand_mean, color=CUTOFF_COLOR, linewidth=2.2, linestyle="--")
    ax2.set_xlabel("Respondent mean score (own answered items only)")
    ax2.set_ylabel("Number of respondents")
    ax2.set_title(f"Per-respondent mean score (grand mean = {grand_mean:.2f})")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return {
        "total_missing_cells": int(missing_mask.to_numpy().sum()),
        "respondents_dropped": n_dropped,
        "respondents_retained": n_retained,
        "cells_dropped": cells_dropped,
        "cells_to_impute": cells_to_impute,
        "cells_possible_after_drop": cells_possible_after_drop,
        "pct_missing_after_drop": (cells_to_impute / cells_possible_after_drop * 100) if cells_possible_after_drop else float("nan"),
        "grand_mean": grand_mean,
        "top_row_missing_count": int(sorted_counts.iloc[0]),
        "top_row_respondent_id": str(sorted_ids.iloc[0]),
        "output_path": str(output_path),
    }


def plot_item_variance_ranking(
    output_path: Path = FIGURES_DIR / "fig02_item_variance.png",
    imputed_csv: Path = IMPUTED_CSV,
) -> Dict[str, object]:
    """Standard deviation per item on the encoded, UNCENTRED matrix.

    Uses the already-imputed 91-respondent matrix (ordinal regression fill,
    src/preprocessing/imputation.py), not the raw file with missing cells and not the
    row-centred matrix -- row-centring changes item variances and would
    answer a different question ("how spread out is this item after removing
    each respondent's baseline") than the one this graph asks ("how divided
    is the class on this item").
    """
    imputed_df = pd.read_csv(imputed_csv)
    question_cols, question_codes, _ = parse_question_columns(imputed_df)
    item_df = imputed_df[question_cols].copy()
    item_df.columns = [question_codes[col] for col in question_cols]

    sds = item_df.std(axis=0, ddof=1).sort_values(ascending=False)
    blocks = [code[0] for code in sds.index]
    colors = [BLOCK_COLORS[b] for b in blocks]

    fig, ax = plt.subplots(figsize=(8, 14))
    y_pos = np.arange(len(sds))
    ax.barh(y_pos, sds.to_numpy(), color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sds.index, fontsize=7)
    ax.invert_yaxis()  # highest SD (most contested) at top
    ax.set_xlabel("Standard deviation (encoded, uncentred, 91 respondents)")
    ax.set_title("Item variance ranking: contested (top) to consensus (bottom)")
    ax.legend(
        handles=[Patch(facecolor=color, label=BLOCK_NAMES[b]) for b, color in BLOCK_COLORS.items()],
        loc="lower right",
        frameon=True,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    block_mean_sd = {BLOCK_NAMES[b]: float(item_df[[c for c in sds.index if c[0] == b]].std(axis=0, ddof=1).mean()) for b in BLOCK_COLORS}

    return {
        "n_items": len(sds),
        "min_sd": float(sds.min()),
        "min_sd_item": sds.idxmin(),
        "max_sd": float(sds.max()),
        "max_sd_item": sds.idxmax(),
        "top5_contested": list(sds.index[:5]),
        "bottom5_consensus": list(sds.index[-5:]),
        "block_mean_sd": block_mean_sd,
        "output_path": str(output_path),
    }


def plot_row_centring_effect(
    output_path: Path = FIGURES_DIR / "fig03_row_centring.png",
    imputed_csv: Path = IMPUTED_CSV,
    min_pairwise_n: int = 50,
    n_bins: int = 40,
) -> Dict[str, object]:
    """Overlaid distributions of all 1,770 pairwise item correlations,
    raw vs. row-centred, on identical bin edges (see module pitfall notes:
    different bins per curve would make the visual comparison misleading).
    """
    imputed_df = pd.read_csv(imputed_csv)
    question_cols, question_codes, _ = parse_question_columns(imputed_df)
    item_df = imputed_df[question_cols].copy()
    item_df.columns = [question_codes[col] for col in question_cols]
    centred_df = row_centre(item_df)

    raw_corr, _, _ = compute_spearman_matrix(item_df, min_pairwise_n)
    centred_corr, _, _ = compute_spearman_matrix(centred_df, min_pairwise_n)

    n = len(raw_corr)
    iu = np.triu_indices(n, k=1)
    raw_values = raw_corr.to_numpy()[iu]
    centred_values = centred_corr.to_numpy()[iu]
    assert len(raw_values) == n * (n - 1) // 2 == 1770

    raw_mean = float(np.mean(raw_values))
    centred_mean = float(np.mean(centred_values))

    combined_min = min(raw_values.min(), centred_values.min())
    combined_max = max(raw_values.max(), centred_values.max())
    bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(raw_values, bins=bin_edges, density=True, color=RAW_CORR_COLOR, alpha=0.55,
            label=f"Raw (mean r = {raw_mean:.3f})", edgecolor="white")
    ax.hist(centred_values, bins=bin_edges, density=True, color=CENTRED_CORR_COLOR, alpha=0.55,
            label=f"Row-centred (mean r = {centred_mean:.3f})", edgecolor="white")
    ax.axvline(raw_mean, color=RAW_CORR_COLOR, linewidth=2.0, linestyle="--")
    ax.axvline(centred_mean, color=CENTRED_CORR_COLOR, linewidth=2.0, linestyle="--")
    ax.axvline(0.0, color="#888888", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Pairwise item Spearman correlation (upper triangle, 1,770 pairs)")
    ax.set_ylabel("Density")
    ax.set_title("Row-centring effect on the item-item correlation distribution")
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    return {
        "n_pairs": len(raw_values),
        "raw_mean": raw_mean,
        "raw_median": float(np.median(raw_values)),
        "raw_skew": float(pd.Series(raw_values).skew()),
        "centred_mean": centred_mean,
        "centred_median": float(np.median(centred_values)),
        "centred_skew": float(pd.Series(centred_values).skew()),
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    stats = plot_missingness_and_response_distribution()
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()
    stats2 = plot_item_variance_ranking()
    for key, value in stats2.items():
        print(f"{key}: {value}")
    print()
    stats3 = plot_row_centring_effect()
    for key, value in stats3.items():
        print(f"{key}: {value}")
