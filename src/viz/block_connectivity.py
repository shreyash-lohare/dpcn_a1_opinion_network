"""Graph 9: topic-block connectivity (4x4).

The survey ships with its own thematic partition -- 15 items each for
Technology (T), Education (E), Society/Ethics (S) and Environment (V). That
partition is an *editorial* claim about what belongs together. This module
asks whether it is also a *statistical* one: do items inside a block agree
with each other more than they agree with items in other blocks?

The 60x60 item correlation matrix is collapsed into a 4x4 matrix by averaging
the correlations in each block pair. Diagonal cells average the 105 distinct
within-block pairs; off-diagonal cells average all 225 cross-block pairs. Two
versions are shown side by side -- raw and row-centred -- because the raw
matrix is dominated by acquiescence (see similarity.py) and would make every
block pair look strongly connected regardless of topic.

Significance comes from a *block-label permutation*: the correlation matrix is
held fixed and the T/E/S/V labels are reshuffled across the 60 items, keeping
the 15/15/15/15 sizes. This is deliberately not the response-permutation null
that Shreyash owns (src/analysis/, which regenerates the whole
centre-correlate-threshold pipeline to test whether the network itself is
meaningful). The question here is narrower and needs no threshold: given this
correlation matrix, is the survey's own topic partition better aligned with it
than an arbitrary partition of the same shape would be?

Inputs are Arijeet artefacts only (the imputed and row-centred matrices from
src/preprocessing/pipeline.py), so this graph is independent of every downstream handoff.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.settings import PipelineConfig
from src.preprocessing.loader import parse_question_columns
from src.preprocessing.similarity import compute_spearman_matrix, row_centre

FIGURES_DIR = Path("figures")
OUTPUT_DIR = Path("outputs/block_connectivity")
IMPUTED_CSV = Path("outputs/sanitised_data/dataset_imputed.csv")

BLOCK_ORDER = ["T", "E", "S", "V"]
BLOCK_NAMES = {"T": "Technology", "E": "Education", "S": "Society/Ethics", "V": "Environment"}

HEATMAP_CMAP = "RdBu_r"
NULL_COLOR = "#B8C0CC"
OBSERVED_COLOR = "#E4572E"


def _block_of(code: str) -> str:
    return code[0]


def collapse_to_blocks(
    corr: pd.DataFrame, block_order: List[str] = BLOCK_ORDER
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Average a 60x60 item correlation matrix into a 4x4 block matrix.

    Diagonal cells use only the distinct within-block pairs (upper triangle,
    self-correlations excluded) -- including the 1.0 diagonal would inflate
    every within-block cell by a constant and manufacture the very
    block-assortativity this graph is testing for.
    """
    labels = np.array([_block_of(code) for code in corr.index])
    values = corr.to_numpy(dtype=float)
    n = len(corr)

    means = pd.DataFrame(np.nan, index=block_order, columns=block_order, dtype=float)
    counts = pd.DataFrame(0, index=block_order, columns=block_order, dtype=int)

    for a in block_order:
        rows = labels == a
        for b in block_order:
            cols = labels == b
            if a == b:
                sub = values[np.ix_(rows, rows)]
                iu = np.triu_indices(sub.shape[0], k=1)
                cell = sub[iu]
            else:
                cell = values[np.ix_(rows, cols)].ravel()
            cell = cell[np.isfinite(cell)]
            counts.loc[a, b] = len(cell)
            if len(cell):
                means.loc[a, b] = float(np.mean(cell))
    assert counts.to_numpy().diagonal().tolist() == [105] * len(block_order) or n != 60
    return means, counts


def within_between_gap(corr: pd.DataFrame, labels: np.ndarray) -> Tuple[float, float]:
    """Mean correlation among same-block pairs and among cross-block pairs."""
    values = corr.to_numpy(dtype=float)
    iu = np.triu_indices(len(corr), k=1)
    pair_values = values[iu]
    same_block = labels[iu[0]] == labels[iu[1]]
    finite = np.isfinite(pair_values)
    within = pair_values[same_block & finite]
    between = pair_values[~same_block & finite]
    return float(np.mean(within)), float(np.mean(between))


def permute_block_labels(
    corr: pd.DataFrame, n_permutations: int = 5000, random_state: int = 42
) -> Dict[str, object]:
    """Null distribution of the within-minus-between gap under relabelling.

    The correlation matrix is never recomputed; only the assignment of items
    to blocks is shuffled. Block sizes are preserved by permuting the existing
    label vector, so the null keeps the 15/15/15/15 design and the same number
    of within- and between-block pairs.
    """
    labels = np.array([_block_of(code) for code in corr.index])
    observed_within, observed_between = within_between_gap(corr, labels)
    observed_gap = observed_within - observed_between

    rng = np.random.default_rng(random_state)
    null_gaps = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permutation(labels)
        w, b = within_between_gap(corr, shuffled)
        null_gaps[i] = w - b

    null_mean = float(np.mean(null_gaps))
    null_sd = float(np.std(null_gaps, ddof=1))
    z = (observed_gap - null_mean) / null_sd if null_sd > 0 else float("nan")
    # one-sided: how often does an arbitrary partition beat the real one
    p = float((np.sum(null_gaps >= observed_gap) + 1) / (n_permutations + 1))

    return {
        "observed_within": observed_within,
        "observed_between": observed_between,
        "observed_gap": observed_gap,
        "null_gaps": null_gaps,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "z": float(z),
        "p_value": p,
        "n_permutations": n_permutations,
    }


def _draw_block_heatmap(ax, means: pd.DataFrame, title: str, vlim: float, show_ylabels: bool):
    image = ax.imshow(means.to_numpy(dtype=float), cmap=HEATMAP_CMAP, vmin=-vlim, vmax=vlim)
    ticks = np.arange(len(means))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(means.columns, fontsize=11)
    if show_ylabels:
        ax.set_yticklabels([f"{b}  {BLOCK_NAMES[b]}" for b in means.index], fontsize=10)
    else:
        ax.set_yticks([])  # the left panel already labels the rows
    ax.set_title(title, fontsize=11)
    for i in range(len(means)):
        for j in range(len(means)):
            value = means.iat[i, j]
            if not np.isfinite(value):
                continue
            # white text on the saturated ends, dark text in the pale middle
            shade = "white" if abs(value) > 0.60 * vlim else "#1A1A1A"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center",
                    color=shade, fontsize=10, fontweight=weight)
    return image


def plot_block_connectivity(
    output_path: Path = FIGURES_DIR / "fig09_block_connectivity.png",
    imputed_csv: Path = IMPUTED_CSV,
    config: PipelineConfig | None = None,
    n_permutations: int = 5000,
) -> Dict[str, object]:
    """Build Graph 9 and return the numbers the report quotes."""
    config = config or PipelineConfig()

    imputed_df = pd.read_csv(imputed_csv)
    question_cols, question_codes, _ = parse_question_columns(imputed_df)
    item_df = imputed_df[question_cols].copy()
    item_df.columns = [question_codes[col] for col in question_cols]
    centred_df = row_centre(item_df)

    raw_corr, _, _ = compute_spearman_matrix(item_df, config.min_pairwise_n)
    centred_corr, _, _ = compute_spearman_matrix(centred_df, config.min_pairwise_n)

    raw_means, cell_counts = collapse_to_blocks(raw_corr)
    centred_means, _ = collapse_to_blocks(centred_corr)

    permutation = permute_block_labels(centred_corr, n_permutations, config.random_state)
    raw_permutation = permute_block_labels(raw_corr, n_permutations, config.random_state)

    vlim = float(np.nanmax(np.abs(np.concatenate(
        [raw_means.to_numpy(dtype=float).ravel(), centred_means.to_numpy(dtype=float).ravel()]
    ))))
    vlim = max(vlim, 0.05)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0), gridspec_kw={"width_ratios": [1, 0.82, 1.25]})

    _draw_block_heatmap(axes[0], raw_means, "Raw (uncentred)", vlim, show_ylabels=True)
    image = _draw_block_heatmap(axes[1], centred_means, "Row-centred (network basis)", vlim,
                                show_ylabels=False)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04,
                 label="Mean item-item Spearman r (shared scale)")

    ax = axes[2]
    null_gaps = permutation["null_gaps"]
    ax.hist(null_gaps, bins=45, color=NULL_COLOR, edgecolor="white",
            label=f"Shuffled block labels ({n_permutations:,} draws)")
    ax.axvline(permutation["observed_gap"], color=OBSERVED_COLOR, linewidth=2.4,
               label=f"Observed gap = {permutation['observed_gap']:+.4f}")
    ax.axvline(permutation["null_mean"], color="#555555", linewidth=1.4, linestyle=":",
               label=f"Null mean = {permutation['null_mean']:+.4f}")
    ax.set_xlabel("Within-block mean r  minus  between-block mean r (row-centred)")
    ax.set_ylabel("Permutations")
    ax.set_title(f"Topic partition vs. shuffled labels\n"
                 f"z = {permutation['z']:.2f},  p = {permutation['p_value']:.4f}", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=9)

    fig.suptitle(
        "Graph 9: topic-block connectivity, "
        f"{item_df.shape[0]} respondents x {item_df.shape[1]} items",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_means.to_csv(OUTPUT_DIR / "block_means_raw.csv")
    centred_means.to_csv(OUTPUT_DIR / "block_means_centred.csv")
    cell_counts.to_csv(OUTPUT_DIR / "block_pair_counts.csv")

    strongest = _extreme_offdiagonal(centred_means, largest=True)
    weakest = _extreme_offdiagonal(centred_means, largest=False)

    return {
        "n_respondents": int(item_df.shape[0]),
        "n_items": int(item_df.shape[1]),
        "raw_block_means": raw_means.round(4).to_dict(),
        "centred_block_means": centred_means.round(4).to_dict(),
        "raw_within_mean": raw_permutation["observed_within"],
        "raw_between_mean": raw_permutation["observed_between"],
        "raw_gap": raw_permutation["observed_gap"],
        "raw_z": raw_permutation["z"],
        "raw_p_value": raw_permutation["p_value"],
        "centred_within_mean": permutation["observed_within"],
        "centred_between_mean": permutation["observed_between"],
        "centred_gap": permutation["observed_gap"],
        "centred_null_mean": permutation["null_mean"],
        "centred_null_sd": permutation["null_sd"],
        "centred_z": permutation["z"],
        "centred_p_value": permutation["p_value"],
        "n_permutations": n_permutations,
        "strongest_cross_block": strongest,
        "weakest_cross_block": weakest,
        "within_block_ranking": _diagonal_ranking(centred_means),
        "output_path": str(output_path),
    }


def _extreme_offdiagonal(means: pd.DataFrame, largest: bool) -> Tuple[str, str, float]:
    best = None
    for i, a in enumerate(means.index):
        for b in means.columns[i + 1 :]:
            value = float(means.loc[a, b])
            if not np.isfinite(value):
                continue
            if best is None or (value > best[2] if largest else value < best[2]):
                best = (a, b, value)
    return best


def _diagonal_ranking(means: pd.DataFrame) -> List[Tuple[str, float]]:
    diagonal = [(block, float(means.loc[block, block])) for block in means.index]
    return sorted(diagonal, key=lambda item: item[1], reverse=True)


if __name__ == "__main__":
    stats = plot_block_connectivity()
    for key, value in stats.items():
        print(f"{key}: {value}")
