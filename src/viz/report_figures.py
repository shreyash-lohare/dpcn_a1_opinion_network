"""Descriptive figures for the final report.

Built from the companion descriptive statistics in data/ (per-item and
per-block summaries computed on the raw file with pairwise deletion). These
use the original 1-5 encoding; subtract 3 to map onto the -2..+2 scale used
by the modelling pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

FIGURES_DIR = Path("figures")
ITEMS_CSV = Path("data/uc_survey_items.csv")
BLOCKS_CSV = Path("data/uc_survey_blocks.csv")
DERIVED_JSON = Path("data/uc_survey_derived.json")

LIKERT_COLORS = {
    "Strongly Disagree": "#B2182B",
    "Disagree": "#EF8A62",
    "Neutral": "#DDDDDD",
    "Agree": "#67A9CF",
    "Strongly Agree": "#2166AC",
}
BLOCK_COLORS = {"T": "#2E86AB", "E": "#7B9E4A", "S": "#8E5EA2", "V": "#D17A22"}
ACCENT = "#E4572E"


def _save(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_response_ceiling(output_path: Path = FIGURES_DIR / "fig_response_ceiling.png") -> Dict[str, object]:
    """Overall response distribution and per-block agreement profile.

    This is the figure that motivates row-centring: agreement dominates every
    block, so raw similarity between respondents largely measures how
    agreeable they are rather than what they believe.
    """
    derived = json.loads(DERIVED_JSON.read_text())
    blocks = pd.read_csv(BLOCKS_CSV)

    overall = derived["overall"]
    order = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
    counts = [overall[key] for key in order]
    total = sum(counts)
    pct = [100 * c / total for c in counts]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.15]})

    ax = axes[0]
    bars = ax.bar(range(len(order)), pct, color=[LIKERT_COLORS[k] for k in order], edgecolor="white")
    for rect, value, count in zip(bars, pct, counts):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 0.8, f"{value:.0f}%\n({count:,})",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Str.\nDisagree", "Disagree", "Neutral", "Agree", "Str.\nAgree"], fontsize=8)
    ax.set_ylabel("Share of valid answers (%)")
    ax.set_ylim(0, max(pct) * 1.28)
    ax.set_title(f"(a) {pct[3] + pct[4]:.0f}% of {total:,} answers are on the agree side", fontsize=9.5)

    ax2 = axes[1]
    y_pos = np.arange(len(blocks))[::-1]
    for y, row in zip(y_pos, blocks.itertuples(index=False)):
        left = -(row.pct_disagree + row.pct_neutral / 2)
        for width, key in ((row.pct_disagree, "Disagree"), (row.pct_neutral, "Neutral"), (row.pct_agree, "Agree")):
            colour = {"Disagree": LIKERT_COLORS["Disagree"], "Neutral": LIKERT_COLORS["Neutral"],
                      "Agree": LIKERT_COLORS["Agree"]}[key]
            ax2.barh(y, width, left=left, color=colour, edgecolor="white", height=0.62)
            left += width
        ax2.text(left + 2, y, f"mean {row.mean:.2f}", va="center", fontsize=8, color="#444444")
    ax2.axvline(0, color="#666666", linewidth=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{row.name}\n" + r"$\alpha$=" + f"{row.cronbach_alpha:.2f}"
                         for row in blocks.itertuples(index=False)], fontsize=8)
    ax2.set_xlabel("Percentage of responses (centred on neutral)")
    ax2.set_xlim(-32, 118)
    ax2.set_title("(b) Agreement profile by thematic block", fontsize=9.5)
    ax2.legend(handles=[Patch(facecolor=LIKERT_COLORS[k], label=k) for k in ("Disagree", "Neutral", "Agree")],
               loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=7.5, frameon=False)

    plt.tight_layout()
    _save(fig, output_path)

    return {
        "total_valid": total,
        "pct_agree_side": round(pct[3] + pct[4], 1),
        "pct_strongly_agree": round(pct[4], 1),
        "pct_disagree_side": round(pct[0] + pct[1], 1),
        "grand_mean_1to5": round(sum(c * (i + 1) for i, c in enumerate(counts)) / total, 3),
        "output_path": str(output_path),
    }


def plot_contested_items(output_path: Path = FIGURES_DIR / "fig_contested_items.png",
                         top_n: int = 12) -> Dict[str, object]:
    """Diverging Likert bars for the items attracting the most disagreement."""
    items = pd.read_csv(ITEMS_CSV)
    items["pct_sd"] = 100 * items["strongly_disagree"] / items["n"]
    items["pct_d"] = 100 * items["disagree"] / items["n"]
    items["pct_n"] = 100 * items["neutral"] / items["n"]
    items["pct_a"] = 100 * items["agree"] / items["n"]
    items["pct_sa"] = 100 * items["strongly_agree"] / items["n"]

    selected = items.nlargest(top_n, "pct_disagree").sort_values("pct_disagree")

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    y_pos = np.arange(len(selected))
    for y, row in zip(y_pos, selected.itertuples(index=False)):
        left = -(row.pct_sd + row.pct_d + row.pct_n / 2)
        for width, key in ((row.pct_sd, "Strongly Disagree"), (row.pct_d, "Disagree"),
                           (row.pct_n, "Neutral"), (row.pct_a, "Agree"), (row.pct_sa, "Strongly Agree")):
            ax.barh(y, width, left=left, color=LIKERT_COLORS[key], edgecolor="white", height=0.72)
            left += width
        ax.text(left + 1.5, y, f"{row.mean:.2f}", va="center", fontsize=7.5, color="#444444")

    ax.axvline(0, color="#666666", linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(selected["code"], fontsize=8)
    for tick, code in zip(ax.get_yticklabels(), selected["code"]):
        tick.set_color(BLOCK_COLORS[code[0]])
    ax.set_xlabel("Percentage of responses (centred on neutral); item mean on the right")
    ax.set_xlim(-72, 95)
    ax.set_title(f"The {top_n} items attracting the most disagreement", fontsize=10)
    ax.legend(handles=[Patch(facecolor=LIKERT_COLORS[k], label=k) for k in LIKERT_COLORS],
              loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=5, fontsize=7.5, frameon=False)

    plt.tight_layout()
    _save(fig, output_path)

    return {
        "most_disagreed": selected.iloc[-1]["code"],
        "most_disagreed_pct": float(selected.iloc[-1]["pct_disagree"]),
        "block_counts": selected["block"].value_counts().to_dict(),
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    for func in (plot_response_ceiling, plot_contested_items):
        print(f"--- {func.__name__} ---")
        for key, value in func().items():
            print(f"  {key}: {value}")
