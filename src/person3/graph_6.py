"""Graph 6 -- reordered correlation heatmap (effort 2).

60 x 60 heatmap of the MP-denoised correlation matrix, rows and columns
reordered by the Louvain community assignment from Graph 10, with a
topic-block colour strip along both margins and separator lines between
communities.

Uses a diverging colourmap centred at zero (RdBu_r) so that negative
correlations are visually distinct from weak positive ones -- a sequential
map would make -0.4 and +0.05 look similar, destroying the whole point of
retaining signs.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from src.common.config import (
    ARTIFACTS,
    BLOCK_COLORS,
    BLOCK_NAMES,
    DIVERGING_CMAP,
    DPI,
    FIGDIR_P3,
    ensure_dirs,
)


def _load_community_order(codes: list, comm_assignments: dict) -> list:
    """Sort items by (community_id, block_letter, item_number).

    Returns a permutation index into codes.
    """
    def sort_key(idx):
        code = codes[idx]
        comm = comm_assignments[code]
        block = code[0]
        num = int(code[1:])
        return (comm, block, num)

    return sorted(range(len(codes)), key=sort_key)


def plot_graph_6(corr: np.ndarray, codes: list,
                 comm_assignments: dict, path_stem: str) -> None:
    """Reordered correlation heatmap with community separators and
    topic-block colour strip."""

    order = _load_community_order(codes, comm_assignments)
    ordered_codes = [codes[i] for i in order]
    ordered_corr = corr[np.ix_(order, order)]

    n_communities = max(comm_assignments.values()) + 1

    # Find community boundaries in the reordered index
    ordered_comms = [comm_assignments[c] for c in ordered_codes]
    boundaries = []
    for i in range(1, len(ordered_comms)):
        if ordered_comms[i] != ordered_comms[i - 1]:
            boundaries.append(i)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(10, 9))

    # Heatmap
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(ordered_corr, cmap=DIVERGING_CMAP, norm=norm,
                   aspect="equal", interpolation="nearest")

    # Community separator lines
    for b in boundaries:
        ax.axhline(b - 0.5, color="black", linewidth=0.8, alpha=0.7)
        ax.axvline(b - 0.5, color="black", linewidth=0.8, alpha=0.7)

    # Topic-block colour strip on top and left margins
    strip_width = 1.2
    for idx, code in enumerate(ordered_codes):
        block = code[0]
        color = BLOCK_COLORS[block]
        # Top strip
        ax.add_patch(Rectangle(
            (idx - 0.5, -strip_width - 0.5), 1, strip_width,
            facecolor=color, edgecolor="none", clip_on=False))
        # Left strip
        ax.add_patch(Rectangle(
            (-strip_width - 0.5, idx - 0.5), strip_width, 1,
            facecolor=color, edgecolor="none", clip_on=False))

    # Tick labels
    ax.set_xticks(range(len(ordered_codes)))
    ax.set_xticklabels(ordered_codes, fontsize=5.5, rotation=90)
    ax.set_yticks(range(len(ordered_codes)))
    ax.set_yticklabels(ordered_codes, fontsize=5.5)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Pearson correlation (denoised)", fontsize=10)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    ax.set_title(
        "Item-item correlations reordered by Louvain community\n"
        f"MP-denoised matrix, {n_communities} communities, "
        "diverging scale centred at zero",
        fontsize=11, loc="left", pad=20)

    # Legend for topic-block strips
    from matplotlib.lines import Line2D
    block_handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=BLOCK_COLORS[b],
               markersize=10, label=f"{b}: {BLOCK_NAMES[b]}")
        for b in ["T", "E", "S", "V"]
    ]
    ax.legend(handles=block_handles, fontsize=8,
              loc="upper left", bbox_to_anchor=(1.15, 1.0),
              framealpha=0.9, title="Topic block", title_fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> dict:
    """Generate Graph 6: reordered correlation heatmap."""
    ensure_dirs()

    # Load artifacts
    corr = np.load(ARTIFACTS / "corr_denoised.npy")
    items_info = json.loads(
        (ARTIFACTS / "items.json").read_text(encoding="utf-8"))
    graph10_info = json.loads(
        (ARTIFACTS / "graph10_communities.json").read_text(encoding="utf-8"))

    codes = items_info["codes"]
    comm_assignments = graph10_info["community_assignments"]

    print(f"  denoised matrix: {corr.shape}")
    print(f"  {len(codes)} items, "
          f"{graph10_info['n_communities']} communities")

    plot_graph_6(corr, codes, comm_assignments,
                 str(FIGDIR_P3 / "graph_6_reordered_heatmap"))

    print(f"\n  wrote {FIGDIR_P3 / 'graph_6_reordered_heatmap'}.png / .pdf")

    return {"n_items": len(codes),
            "n_communities": graph10_info["n_communities"]}


if __name__ == "__main__":
    run()

