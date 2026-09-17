"""Graph 7 -- threshold / percolation sweep.

Reporting a network at one threshold invites the obvious objection that the
threshold was chosen to produce the result. The sweep answers it: the graph is
rebuilt at every cutoff from 0.05 to 0.50 and four structural quantities are
tracked across the whole range, so the chosen value is visibly a point on a
curve rather than a lucky pick.

This also absorbs the giant-connected-component analysis entirely, covering two
standard checklist items in one figure.

Edges are built from the MP-denoised matrix (Graph 5). Signs are retained as an
edge attribute; Louvain runs on |r| because modularity is undefined for
negative weights.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.common.config import (
    ARTIFACTS,
    DPI,
    FIGDIR_P2,
    MARK_COLOR,
    NULL_COLOR,
    REAL_COLOR,
    SEED,
    THRESHOLD_GRID,
    ensure_dirs,
)
from src.common.matrix import build_matrices


def build_signed_graph(corr: np.ndarray, codes, threshold: float) -> nx.Graph:
    """Signed, weighted graph of item pairs above |r| = threshold.

    ``weight`` is |r| (used by Louvain and by any weighted metric); ``r`` keeps
    the signed value; ``sign`` is +1/-1 for the structural-balance analysis.
    """
    G = nx.Graph()
    G.add_nodes_from(codes)
    n = len(codes)
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr[i, j])
            if abs(r) > threshold:
                G.add_edge(codes[i], codes[j], weight=abs(r), r=r, sign=1 if r > 0 else -1)
    return G


def measure(G: nx.Graph, seed: int = SEED) -> dict:
    """Structural summary of one thresholded graph."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    components = list(nx.connected_components(G))
    gcc = max((len(c) for c in components), default=0)
    if m > 0:
        communities = nx.community.louvain_communities(G, weight="weight", seed=seed)
        modularity = nx.community.modularity(G, communities, weight="weight")
        n_comm = len(communities)
    else:
        modularity, n_comm = np.nan, 0
    return {
        "edges": m,
        "mean_degree": 2 * m / n if n else 0.0,
        "n_components": len(components),
        "gcc_size": gcc,
        "gcc_fraction": gcc / n if n else 0.0,
        "n_isolates": sum(1 for _, d in G.degree() if d == 0),
        "modularity": modularity,
        "n_communities": n_comm,
    }


def sweep(corr: np.ndarray, codes, grid=THRESHOLD_GRID, seed: int = SEED) -> pd.DataFrame:
    rows = []
    for t in grid:
        G = build_signed_graph(corr, codes, float(t))
        rows.append({"threshold": round(float(t), 4), **measure(G, seed=seed)})
    return pd.DataFrame(rows)


def print_table(table: pd.DataFrame, chance: pd.DataFrame | None) -> None:
    print("=" * 92)
    print("GRAPH 7 THRESHOLD SWEEP (signed graph rebuilt from the MP-denoised matrix)")
    print("=" * 92)
    header = (
        f"  {'thr':>5} {'edges':>6} {'mean k':>7} {'comps':>6} {'isol':>5} "
        f"{'GCC':>5} {'GCC%':>6} {'Q':>7} {'#comm':>6}"
    )
    if chance is not None:
        header += f" {'raw edges':>10} {'chance%':>8}"
    print(header)
    for _, row in table.iterrows():
        line = (
            f"  {row['threshold']:>5.2f} {int(row['edges']):>6} {row['mean_degree']:>7.2f} "
            f"{int(row['n_components']):>6} {int(row['n_isolates']):>5} {int(row['gcc_size']):>5} "
            f"{row['gcc_fraction'] * 100:>5.0f}% {row['modularity']:>7.3f} {int(row['n_communities']):>6}"
        )
        if chance is not None:
            c = chance.loc[np.isclose(chance["threshold"], row["threshold"])]
            if len(c):
                line += f" {int(c.iloc[0]['real_edges']):>10} {c.iloc[0]['chance_fraction'] * 100:>7.0f}%"
        print(line)
    print("=" * 92)


def plot_sweep(table, chance, chosen: float, path_stem, rule: str) -> None:
    # 2x2 rather than 4x1: the report is two-column, and a 4-high stack becomes
    # a figure roughly 19cm tall, which would take three quarters of a page as a
    # full-width float. Each column still shares the threshold axis.
    fig, axes2d = plt.subplots(2, 2, figsize=(7.5, 4.8), sharex=True)
    ax_a, ax_b, ax_c, ax_d = axes2d[0, 0], axes2d[0, 1], axes2d[1, 0], axes2d[1, 1]
    axes = [ax_a, ax_b, ax_c, ax_d]

    # --- (a) edge count, with the chance expectation overlaid -------------
    ax_a.plot(table["threshold"], table["edges"], color=REAL_COLOR, linewidth=2.2,
              label="Edges in denoised graph")
    if chance is not None:
        ax_a.plot(chance["threshold"], chance["real_edges"], color="#9AA5B1",
                  linewidth=1.6, label="Edges in raw (un-denoised) graph")
        ax_a.plot(chance["threshold"], chance["chance_edges_mean"], color=NULL_COLOR,
                  linestyle="--", linewidth=2.0, label="Expected by chance (permuted null)")
    ax_a.set_ylabel("Edges")
    ax_a.set_yscale("log")
    ax_a.set_title("(a) Edge count vs chance expectation", fontsize=10, loc="left")
    ax_a.set_ylim(bottom=0.3)
    ax_a.legend(fontsize=7.5, loc="lower left", framealpha=0.92)

    # --- (b) giant component fraction -------------------------------------
    ax_b.plot(table["threshold"], table["gcc_fraction"] * 100, color=REAL_COLOR, linewidth=2.2)
    ax_b.axhline(50, color="#9AA5B1", linestyle=":", linewidth=1.2)
    ax_b.text(0.055, 53, "50% of nodes", fontsize=8, color="#6B7580")
    ax_b.set_ylabel("GCC (% of 60 items)")
    ax_b.set_ylim(0, 105)
    ax_b.set_title("(b) Giant component: where the network shatters", fontsize=10, loc="left")

    # --- (c) component count and isolates ---------------------------------
    ax_c.plot(table["threshold"], table["n_components"], color=REAL_COLOR, linewidth=2.2,
              label="Connected components")
    ax_c.plot(table["threshold"], table["n_isolates"], color=NULL_COLOR, linewidth=1.8,
              linestyle="--", label="Isolated items")
    ax_c.set_ylabel("Count")
    ax_c.set_title("(c) Fragmentation", fontsize=10, loc="left")
    ax_c.legend(fontsize=8, loc="upper left")

    # --- (d) modularity ----------------------------------------------------
    ax_d.plot(table["threshold"], table["modularity"], color=REAL_COLOR, linewidth=2.2)
    ax_d.set_ylabel("Louvain modularity $Q$")
    ax_d.set_xlabel("Edge threshold $|r|$")
    ax_c.set_xlabel("Edge threshold $|r|$")
    ax_d.set_title("(d) Modularity (rises as the graph fragments)", fontsize=10, loc="left")

    for ax in axes:
        ax.axvline(chosen, color=MARK_COLOR, linewidth=1.8, alpha=0.85)
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(0.05, 0.50)
    ax_a.annotate(
        f"chosen $|r|$ = {chosen:.2f}",
        xy=(chosen, ax_a.get_ylim()[1]),
        xytext=(7, -14),
        textcoords="offset points",
        fontsize=9.5,
        fontweight="bold",
        color=MARK_COLOR,
    )
    # The selection rule lives in the report caption, not baked into the image
    # (CLAUDE.md section 7): as a suptitle it would force the figure metres wide.
    fig.suptitle(
        "Threshold sweep on the MP-denoised item network",
        fontsize=12,
        y=0.985,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def load_chance() -> pd.DataFrame | None:
    path = ARTIFACTS / "chance_edges_by_threshold.csv"
    return pd.read_csv(path) if path.exists() else None


def run_sweep() -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    ensure_dirs()
    am = build_matrices()
    corr = np.load(ARTIFACTS / "corr_denoised.npy")
    table = sweep(corr, am.codes)
    chance = load_chance()
    table.to_csv(ARTIFACTS / "threshold_sweep.csv", index=False)
    return table, chance, am.codes


def finalise(chosen: float, rule: str) -> dict:
    """Write threshold.json and the figure once the analyst has chosen a value."""
    ensure_dirs()
    am = build_matrices()
    corr = np.load(ARTIFACTS / "corr_denoised.npy")
    table = pd.read_csv(ARTIFACTS / "threshold_sweep.csv")
    chance = load_chance()

    G = build_signed_graph(corr, am.codes, chosen)
    stats = measure(G)
    chance_row = None
    if chance is not None:
        sel = chance.loc[np.isclose(chance["threshold"], chosen)]
        if len(sel):
            chance_row = sel.iloc[0].to_dict()

    payload = {
        "chosen_threshold": float(chosen),
        "rule": rule,
        "basis": f"MP-denoised correlation matrix ({json.loads((ARTIFACTS / 'mp_dimensions.json').read_text())['n_significant']} significant components)",
        "edges": int(stats["edges"]),
        "mean_degree": float(stats["mean_degree"]),
        "n_components": int(stats["n_components"]),
        "n_isolates": int(stats["n_isolates"]),
        "gcc_size": int(stats["gcc_size"]),
        "gcc_fraction": float(stats["gcc_fraction"]),
        "modularity": float(stats["modularity"]),
        "n_communities": int(stats["n_communities"]),
        "positive_edges": int(sum(1 for *_, d in G.edges(data=True) if d["sign"] > 0)),
        "negative_edges": int(sum(1 for *_, d in G.edges(data=True) if d["sign"] < 0)),
        "raw_edges_at_threshold": int(chance_row["real_edges"]) if chance_row else None,
        "chance_edges_at_threshold": float(chance_row["chance_edges_mean"]) if chance_row else None,
        "chance_fraction_at_threshold": float(chance_row["chance_fraction"]) if chance_row else None,
    }
    (ARTIFACTS / "threshold.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_sweep(table, chance, chosen, FIGDIR_P2 / "graph_7_threshold_sweep", rule)
    return payload


if __name__ == "__main__":
    t, c, _ = run_sweep()
    print_table(t, c)
