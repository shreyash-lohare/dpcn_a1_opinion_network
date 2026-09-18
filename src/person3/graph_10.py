"""Graph 10 -- Communities vs three null models (effort 5).

Two-panel figure.
Left : item network coloured by Louvain community (not topic block).
Right: observed modularity as a vertical line against three null
       distributions (ER, rewiring, column-permutation) shown as violins.

The point is not the modularity *value* -- it is that the answer to
"is this community structure real?" depends on which null you use.
Expect the observed value to clear ER and rewiring but sit close to
the permutation distribution.

Second analysis: community-vs-topic-block comparison via NMI and a
contingency table, checking whether data-driven communities respect
T/E/S/V or cut across them.
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import igraph as ig
import leidenalg

from src.common.config import (
    ARTIFACTS,
    BLOCK_COLORS,
    BLOCK_NAMES,
    DPI,
    FIGDIR_P3,
    MARK_COLOR,
    NULL_COLOR,
    REAL_COLOR,
    SEED,
    ensure_dirs,
)
from src.common.nulls import er_null, rewiring_null, zscore


# ---------------------------------------------------------------------------
# Graph building (reuses the same logic as person2/graph_7.py)
# ---------------------------------------------------------------------------

def build_signed_graph(corr: np.ndarray, codes: List[str],
                       threshold: float) -> nx.Graph:
    """Signed, weighted graph of item pairs above |r| = threshold."""
    G = nx.Graph()
    G.add_nodes_from(codes)
    n = len(codes)
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr[i, j])
            if abs(r) > threshold:
                G.add_edge(codes[i], codes[j],
                           weight=abs(r), r=r,
                           sign=1 if r > 0 else -1)
    return G


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def detect_communities_louvain(G: nx.Graph, seed: int = SEED
                               ) -> Tuple[float, List[frozenset]]:
    """Louvain on |r| weights. Returns (modularity, communities)."""
    communities = nx.community.louvain_communities(
        G, weight="weight", seed=seed)
    Q = float(nx.community.modularity(G, communities, weight="weight"))
    return Q, communities


def detect_communities_leiden(G: nx.Graph, seed: int = SEED
                              ) -> Tuple[float, List[frozenset]]:
    """Leiden on |r| weights via leidenalg. Returns (modularity, communities)."""
    if G.number_of_edges() == 0:
        return 0.0, [frozenset([n]) for n in G.nodes()]
    
    H = ig.Graph.from_networkx(G)
    if "weight" in H.edge_attributes():
        weights = [float(w) if w is not None else 1.0 for w in H.es["weight"]]
        partition = leidenalg.find_partition(
            H, leidenalg.ModularityVertexPartition, weights=weights, seed=seed)
    else:
        partition = leidenalg.find_partition(
            H, leidenalg.ModularityVertexPartition, seed=seed)

    nx_nodes = list(G.nodes())
    communities = [frozenset(nx_nodes[idx] for idx in p) for p in partition]
    Q = float(nx.community.modularity(G, communities, weight="weight"))
    return Q, communities


def community_labels(codes: List[str],
                     communities: List[frozenset]) -> Dict[str, int]:
    """Map each item code to its community index."""
    labels = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            labels[node] = idx
    return labels


# ---------------------------------------------------------------------------
# Null-model modularity for the item network
# ---------------------------------------------------------------------------

def _modularity_fn_leiden(G: nx.Graph) -> float:
    """Leiden modularity of a graph, for use as a null measure."""
    Q, _ = detect_communities_leiden(G, seed=SEED)
    return Q


def permutation_null_item_modularity(
    null_corr_matrix: np.ndarray,
    codes: List[str],
    threshold: float,
    n_respondents: int,
) -> np.ndarray:
    """Compute Leiden modularity for each permutation replicate.

    ``null_corr_matrix`` has shape (n_reps, n_pairs) where n_pairs = 1770.
    Each row is the upper triangle of a 60x60 correlation matrix computed
    from column-permuted + row-centred data (already done by P2).
    We rebuild the full matrix, threshold, build graph, run Leiden.
    """
    n_reps, n_pairs = null_corr_matrix.shape
    n_items = len(codes)
    mods = np.empty(n_reps)
    for i in range(n_reps):
        # Reconstruct full symmetric correlation matrix
        corr = np.eye(n_items)
        corr[np.triu_indices(n_items, 1)] = null_corr_matrix[i]
        corr = corr + corr.T - np.diag(np.diag(corr))
        G = build_signed_graph(corr, codes, threshold)
        mods[i] = _modularity_fn_leiden(G)
        if (i + 1) % 50 == 0:
            print(f"    permutation replicate {i + 1}/{n_reps}")
    return mods


# ---------------------------------------------------------------------------
# Community vs topic-block analysis
# ---------------------------------------------------------------------------

def _nmi(labels_a, labels_b) -> float:
    """Normalized mutual information between two label vectors (scipy only)."""
    # Build contingency matrix
    cats_a = sorted(set(labels_a))
    cats_b = sorted(set(labels_b))
    map_a = {c: i for i, c in enumerate(cats_a)}
    map_b = {c: i for i, c in enumerate(cats_b)}
    ct = np.zeros((len(cats_a), len(cats_b)), dtype=float)
    for a, b in zip(labels_a, labels_b):
        ct[map_a[a], map_b[b]] += 1
    # Entropies from marginals
    n = ct.sum()
    pa = ct.sum(axis=1) / n
    pb = ct.sum(axis=0) / n
    ha = -np.sum(pa[pa > 0] * np.log(pa[pa > 0]))
    hb = -np.sum(pb[pb > 0] * np.log(pb[pb > 0]))
    # Joint entropy
    pab = ct / n
    hab = -np.sum(pab[pab > 0] * np.log(pab[pab > 0]))
    mi = ha + hb - hab
    denom = (ha + hb) / 2
    return float(mi / denom) if denom > 0 else 0.0


def community_vs_blocks(codes: List[str], comm_labels: Dict[str, int]):
    """Contingency table and NMI between communities and topic blocks."""
    blocks = [c[0] for c in codes]
    comms = [comm_labels[c] for c in codes]
    nmi = _nmi(blocks, comms)

    # Build contingency table
    block_set = sorted(set(blocks))
    comm_set = sorted(set(comms))
    table = {}
    for b in block_set:
        table[b] = {}
        for ci in comm_set:
            table[b][ci] = 0
    for code in codes:
        b = code[0]
        ci = comm_labels[code]
        table[b][ci] += 1

    return nmi, table


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _color_palette(n_communities: int):
    """Generate distinct colours for communities."""
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(n_communities, 3))
    return [cmap(i) for i in range(n_communities)]


def plot_graph_10(G: nx.Graph, codes: List[str],
                  communities: List[frozenset],
                  Q_observed: float,
                  perm_mods: np.ndarray,
                  rewire_mods: np.ndarray,
                  er_mods: np.ndarray,
                  comm_labels: Dict[str, int],
                  path_stem: str) -> None:
    """Two-panel figure: network layout + null comparison."""
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(13, 5.8),
        gridspec_kw={"width_ratios": [1.15, 1]})

    n_comms = len(communities)
    palette = _color_palette(n_comms)

    # --- Left panel: network coloured by community -------------------------
    node_colors = [palette[comm_labels[n]] for n in G.nodes()]
    # Edge colours: blue for positive, red for negative
    edge_colors = ["#4A90D9" if G[u][v]["sign"] > 0 else "#D94A4A"
                   for u, v in G.edges()]
    edge_alphas = [0.5 for _ in G.edges()]

    pos = nx.kamada_kawai_layout(G)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax_left,
                           edge_color=edge_colors,
                           alpha=0.35, width=0.8)
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax_left,
                           node_color=node_colors,
                           node_size=180, edgecolors="white",
                           linewidths=0.6)
    # Draw labels
    nx.draw_networkx_labels(G, pos, ax=ax_left, font_size=5.5,
                            font_weight="bold")

    # Topic-block border: draw a second ring coloured by block
    for node in G.nodes():
        block = node[0]
        x, y = pos[node]
        ax_left.scatter(x, y, s=320, facecolors="none",
                        edgecolors=BLOCK_COLORS[block],
                        linewidths=1.5, zorder=3)

    ax_left.set_title(
        f"(a) Item network (|r| > 0.22), coloured by Leiden community\n"
        f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
        f"{n_comms} communities, Q = {Q_observed:.3f}",
        fontsize=10, loc="left")
    ax_left.axis("off")

    # Legend for topic-block borders
    from matplotlib.lines import Line2D
    block_handles = [
        Line2D([0], [0], marker="o", color="w",
               markeredgecolor=BLOCK_COLORS[b],
               markerfacecolor="none", markersize=8,
               markeredgewidth=2, label=f"{b}: {BLOCK_NAMES[b]}")
        for b in ["T", "E", "S", "V"]
    ]
    ax_left.legend(handles=block_handles, fontsize=7.5,
                   loc="lower left", framealpha=0.9,
                   title="Topic block (border)", title_fontsize=8)

    # --- Right panel: null distributions -----------------------------------
    null_data = [er_mods, rewire_mods, perm_mods]
    null_names = ["Erdos-Renyi\n(n, m only)",
                  "Degree-preserving\nrewiring",
                  "Column\npermutation"]
    positions = [1, 2, 3]

    parts = ax_right.violinplot(null_data, positions=positions,
                                showmeans=True, showmedians=False,
                                showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(NULL_COLOR)
        pc.set_alpha(0.45)
    parts["cmeans"].set_color(NULL_COLOR)
    parts["cmeans"].set_linewidth(1.5)

    # Observed line
    ax_right.axhline(Q_observed, color=REAL_COLOR, linewidth=2.2,
                     linestyle="-", label=f"Observed Q = {Q_observed:.3f}",
                     zorder=5)

    # Annotate z-scores above each violin
    for i, (nm, data) in enumerate(zip(null_names, null_data)):
        stats = zscore(Q_observed, data)
        y_top = float(np.max(data)) + 0.01
        ax_right.text(positions[i], y_top,
                      f"z={stats['z']:.1f}, p={stats['p']:.3f}",
                      ha="center", va="bottom", fontsize=7.5,
                      color=MARK_COLOR, fontweight="bold")

    ax_right.set_xticks(positions)
    ax_right.set_xticklabels(null_names, fontsize=9)
    ax_right.set_ylabel("Leiden modularity Q", fontsize=10)
    ax_right.set_title(
        "(b) Observed modularity vs three null models\n"
        "Column permutation is the only valid test",
        fontsize=10, loc="left")
    ax_right.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax_right.grid(alpha=0.22, linewidth=0.6, axis="y")
    ax_right.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> dict:
    """Generate Graph 10: communities vs three null models."""
    ensure_dirs()

    # Load artifacts
    corr = np.load(ARTIFACTS / "corr_denoised.npy")
    threshold_info = json.loads(
        (ARTIFACTS / "threshold.json").read_text(encoding="utf-8"))
    items_info = json.loads(
        (ARTIFACTS / "items.json").read_text(encoding="utf-8"))
    codes = items_info["codes"]
    threshold = threshold_info["chosen_threshold"]

    # Build graph
    G = build_signed_graph(corr, codes, threshold)
    print(f"  item graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges at |r| > {threshold}")

    # Detect communities: Compare Louvain and Leiden
    Q_louvain, comms_louvain = detect_communities_louvain(G)
    Q_leiden, comms_leiden = detect_communities_leiden(G)
    print(f"  Louvain: {len(comms_louvain)} communities, Q = {Q_louvain:.4f}")
    print(f"  Leiden : {len(comms_leiden)} communities, Q = {Q_leiden:.4f}")

    # We proceed with Leiden as requested
    Q_observed = Q_leiden
    communities = comms_leiden
    comm_labels = community_labels(codes, communities)
    n_comms = len(communities)

    # Community vs topic-block analysis
    nmi, contingency = community_vs_blocks(codes, comm_labels)
    print(f"  Leiden NMI (community vs topic block) = {nmi:.4f}")

    # --- Null distributions for the ITEM network --------------------------
    print("  computing null distributions for item network (using Leiden)...")

    # 1. Permutation null: rebuild from cached correlations
    null_data = np.load(ARTIFACTS / "null_replicates.npz")
    null_corr = null_data["null_corr"]  # (200, 1770)
    print(f"  permutation null: {null_corr.shape[0]} replicates loaded")
    perm_mods = permutation_null_item_modularity(
        null_corr, codes, threshold, items_info["n_respondents"])

    # 2. Rewiring null
    print("  rewiring null: 200 replicates...")
    rewire_mods = np.array(rewiring_null(
        G, n_reps=200, seed=SEED, measure_fn=_modularity_fn_leiden))

    # 3. ER null
    print("  ER null: 200 replicates...")
    er_mods = np.array(er_null(
        G, n_reps=200, seed=SEED, measure_fn=_modularity_fn_leiden))

    # z-scores
    perm_stats = zscore(Q_observed, perm_mods)
    rewire_stats = zscore(Q_observed, rewire_mods)
    er_stats = zscore(Q_observed, er_mods)

    print(f"\n  {'null':<20} {'mean':>8} {'sd':>8} {'z':>8} {'p':>8}")
    for name, s in [("permutation", perm_stats),
                    ("rewiring", rewire_stats),
                    ("ER", er_stats)]:
        print(f"  {name:<20} {s['null_mean']:>8.3f} {s['null_sd']:>8.3f} "
              f"{s['z']:>8.2f} {s['p']:>8.3f}")

    # Plot
    plot_graph_10(G, codes, communities, Q_observed,
                  perm_mods, rewire_mods, er_mods,
                  comm_labels,
                  str(FIGDIR_P3 / "graph_10_communities_nulls"))
    print(f"\n  wrote {FIGDIR_P3 / 'graph_10_communities_nulls'}.png / .pdf")

    # Save artifacts
    payload = {
        "threshold": threshold,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_communities": n_comms,
        "Q_observed": Q_observed,
        "Q_louvain": Q_louvain,
        "community_assignments": {
            code: int(comm_labels[code]) for code in codes
        },
        "community_sizes": [len(c) for c in communities],
        "nmi_vs_topic_blocks": nmi,
        "contingency_table": {
            block: {str(k): v for k, v in row.items()}
            for block, row in contingency.items()
        },
        "nulls": {
            "permutation": perm_stats,
            "rewiring": rewire_stats,
            "er": er_stats,
        },
    }
    (ARTIFACTS / "graph10_communities.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print("  wrote artifacts/graph10_communities.json")

    return payload


if __name__ == "__main__":
    run()
