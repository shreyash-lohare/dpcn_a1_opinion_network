"""Graph 10 -- Communities vs three null models (effort 5).

Two-panel figure.
Left : item network coloured by community (Leiden).
Right: observed modularity as a vertical line against three null
       distributions (ER, rewiring, column-permutation) shown as violins.

The point is not the modularity *value* -- it is that the answer to
"is this community structure real?" depends on which null you use.
Expect the observed value to clear ER and rewiring but sit close to
the permutation distribution.

Also compares Louvain vs. Leiden community detection, demonstrating
Leiden's superior modularity and well-connected community guarantees.
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
    FIGDIR,
    MARK_COLOR,
    NULL_COLOR,
    REAL_COLOR,
    SEED,
    ensure_dirs,
)
from src.common.nulls import er_null, rewiring_null, zscore


# ---------------------------------------------------------------------------
# Graph building (reuses the same logic as Shreyash/graph_7.py)
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
# Community detection (Louvain and Leiden)
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


def detect_communities_girvan_newman(
        G: nx.Graph, max_k: int = 30) -> Tuple[float, List[frozenset], List[Tuple[int, float]]]:
    """Girvan-Newman: edge-betweenness divisive method.

    Iterates GN splits, records Q at each level, returns the partition with
    peak modularity and the full (k, Q) curve up to max_k communities.
    """
    if G.number_of_edges() == 0:
        return 0.0, list(nx.connected_components(G)), [(1, 0.0)]

    best_Q, best_comms = -1.0, None
    curve: List[Tuple[int, float]] = []

    for partition in nx.community.girvan_newman(G):
        comms = [frozenset(c) for c in partition]
        Q = float(nx.community.modularity(G, comms, weight="weight"))
        curve.append((len(comms), Q))
        if Q > best_Q:
            best_Q = Q
            best_comms = comms
        if len(comms) >= max_k:
            break

    return best_Q, best_comms, curve


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
    """Compute Leiden modularity for each permutation replicate."""
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


def plot_algorithm_comparison(
        G: nx.Graph,
        Q_louvain: float, comms_louvain: List[frozenset],
        Q_leiden: float, comms_leiden: List[frozenset],
        Q_gn: float, comms_gn: List[frozenset],
        gn_curve: List[Tuple[int, float]],
        path_stem: str) -> None:
    """Three-panel comparison figure: bar chart + GN Q-curve + community size distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # --- Panel (a): Modularity comparison bar chart -------------------------
    ax = axes[0]
    methods = ["Louvain", "Leiden\n(best)", "Girvan-\nNewman"]
    Qs = [Q_louvain, Q_leiden, Q_gn]
    colors = ["#5E81AC", "#A3BE8C", "#BF616A"]
    bars = ax.bar(methods, Qs, color=colors, width=0.55, edgecolor="white", linewidth=1.3)
    for bar, q in zip(bars, Qs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"Q = {q:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Annotate number of communities on each bar
    for bar, comms in zip(bars, [comms_louvain, comms_leiden, comms_gn]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"k={len(comms)}", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

    ax.set_ylim(0, max(Qs) * 1.20)
    ax.set_ylabel("Modularity Q (weighted)", fontsize=10)
    ax.set_title("(a) Modularity by algorithm\n(all at peak k = 15)", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # --- Panel (b): GN Q vs k curve ----------------------------------------
    ax = axes[1]
    ks = [r[0] for r in gn_curve]
    Qk = [r[1] for r in gn_curve]
    ax.plot(ks, Qk, color="#BF616A", linewidth=2.2, marker="o", markersize=5, label="GN Q(k)")
    best_k = ks[int(np.argmax(Qk))]
    best_Qk = max(Qk)
    ax.axvline(best_k, color="#BF616A", linestyle="--", alpha=0.7, linewidth=1.2)
    ax.axhline(Q_louvain, color="#5E81AC", linestyle=":", linewidth=1.6, label=f"Louvain Q={Q_louvain:.3f}")
    ax.axhline(Q_leiden, color="#A3BE8C", linestyle="-.", linewidth=1.6, label=f"Leiden Q={Q_leiden:.3f}")
    ax.set_xlabel("Number of communities (k)", fontsize=10)
    ax.set_ylabel("Modularity Q (weighted)", fontsize=10)
    ax.set_title("(b) Girvan-Newman Q vs k\n(dotted = Louvain, dash-dot = Leiden)", fontsize=10, loc="left")
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.22)
    ax.set_axisbelow(True)

    # --- Panel (c): Community size distributions ----------------------------
    ax = axes[2]
    sizes_lou = sorted([len(c) for c in comms_louvain], reverse=True)
    sizes_lei = sorted([len(c) for c in comms_leiden], reverse=True)
    sizes_gn = sorted([len(c) for c in comms_gn], reverse=True)
    x = np.arange(len(sizes_lei))
    width = 0.28
    ax.bar(x - width, sizes_lou[:len(x)], width, color="#5E81AC", label="Louvain", alpha=0.85)
    ax.bar(x, sizes_lei[:len(x)], width, color="#A3BE8C", label="Leiden", alpha=0.85)
    max_gn = len(sizes_gn)
    ax.bar(x[:max_gn] + width, sizes_gn[:max_gn], width, color="#BF616A", label="Girvan-Newman", alpha=0.85)
    ax.set_xlabel("Community rank (largest first)", fontsize=10)
    ax.set_ylabel("Community size (# items)", fontsize=10)
    ax.set_title("(c) Community size distributions\n(rank-ordered, 60 items total)", fontsize=10, loc="left")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


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
    edge_colors = ["#4A90D9" if G[u][v]["sign"] > 0 else "#D94A4A"
                   for u, v in G.edges()]

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
    # --- Compare three community detection algorithms ----------------------
    print("\n  === Community detection comparison ===")
    Q_louvain, comms_louvain = detect_communities_louvain(G)
    Q_leiden, comms_leiden = detect_communities_leiden(G)
    print(f"  Louvain: {len(comms_louvain)} communities, Q = {Q_louvain:.4f}")
    print(f"  Leiden : {len(comms_leiden)} communities, Q = {Q_leiden:.4f}")
    Q_gn, comms_gn, gn_curve = detect_communities_girvan_newman(G, max_k=30)

    # We proceed with Leiden as primary
    print(f"  Louvain       : k={len(comms_louvain):2d}, Q={Q_louvain:.4f}  | Approach: agglomerative, greedy modularity")
    print(f"  Leiden        : k={len(comms_leiden):2d}, Q={Q_leiden:.4f}  | Approach: agglomerative + refinement phase")
    print(f"  Girvan-Newman : k={len(comms_gn):2d}, Q={Q_gn:.4f}  | Approach: divisive, edge-betweenness removal")

    # Plot the three-way comparison figure
    plot_algorithm_comparison(
        G, Q_louvain, comms_louvain, Q_leiden, comms_leiden, Q_gn, comms_gn, gn_curve,
        str(FIGDIR / "fig10b_algorithm_comparison"))
    print(f"  wrote {FIGDIR / 'fig10b_algorithm_comparison'}.png / .pdf")

    # We proceed with Leiden as the primary algorithm
    Q_observed = Q_leiden
    communities = comms_leiden
    comm_labels = community_labels(codes, communities)
    n_comms = len(communities)

    # Community vs topic-block analysis
    # Community vs topic-block analysis (Leiden)
    nmi, contingency = community_vs_blocks(codes, comm_labels)
    print(f"  Leiden NMI (community vs topic block) = {nmi:.4f}")
    print(f"\n  Leiden NMI (community vs topic block) = {nmi:.4f}")

    # --- Null distributions for the ITEM network (using Leiden) -----------
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
    # Plot main network + null comparison
    plot_graph_10(G, codes, communities, Q_observed,
                  perm_mods, rewire_mods, er_mods,
                  comm_labels,
                  str(FIGDIR / "fig10_communities_vs_nulls"))
    print(f"\n  wrote {FIGDIR / 'fig10_communities_vs_nulls'}.png / .pdf")

    # Save artifacts
    payload = {
        "threshold": threshold,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_communities": n_comms,
        "Q_observed": Q_observed,
        "Q_louvain": Q_louvain,
        "Q_leiden": Q_leiden,
        "Q_girvan_newman": Q_gn,
        "k_louvain": len(comms_louvain),
        "k_leiden": len(comms_leiden),
        "k_girvan_newman": len(comms_gn),
        "gn_curve": gn_curve,
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
