"""Deffuant-Weisbuch bounded-confidence opinion dynamics on the respondent network.

This is the only part of the project involving an actual dynamical process
rather than static network description.

Node set is RESPONDENTS (not items): each respondent holds one opinion, seeded
from their own answer to a contested item, and interacts with respondents they
are connected to. The respondent graph is built here from the row-centred
matrix (src/preprocessing/similarity.py), since the main pipeline only builds the item-item
network.

Discretisation note
-------------------
Rescaling a 5-point Likert item from [-2, 2] to [0, 1] places the categories at
exactly {0, 0.25, 0.5, 0.75, 1.0}, i.e. a grid of spacing 0.25. Under the
bounded-confidence rule (interact only if |x_u - x_v| < eps), no two adjacent
categories can EVER interact while eps <= 0.25. The resulting plateau at five
clusters for small eps is therefore forced by arithmetic, not an emergent
property of the network. `run_epsilon_sweep` supports a jittered control seed
that breaks the grid, which demonstrates this rather than merely asserting it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.preprocessing.loader import parse_question_columns

FIGURES_DIR = Path("figures")
OUTPUTS_DIR = Path("outputs/dynamics")
ROW_CENTRED_CSV = Path("outputs/sanitised_data/row_centred_data.csv")
IMPUTED_CSV = Path("outputs/sanitised_data/dataset_imputed.csv")

KNN_COLOR = "#2B3A55"
CONTROL_COLOR = "#E4572E"
COMPLETE_COLOR = "#1B998B"


def load_item_matrix(csv_path: Path) -> pd.DataFrame:
    """Read an exported respondent x item CSV and rename columns to item codes."""
    frame = pd.read_csv(csv_path)
    question_cols, question_codes, _ = parse_question_columns(frame)
    matrix = frame[question_cols].copy()
    matrix.columns = [question_codes[col] for col in question_cols]
    return matrix


def build_respondent_knn_graph(centred_matrix: pd.DataFrame, k: int = 4) -> nx.Graph:
    """Symmetrised (union) kNN graph over respondents, Pearson on row-centred data.

    Union rather than mutual kNN: mutual kNN fragments this data badly, and a
    fragmented graph would inflate the final cluster count for reasons that
    have nothing to do with the confidence threshold under study.
    """
    similarity = np.corrcoef(centred_matrix.to_numpy(dtype=float))
    np.fill_diagonal(similarity, -np.inf)

    graph = nx.Graph()
    graph.add_nodes_from(range(len(similarity)))
    for i, row in enumerate(similarity):
        for j in np.argsort(row)[-k:]:
            graph.add_edge(i, int(j), weight=float(row[j]))
    return graph


def rescale_to_unit(values: np.ndarray, low: float = -2.0, high: float = 2.0) -> np.ndarray:
    return (values - low) / (high - low)


def deffuant_weisbuch(
    edges: np.ndarray,
    opinions: np.ndarray,
    epsilon: float,
    n_interactions: int,
    rng: np.random.Generator,
    mu: float = 0.5,
    convergence_check_every: int = 5000,
    snapshot_every: Optional[int] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run DW until the state stops changing or the interaction budget is spent.

    mu=0.5 means each of the two interacting nodes moves halfway toward the
    other, i.e. both land on their midpoint.
    """
    x = [float(value) for value in opinions]
    edge_u = [int(u) for u in edges[:, 0]]
    edge_v = [int(v) for v in edges[:, 1]]
    n_edges = len(edge_u)

    snapshots: List[np.ndarray] = []
    if snapshot_every:
        snapshots.append(np.array(x))

    picks = rng.integers(0, n_edges, size=n_interactions)
    changed_since_check = False

    for step, edge_index in enumerate(picks):
        u = edge_u[edge_index]
        v = edge_v[edge_index]
        diff = x[v] - x[u]
        if -epsilon < diff < epsilon:
            shift = mu * diff
            x[u] += shift
            x[v] -= shift
            if shift:
                changed_since_check = True

        if snapshot_every and (step + 1) % snapshot_every == 0:
            snapshots.append(np.array(x))

        if (step + 1) % convergence_check_every == 0:
            if not changed_since_check:
                break
            changed_since_check = False

    final = np.array(x)
    if snapshot_every:
        snapshots.append(final)
    return final, snapshots


def count_clusters(opinions: np.ndarray, decimals: int = 2) -> int:
    return int(len(np.unique(np.round(opinions, decimals))))


def run_epsilon_sweep(
    edges: np.ndarray,
    seed_opinions: np.ndarray,
    epsilons: Sequence[float],
    n_repeats: int = 20,
    n_interactions: int = 60_000,
    base_seed: int = 42,
    jitter: float = 0.0,
) -> pd.DataFrame:
    """Mean/std final cluster count per epsilon over repeated runs.

    `jitter` > 0 perturbs the seed opinions uniformly by +/- jitter before each
    run, destroying the discrete Likert grid. This is the control that shows
    the small-epsilon plateau is a discretisation artefact.
    """
    rows = []
    for epsilon in epsilons:
        counts = []
        for repeat in range(n_repeats):
            rng = np.random.default_rng(base_seed + repeat * 1000 + int(epsilon * 10_000))
            start = seed_opinions.astype(float).copy()
            if jitter:
                start = np.clip(start + rng.uniform(-jitter, jitter, size=len(start)), 0.0, 1.0)
            final, _ = deffuant_weisbuch(edges, start, epsilon, n_interactions, rng)
            counts.append(count_clusters(final))
        rows.append(
            {
                "epsilon": float(epsilon),
                "mean_clusters": float(np.mean(counts)),
                "std_clusters": float(np.std(counts)),
                "min_clusters": int(np.min(counts)),
                "max_clusters": int(np.max(counts)),
            }
        )
    return pd.DataFrame(rows)


def plot_opinion_dynamics(
    output_path: Path = FIGURES_DIR / "fig12_opinion_dynamics.png",
    seed_item: str = "E03",
    k: int = 4,
    n_repeats: int = 20,
    n_interactions: int = 60_000,
    trajectory_epsilons: Tuple[float, float] = (0.15, 0.45),
    base_seed: int = 42,
) -> Dict[str, object]:
    centred = load_item_matrix(ROW_CENTRED_CSV)
    imputed = load_item_matrix(IMPUTED_CSV)

    graph = build_respondent_knn_graph(centred, k=k)
    edges = np.array(graph.edges(), dtype=int)
    seed_opinions = rescale_to_unit(imputed[seed_item].to_numpy(dtype=float))

    n_nodes = graph.number_of_nodes()
    complete_edges = np.array(list(nx.complete_graph(n_nodes).edges()), dtype=int)

    epsilons = np.round(np.arange(0.05, 0.605, 0.05), 3)

    knn_sweep = run_epsilon_sweep(edges, seed_opinions, epsilons, n_repeats, n_interactions, base_seed)
    control_sweep = run_epsilon_sweep(
        edges, seed_opinions, epsilons, n_repeats, n_interactions, base_seed, jitter=0.125
    )
    complete_sweep = run_epsilon_sweep(
        complete_edges, seed_opinions, epsilons, n_repeats, n_interactions, base_seed
    )

    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])

    ax = fig.add_subplot(grid[0, :])
    ax.errorbar(knn_sweep["epsilon"], knn_sweep["mean_clusters"], yerr=knn_sweep["std_clusters"],
                marker="o", capsize=3, color=KNN_COLOR, linewidth=2,
                label=f"{seed_item} on respondent kNN graph (k={k})")
    ax.errorbar(complete_sweep["epsilon"], complete_sweep["mean_clusters"], yerr=complete_sweep["std_clusters"],
                marker="s", capsize=3, color=COMPLETE_COLOR, linewidth=1.6, linestyle="-.",
                label="Same seed, complete graph (topology removed)")
    ax.errorbar(control_sweep["epsilon"], control_sweep["mean_clusters"], yerr=control_sweep["std_clusters"],
                marker="^", capsize=3, color=CONTROL_COLOR, linewidth=1.6, linestyle="--",
                label="Control: jittered seed (Likert grid destroyed)")
    ax.axvline(0.25, color="#888888", linewidth=1.0, linestyle=":")
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 10, 20, 40])
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:g}"))
    ax.annotate(
        "Likert grid spacing = 0.25.\nBelow this, no two adjacent categories\n"
        "can ever interact: the 5-cluster plateau\nis arithmetic, not emergence.",
        xy=(0.25, 6.0), xytext=(0.305, 13.0),
        fontsize=8, color="#555555",
        arrowprops=dict(arrowstyle="->", color="#888888", linewidth=0.8),
    )
    ax.set_xlabel(r"Confidence threshold $\varepsilon$")
    ax.set_ylabel("Mean number of final opinion clusters (log scale)")
    ax.set_title(
        f"Deffuant-Weisbuch on {seed_item} (\"compulsory attendance\"), "
        f"{n_repeats} repetitions, {n_interactions:,} interactions per run"
    )
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    for position, epsilon in enumerate(trajectory_epsilons):
        ax_t = fig.add_subplot(grid[1, position])
        rng = np.random.default_rng(base_seed)
        snapshot_every = max(1, n_interactions // 120)
        _final, snapshots = deffuant_weisbuch(
            edges, seed_opinions.copy(), epsilon, n_interactions, rng,
            snapshot_every=snapshot_every, convergence_check_every=n_interactions + 1,
        )
        history = np.array(snapshots)
        steps = np.arange(len(history)) * snapshot_every
        for node in range(history.shape[1]):
            ax_t.plot(steps, history[:, node], color=KNN_COLOR, alpha=0.25, linewidth=0.8)
        ax_t.set_xlabel("Interactions")
        ax_t.set_ylabel("Opinion")
        ax_t.set_ylim(-0.05, 1.05)
        ax_t.set_title(
            rf"Trajectories at $\varepsilon$ = {epsilon:.2f} "
            f"({count_clusters(history[-1])} clusters)"
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    knn_sweep.to_csv(OUTPUTS_DIR / "dw_sweep_knn.csv", index=False)
    control_sweep.to_csv(OUTPUTS_DIR / "dw_sweep_control_jittered.csv", index=False)
    complete_sweep.to_csv(OUTPUTS_DIR / "dw_sweep_complete_graph.csv", index=False)
    nx.write_edgelist(graph, OUTPUTS_DIR / "respondent_knn_graph.edgelist", data=["weight"])

    return {
        "seed_item": seed_item,
        "n_nodes": n_nodes,
        "n_edges": graph.number_of_edges(),
        "n_components": nx.number_connected_components(graph),
        "mean_degree": float(np.mean([d for _, d in graph.degree()])),
        "seed_unique_values": sorted(np.unique(seed_opinions).tolist()),
        "knn_sweep": knn_sweep.to_dict("records"),
        "control_sweep": control_sweep.to_dict("records"),
        "complete_sweep": complete_sweep.to_dict("records"),
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    stats = plot_opinion_dynamics()
    for key, value in stats.items():
        if key.endswith("_sweep"):
            print(f"{key}:")
            for row in value:
                print(f"    eps={row['epsilon']:.2f}  clusters={row['mean_clusters']:.2f} +/- {row['std_clusters']:.2f}")
        else:
            print(f"{key}: {value}")
