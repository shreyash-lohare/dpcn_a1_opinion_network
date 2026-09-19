"""All plotting. Every figure is written to the pipeline's output directory."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx


def visualize_network(
    graph: nx.Graph,
    output_path: Path,
    title: str,
    edge_threshold: Optional[float] = None,
    random_state: int = 42,
    save_pdf: bool = True,
) -> nx.Graph:
    if edge_threshold is None:
        draw_graph = graph.copy()
    else:
        draw_graph = nx.Graph()
        draw_graph.add_nodes_from(graph.nodes(data=True))
        draw_graph.add_edges_from(
            (u, v, data)
            for u, v, data in graph.edges(data=True)
            if data["abs_weight"] >= edge_threshold
        )

    pos = nx.spring_layout(draw_graph, seed=random_state, weight="abs_weight", k=0.62, iterations=300)
    block_colors = {"T": "#2E86AB", "E": "#7B9E4A", "S": "#8E5EA2", "V": "#D17A22"}
    node_colors = [block_colors.get(node[0], "#666666") for node in draw_graph.nodes]

    positive_edges = [(u, v) for u, v, data in draw_graph.edges(data=True) if data["weight"] >= 0]
    negative_edges = [(u, v) for u, v, data in draw_graph.edges(data=True) if data["weight"] < 0]
    pos_widths = [0.4 + 5.5 * draw_graph[u][v]["abs_weight"] for u, v in positive_edges]
    neg_widths = [0.4 + 5.5 * draw_graph[u][v]["abs_weight"] for u, v in negative_edges]
    pos_alpha = [0.15 + 0.65 * draw_graph[u][v]["abs_weight"] for u, v in positive_edges]
    neg_alpha = [0.15 + 0.65 * draw_graph[u][v]["abs_weight"] for u, v in negative_edges]

    plt.figure(figsize=(13, 10))
    nx.draw_networkx_edges(
        draw_graph,
        pos,
        edgelist=positive_edges,
        width=pos_widths,
        edge_color="#1F77B4",
        alpha=pos_alpha,
    )
    nx.draw_networkx_edges(
        draw_graph,
        pos,
        edgelist=negative_edges,
        width=neg_widths,
        edge_color="#D62728",
        style="dashed",
        alpha=neg_alpha,
    )
    nx.draw_networkx_nodes(
        draw_graph,
        pos,
        node_size=520,
        node_color=node_colors,
        edgecolors="#222222",
        linewidths=0.7,
    )
    nx.draw_networkx_labels(draw_graph, pos, font_size=8, font_weight="bold")

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="#1F77B4", lw=3, label="Positive Spearman correlation"),
        Line2D([0], [0], color="#D62728", lw=3, linestyle="--", label="Negative Spearman correlation"),
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markeredgecolor="#222222",
            label=label,
            markersize=9,
        )
        for label, color in [
            ("Technology", block_colors["T"]),
            ("Education", block_colors["E"]),
            ("Society/Ethics", block_colors["S"]),
            ("Environment", block_colors["V"]),
        ]
    )
    plt.legend(handles=legend_handles, loc="best", frameon=True)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    if save_pdf:
        plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    return draw_graph
