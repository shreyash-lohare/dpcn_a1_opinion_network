"""Graph construction and descriptive network statistics for the item network."""

from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd


def build_correlation_network(
    corr: pd.DataFrame,
    ns: pd.DataFrame,
    pvals: pd.DataFrame,
    question_text: Dict[str, str],
) -> nx.Graph:
    graph = nx.Graph()
    for code in corr.columns:
        graph.add_node(code, label=code, text=question_text.get(code, ""), block=code[0])

    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            graph.add_edge(
                left,
                right,
                weight=float(value),
                abs_weight=float(abs(value)),
                pairwise_n=int(ns.loc[left, right]),
                p_value=float(pvals.loc[left, right]) if pd.notna(pvals.loc[left, right]) else np.nan,
            )
    return graph


def network_edges_frame(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for left, right, data in graph.edges(data=True):
        rows.append(
            {
                "source": left,
                "target": right,
                "correlation": data["weight"],
                "abs_correlation": data["abs_weight"],
                "pairwise_n": data["pairwise_n"],
                "p_value": data["p_value"],
                "sign": "positive" if data["weight"] >= 0 else "negative",
            }
        )
    return pd.DataFrame(rows).sort_values(["abs_correlation", "source", "target"], ascending=[False, True, True])


def compute_network_statistics(graph: nx.Graph, corr: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    weighted_degree = dict(graph.degree(weight="abs_weight"))
    signed_strength = {
        node: sum(data["weight"] for _, _, data in graph.edges(node, data=True))
        for node in graph.nodes
    }
    degree_centrality = nx.degree_centrality(graph)
    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "question_id": node,
                "block": graph.nodes[node].get("block"),
                "degree": int(graph.degree(node)),
                "weighted_degree_abs_strength": weighted_degree[node],
                "signed_strength": signed_strength[node],
                "degree_centrality": degree_centrality[node],
            }
        )
    node_stats = pd.DataFrame(rows).sort_values("question_id")

    weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    abs_values = np.abs(corr.where(~np.eye(len(corr), dtype=bool)).to_numpy(dtype=float))
    abs_values = abs_values[np.isfinite(abs_values)]
    summary = {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "density": nx.density(graph),
        "mean_degree": float(np.mean([degree for _, degree in graph.degree()])),
        "weighted_average_correlation": float(np.mean(weights)) if weights else np.nan,
        "mean_abs_correlation": float(np.mean(abs_values)) if len(abs_values) else np.nan,
        "median_abs_correlation": float(np.median(abs_values)) if len(abs_values) else np.nan,
        "positive_edge_count": int(sum(weight >= 0 for weight in weights)),
        "negative_edge_count": int(sum(weight < 0 for weight in weights)),
    }
    return summary, node_stats
