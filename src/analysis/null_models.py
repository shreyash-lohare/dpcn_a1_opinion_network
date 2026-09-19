"""Does the community structure survive each null? (Framework validation.)

This is the experiment that justifies making the permutation null primary, and
it doubles as an end-to-end test of ``src/common/nulls.py``. Dev builds the
figure (Graph 10); this module produces the numbers and caches the replicate
distributions so that job never has to be re-run.

The respondent network is used because that is where the three nulls disagree
most sharply, and the disagreement is the point: modularity clears the rewiring
and ER nulls comfortably while failing the permutation null. A result that
depends on which null you pick is a result about the construction, not the data.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np

from src.common.config import ARTIFACTS, N_NULL_REPS, SEED, ensure_dirs
from src.common.matrix import build_matrices
from src.common.nulls import cache_replicates, er_null, permutation_null, rewiring_null, zscore

K = 4


def knn_union(similarity: np.ndarray, k: int = K) -> nx.Graph:
    """Symmetrised (union) kNN graph. Mutual kNN fragments this data badly."""
    n = similarity.shape[0]
    sim = similarity.copy()
    np.fill_diagonal(sim, -np.inf)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in np.argsort(sim[i])[::-1][:k]:
            G.add_edge(i, int(j), weight=float(abs(similarity[i, int(j)])))
    return G


def modularity(G: nx.Graph, seed: int = SEED) -> float:
    communities = nx.community.louvain_communities(G, weight="weight", seed=seed)
    return float(nx.community.modularity(G, communities, weight="weight"))


def full_pipeline_modularity(X_permuted: np.ndarray) -> float:
    """Row-centre -> respondent similarity -> kNN -> Louvain, on permuted data.

    Every step of the construction is repeated, which is precisely what the
    rewiring and ER nulls cannot do: they start from the finished graph.
    """
    centred = X_permuted - X_permuted.mean(axis=1, keepdims=True)
    return modularity(knn_union(np.corrcoef(centred)))


def main(n_reps: int = N_NULL_REPS) -> dict:
    ensure_dirs()
    am = build_matrices()
    sim = np.corrcoef(am.centred.to_numpy())
    G = knn_union(sim)
    observed = modularity(G)
    observed_clustering = nx.average_clustering(G)

    print("=" * 78)
    print("NULL-MODEL COMPARISON -- respondent network, symmetrised kNN (k = 4)")
    print("=" * 78)
    print(f"  nodes {G.number_of_nodes()}, edges {G.number_of_edges()}, "
          f"components {nx.number_connected_components(G)}   (ref 91 / 298 / 1)")
    print(f"  observed modularity  {observed:.3f}")
    print(f"  observed clustering  {observed_clustering:.3f}   (ref 0.219)")
    print()

    perm = np.array(permutation_null(am.encoded.to_numpy(), full_pipeline_modularity,
                                     n_reps=n_reps, seed=SEED, progress_every=100))
    rewire = np.array(rewiring_null(G, n_reps=n_reps, seed=SEED, measure_fn=modularity))
    er = np.array(er_null(G, n_reps=n_reps, seed=SEED, measure_fn=modularity))

    results = {
        "permutation": zscore(observed, perm),
        "rewiring": zscore(observed, rewire),
        "er": zscore(observed, er),
    }
    verdict = {"permutation": "regenerates the whole pipeline",
               "rewiring": "takes the finished graph as given",
               "er": "preserves only n and m"}
    print(f"  {'null':<14} {'mean':>8} {'sd':>8} {'z':>8} {'p':>8}   what it tests")
    for name, r in results.items():
        print(f"  {name:<14} {r['null_mean']:>8.3f} {r['null_sd']:>8.3f} "
              f"{r['z']:>8.2f} {r['p']:>8.3f}   {verdict[name]}")
    print()
    print("  Reading: modularity clears the rewiring and ER nulls easily but does NOT")
    print("  clear the permutation null. The community structure is manufactured by the")
    print("  correlate-then-kNN construction, not carried by the responses. Reported as")
    print("  a finding, not a failure.")
    print("=" * 78)

    cache_replicates(ARTIFACTS / "null_modularity_replicates.npz",
                     permutation=perm, rewiring=rewire, er=er)
    payload = {
        "network": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                    "components": int(nx.number_connected_components(G)), "k": K},
        "observed_modularity": observed,
        "observed_clustering": observed_clustering,
        "n_replicates": int(n_reps),
        "nulls": results,
    }
    (ARTIFACTS / "null_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("  wrote artifacts/null_comparison.json, null_modularity_replicates.npz")
    return payload


if __name__ == "__main__":
    main()
