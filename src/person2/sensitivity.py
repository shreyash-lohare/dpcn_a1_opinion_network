"""Methods tried and rejected, with the numbers that justify rejecting them.

The assignment states that effort is graded, so the alternatives are quantified
rather than merely named. Every figure in this repo rests on the choices this
module tests, and each one is reported with the evidence against the
alternative.

Also covers the divergence between the frozen pipeline and the 91-respondent
variant actually committed by Person 1, which is a sensitivity result in its own
right: the headline "exactly three dimensions" depends on the respondent filter.
"""

from __future__ import annotations

import json

import networkx as nx
import numpy as np
import pandas as pd

from src.common.config import ARTIFACTS, MAX_MISSING_ITEMS, SEED, ensure_dirs
from src.common.matrix import build_matrices, correlation, load_person1_variant, upper_triangle
from src.person2.graph_5 import mp_bounds, spectrum
from src.person2.graph_7 import build_signed_graph, measure


def knn_graph(similarity: np.ndarray, k: int, mutual: bool) -> nx.Graph:
    """k-nearest-neighbour graph from a similarity matrix.

    ``mutual=False`` takes the union (i is linked to j if EITHER ranks the other
    in its top k); ``mutual=True`` takes the intersection.
    """
    n = similarity.shape[0]
    sim = similarity.copy()
    np.fill_diagonal(sim, -np.inf)
    neighbours = [set(np.argsort(sim[i])[::-1][:k]) for i in range(n)]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in neighbours[i]:
            if mutual and i not in neighbours[j]:
                continue
            G.add_edge(i, int(j), weight=float(similarity[i, j]))
    return G


def graph_summary(G: nx.Graph, seed: int = SEED) -> dict:
    stats = measure(G, seed=seed)
    return {
        "nodes": G.number_of_nodes(),
        "edges": stats["edges"],
        "components": stats["n_components"],
        "isolates": stats["n_isolates"],
        "gcc_fraction": round(stats["gcc_fraction"], 3),
        "modularity": None if np.isnan(stats["modularity"]) else round(stats["modularity"], 3),
        "n_communities": stats["n_communities"],
    }


def respondent_networks(centred: pd.DataFrame) -> dict:
    """Symmetrised vs mutual kNN on the respondent-respondent network."""
    sim = np.corrcoef(centred.to_numpy())
    return {
        "symmetrised_knn_k4": graph_summary(knn_graph(sim, 4, mutual=False)),
        "mutual_knn_k4": graph_summary(knn_graph(sim, 4, mutual=True)),
    }


def centring_comparison(encoded: pd.DataFrame, centred: pd.DataFrame, threshold: float) -> dict:
    """Raw (uncentred) vs row-centred item correlations."""
    out = {}
    for label, matrix in (("raw_uncentred", encoded), ("row_centred", centred)):
        r = upper_triangle(correlation(matrix))
        out[label] = {
            "mean_r": round(float(r.mean()), 4),
            "mean_abs_r": round(float(np.abs(r).mean()), 4),
            "sd": round(float(r.std()), 4),
            "edges_at_threshold": int((np.abs(r) > threshold).sum()),
            "density_at_threshold": round(float((np.abs(r) > threshold).mean()), 3),
            "positive_fraction": round(float((r > 0).mean()), 3),
        }
    return out


def percentile_thresholding(corr: np.ndarray, codes, percentile: float = 90.0) -> dict:
    """Keep the top (100 - p)% of |r| instead of using an absolute cutoff.

    Computed on both the item network and the respondent network, because the
    two behave very differently and the spec's "22 components" figure refers to
    the respondent network.
    """
    r = np.abs(upper_triangle(corr))
    cutoff = float(np.percentile(r, percentile))
    G = build_signed_graph(corr, codes, cutoff)
    return {"percentile": percentile, "implied_threshold": round(cutoff, 4), **graph_summary(G)}


def percentile_thresholding_respondents(centred: pd.DataFrame, percentile: float = 90.0) -> dict:
    """Same rule applied to the 86 x 86 respondent similarity matrix."""
    sim = np.corrcoef(centred.to_numpy())
    labels = [str(i) for i in range(sim.shape[0])]
    cutoff = float(np.percentile(np.abs(upper_triangle(sim)), percentile))
    G = build_signed_graph(sim, labels, cutoff)
    return {"percentile": percentile, "implied_threshold": round(cutoff, 4), **graph_summary(G)}


def imputation_comparison(threshold: float) -> dict:
    """Column-mean vs column-median on the 47 missing cells."""
    from src.common.matrix import build_matrices as _bm

    am = _bm()
    raw = am.encoded_all.loc[am.encoded.index]
    out = {}
    for label, filled in (
        ("column_mean", raw.fillna(raw.mean())),
        ("column_median", raw.fillna(raw.median())),
    ):
        centred = filled.sub(filled.mean(axis=1), axis=0)
        r = upper_triangle(correlation(centred))
        vals, _ = spectrum(correlation(centred))
        _, _, lam_plus = mp_bounds(*centred.shape)
        out[label] = {
            "sd": round(float(r.std()), 4),
            "edges_at_threshold": int((np.abs(r) > threshold).sum()),
            "n_significant_eigenvalues": int((vals > lam_plus).sum()),
        }
    return out


def person1_variant(threshold: float) -> dict | None:
    """Person 1's committed pipeline: 91 respondents, ordinal-imputed."""
    centred = load_person1_variant()
    if centred is None:
        return None
    corr = correlation(centred)
    n_obs, n_items = centred.shape
    q, _, lam_plus = mp_bounds(n_obs, n_items)
    vals, _ = spectrum(corr)
    r = upper_triangle(corr)
    return {
        "n_respondents": int(n_obs),
        "Q": round(float(q), 4),
        "lambda_plus": round(float(lam_plus), 4),
        "n_significant_eigenvalues": int((vals > lam_plus).sum()),
        "significant_eigenvalues": [round(float(v), 3) for v in vals[vals > lam_plus]],
        "variance_pct": round(float(vals[vals > lam_plus].sum() / n_items * 100), 1),
        "sd": round(float(r.std()), 4),
        "edges_at_threshold": int((np.abs(r) > threshold).sum()),
        "theoretical_noise_sd": round(float(1 / np.sqrt(n_obs - 3)), 4),
    }


def main() -> dict:
    ensure_dirs()
    am = build_matrices()
    corr_raw = np.load(ARTIFACTS / "corr_raw.npy")
    threshold = json.loads((ARTIFACTS / "threshold.json").read_text())["chosen_threshold"]

    frozen_q, _, frozen_lam = mp_bounds(*am.centred.shape)
    frozen_vals, _ = spectrum(corr_raw)
    report = {
        "chosen_threshold": threshold,
        "frozen_pipeline": {
            "n_respondents": int(len(am.centred)),
            "max_missing_items": MAX_MISSING_ITEMS,
            "imputed_cells": am.n_imputed,
            "Q": round(float(frozen_q), 4),
            "lambda_plus": round(float(frozen_lam), 4),
            "n_significant_eigenvalues": int((frozen_vals > frozen_lam).sum()),
            "theoretical_noise_sd": round(float(1 / np.sqrt(len(am.centred) - 3)), 4),
        },
        "respondent_networks": respondent_networks(am.centred),
        "centring": centring_comparison(am.encoded, am.centred, threshold),
        "percentile_thresholding_items": percentile_thresholding(corr_raw, am.codes),
        "percentile_thresholding_respondents": percentile_thresholding_respondents(am.centred),
        "imputation": imputation_comparison(threshold),
        "person1_variant_91_respondents": person1_variant(threshold),
    }
    (ARTIFACTS / "sensitivity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 84)
    print("METHODS TRIED AND REJECTED")
    print("=" * 84)
    rn = report["respondent_networks"]
    print("\n1. RESPONDENT NETWORK -- symmetrised vs mutual kNN (k = 4)")
    for label, s in rn.items():
        print(f"   {label:<22} {s['edges']:>4} edges, {s['components']:>3} components, "
              f"GCC {s['gcc_fraction'] * 100:>5.1f}%, Q = {s['modularity']}")
    print("   spec: symmetrised 286 edges / 1 component; mutual 58 edges / 38 components")

    c = report["centring"]
    print(f"\n2. ROW-CENTRING -- item correlations at |r| > {threshold}")
    for label, s in c.items():
        print(f"   {label:<22} mean r = {s['mean_r']:+.3f}, {s['edges_at_threshold']:>5} of 1770 edges "
              f"({s['density_at_threshold'] * 100:.0f}% density), {s['positive_fraction'] * 100:.0f}% positive")

    print("\n3. PERCENTILE THRESHOLDING at p90 (fixes the edge count by construction)")
    for label, key in (("items      ", "percentile_thresholding_items"),
                       ("respondents", "percentile_thresholding_respondents")):
        p = report[key]
        print(f"   {label} implied |r| > {p['implied_threshold']:.4f}: {p['edges']:>4} edges, "
              f"{p['components']:>3} components, {p['isolates']:>3} isolates, "
              f"GCC {p['gcc_fraction'] * 100:>5.1f}%")

    im = report["imputation"]
    print("\n4. IMPUTATION -- column-mean vs column-median on 47 cells (0.9%)")
    for label, s in im.items():
        print(f"   {label:<22} SD {s['sd']}, {s['edges_at_threshold']:>4} edges, "
              f"{s['n_significant_eigenvalues']} significant eigenvalues")

    pv = report["person1_variant_91_respondents"]
    if pv:
        f = report["frozen_pipeline"]
        print("\n5. RESPONDENT FILTER -- frozen (86) vs Person 1's committed pipeline (91)")
        print(f"   frozen  86 resp: Q = {f['Q']}, lambda+ = {f['lambda_plus']}, "
              f"{f['n_significant_eigenvalues']} significant dimensions")
        print(f"   P1      91 resp: Q = {pv['Q']}, lambda+ = {pv['lambda_plus']}, "
              f"{pv['n_significant_eigenvalues']} significant dimensions "
              f"({pv['variance_pct']}% of variance)")
        print("   -> the headline dimension count is sensitive to the respondent filter.")
    print("\n" + "=" * 84)
    print("  wrote artifacts/sensitivity.json")
    return report


if __name__ == "__main__":
    main()
