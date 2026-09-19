"""Graph 11 -- structural balance vs threshold (effort 5).

Line plot. X-axis: edge threshold |r|. Y-axis: fraction of closed triads
whose sign product is positive (balanced). Observed line plus a shaded band
for the sign-permutation null (~50%). Triad counts annotated at key
thresholds.

The concept: a triad of three mutually connected items is balanced when the
product of its three edge signs is positive (+++ or +--). It is unbalanced
when the product is negative (++- or ---). Balanced triads mean internally
consistent opinion patterns; unbalanced means contradiction.

The sweep matters more than any single number. At |r| > 0.15 two thirds of
edges are chance, yet 83% of triads are balanced. As the threshold tightens
to 0.20 and 0.25, the balanced fraction rises to 96% and 100%, while the
null stays near 50%. That trend is the validation.
"""

from __future__ import annotations

import json
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.config import (
    ARTIFACTS,
    DPI,
    FIGDIR,
    MARK_COLOR,
    NULL_COLOR,
    REAL_COLOR,
    SEED,
    THRESHOLD_GRID,
    ensure_dirs,
)


# ---------------------------------------------------------------------------
# Graph building (same logic as Shreyash/graph_7.py and Dev/graph_10.py)
# ---------------------------------------------------------------------------

def build_signed_graph_edges(corr: np.ndarray, codes: list,
                             threshold: float) -> dict:
    """Return edge list with signs for items above |r| = threshold.

    Returns a dict: {(i, j): sign} for codes[i], codes[j].
    Also returns adjacency as a set of frozen-set pairs for fast lookup.
    """
    n = len(codes)
    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            r = float(corr[i, j])
            if abs(r) > threshold:
                edges[(i, j)] = 1 if r > 0 else -1
    return edges


# ---------------------------------------------------------------------------
# Triad enumeration and balance computation
# ---------------------------------------------------------------------------

def compute_balance(corr: np.ndarray, codes: list,
                    threshold: float) -> dict:
    """Compute balanced-triad fraction at a single threshold.

    Enumerates all closed triads via itertools.combinations (feasible
    at 60 nodes: C(60,3) = 34,220 candidate triples, most not closed).
    """
    n = len(codes)
    edges = build_signed_graph_edges(corr, codes, threshold)

    # Build adjacency for fast lookup
    adj = {}
    for (i, j), sign in edges.items():
        adj[(i, j)] = sign
        adj[(j, i)] = sign

    # Enumerate closed triads
    total_triads = 0
    balanced_triads = 0
    for i, j, k in combinations(range(n), 3):
        if (i, j) in adj and (j, k) in adj and (i, k) in adj:
            sign_product = adj[(i, j)] * adj[(j, k)] * adj[(i, k)]
            total_triads += 1
            if sign_product > 0:
                balanced_triads += 1

    balanced_frac = (balanced_triads / total_triads
                     if total_triads > 0 else float("nan"))
    return {
        "threshold": float(threshold),
        "total_triads": total_triads,
        "balanced_triads": balanced_triads,
        "unbalanced_triads": total_triads - balanced_triads,
        "balanced_fraction": balanced_frac,
    }


def sign_permutation_null(corr: np.ndarray, codes: list,
                          threshold: float,
                          n_reps: int = 500,
                          seed: int = SEED) -> np.ndarray:
    """Permute edge signs (keep structure fixed), recompute balance.

    For each replicate: shuffle signs across existing edges, then count
    balanced triads. Returns array of balanced fractions.
    """
    n = len(codes)
    edges = build_signed_graph_edges(corr, codes, threshold)
    if not edges:
        return np.full(n_reps, float("nan"))

    edge_list = list(edges.keys())
    signs = np.array([edges[e] for e in edge_list])

    # Build adjacency index for triad lookup
    adj_template = {}
    for idx, (i, j) in enumerate(edge_list):
        adj_template[(i, j)] = idx
        adj_template[(j, i)] = idx

    # Find all closed triads once (indices into edge_list)
    triad_indices = []
    for i, j, k in combinations(range(n), 3):
        if ((i, j) in adj_template and (j, k) in adj_template
                and (i, k) in adj_template):
            triad_indices.append((
                adj_template[(i, j)],
                adj_template[(j, k)],
                adj_template[(i, k)],
            ))

    if not triad_indices:
        return np.full(n_reps, float("nan"))

    triad_arr = np.array(triad_indices)  # (n_triads, 3)
    total = len(triad_indices)

    rng = np.random.default_rng(seed)
    results = np.empty(n_reps)
    for rep in range(n_reps):
        perm_signs = rng.permutation(signs)
        # Product of three signs for each triad
        products = (perm_signs[triad_arr[:, 0]]
                    * perm_signs[triad_arr[:, 1]]
                    * perm_signs[triad_arr[:, 2]])
        results[rep] = float(np.sum(products > 0)) / total

    return results


# ---------------------------------------------------------------------------
# Sweep across thresholds
# ---------------------------------------------------------------------------

def balance_sweep(corr: np.ndarray, codes: list,
                  grid=THRESHOLD_GRID,
                  null_reps: int = 500,
                  seed: int = SEED) -> list:
    """Compute balance stats + null at each threshold in the grid."""
    results = []
    for i, t in enumerate(grid):
        t_val = float(t)
        stats = compute_balance(corr, codes, t_val)

        # Only compute null for thresholds with enough triads
        if stats["total_triads"] >= 3:
            null_fracs = sign_permutation_null(
                corr, codes, t_val, n_reps=null_reps, seed=seed)
            stats["null_mean"] = float(np.nanmean(null_fracs))
            stats["null_sd"] = float(np.nanstd(null_fracs, ddof=1))
            if stats["null_sd"] > 0:
                stats["z"] = ((stats["balanced_fraction"] - stats["null_mean"])
                              / stats["null_sd"])
            else:
                stats["z"] = float("nan")
        else:
            stats["null_mean"] = float("nan")
            stats["null_sd"] = float("nan")
            stats["z"] = float("nan")

        results.append(stats)
        if (i + 1) % 10 == 0:
            print(f"    threshold {t_val:.2f}: {stats['total_triads']} triads, "
                  f"balanced = {stats['balanced_fraction']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_graph_11(results: list, operating_threshold: float,
                  path_stem: str) -> None:
    """Structural balance vs threshold with null band."""
    fig, ax = plt.subplots(figsize=(8, 4.8))

    thresholds = [r["threshold"] for r in results]
    balanced = [r["balanced_fraction"] for r in results]
    null_means = [r["null_mean"] for r in results]
    null_sds = [r["null_sd"] for r in results]
    triad_counts = [r["total_triads"] for r in results]

    # Only plot where we have triads
    mask = [t > 0 for t in triad_counts]
    th_plot = [t for t, m in zip(thresholds, mask) if m]
    bal_plot = [b for b, m in zip(balanced, mask) if m]
    nm_plot = [n for n, m in zip(null_means, mask) if m]
    ns_plot = [s for s, m in zip(null_sds, mask) if m]
    tc_plot = [c for c, m in zip(triad_counts, mask) if m]

    # Observed line
    ax.plot(th_plot, bal_plot, color=REAL_COLOR, linewidth=2.5,
            label="Observed balanced fraction", zorder=4)

    # Null band
    nm_arr = np.array(nm_plot)
    ns_arr = np.array(ns_plot)
    valid_null = ~np.isnan(nm_arr)
    th_null = np.array(th_plot)[valid_null]
    nm_valid = nm_arr[valid_null]
    ns_valid = ns_arr[valid_null]

    ax.fill_between(th_null, nm_valid - ns_valid, nm_valid + ns_valid,
                    color=NULL_COLOR, alpha=0.25,
                    label="Sign-permutation null (mean +/- SD)")
    ax.plot(th_null, nm_valid, color=NULL_COLOR, linewidth=1.5,
            linestyle="--", alpha=0.7)

    # Operating threshold
    ax.axvline(operating_threshold, color=MARK_COLOR, linewidth=1.8,
               alpha=0.85, linestyle="-.",
               label=f"Operating threshold |r| = {operating_threshold}")

    # Annotate triad counts at key thresholds
    key_thresholds = [0.15, 0.20, 0.25, 0.30]
    for kt in key_thresholds:
        # Find closest
        idx = min(range(len(th_plot)),
                  key=lambda i: abs(th_plot[i] - kt))
        if abs(th_plot[idx] - kt) < 0.005:
            ax.annotate(
                f"{tc_plot[idx]} triads\n{bal_plot[idx]*100:.0f}%",
                xy=(th_plot[idx], bal_plot[idx]),
                xytext=(10, -20 if kt < 0.25 else 10),
                textcoords="offset points",
                fontsize=8, color=MARK_COLOR, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=MARK_COLOR,
                                lw=1.0),
                zorder=5,
            )

    ax.set_xlabel("Edge threshold $|r|$", fontsize=11)
    ax.set_ylabel("Balanced triad fraction", fontsize=11)
    ax.set_title(
        "Structural balance rises as threshold tightens\n"
        "Sign-permutation null stays near 50%",
        fontsize=12, loc="left")
    ax.set_xlim(0.05, 0.50)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> dict:
    """Generate Graph 11: structural balance vs threshold."""
    ensure_dirs()

    # Load artifacts
    corr = np.load(ARTIFACTS / "corr_denoised.npy")
    items_info = json.loads(
        (ARTIFACTS / "items.json").read_text(encoding="utf-8"))
    threshold_info = json.loads(
        (ARTIFACTS / "threshold.json").read_text(encoding="utf-8"))
    codes = items_info["codes"]
    operating_threshold = threshold_info["chosen_threshold"]

    print("  computing structural balance across threshold sweep...")
    results = balance_sweep(corr, codes, grid=THRESHOLD_GRID,
                            null_reps=500, seed=SEED)

    # Report key numbers
    print("\n  " + "=" * 70)
    print("  GRAPH 11 STRUCTURAL BALANCE")
    print("  " + "=" * 70)
    print(f"  {'threshold':>10} {'triads':>8} {'balanced':>10} "
          f"{'null mean':>10} {'z':>8}")
    for r in results:
        if r["total_triads"] > 0:
            print(f"  {r['threshold']:>10.2f} {r['total_triads']:>8} "
                  f"{r['balanced_fraction']:>10.3f} "
                  f"{r['null_mean']:>10.3f} {r['z']:>8.2f}")
    print("  " + "=" * 70)

    # Plot
    plot_graph_11(results, operating_threshold,
                  str(FIGDIR / "fig11_structural_balance"))
    print(f"\n  wrote {FIGDIR / 'fig11_structural_balance'}.png / .pdf")

    # Save artifact
    payload = {
        "operating_threshold": operating_threshold,
        "sweep": results,
    }
    (ARTIFACTS / "graph11_balance_stats.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print("  wrote artifacts/graph11_balance_stats.json")

    return payload


if __name__ == "__main__":
    run()

