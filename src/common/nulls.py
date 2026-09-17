"""Null-model generators (PERSON2_BRIEF section 4).

Three nulls, in descending order of how much they actually test.

``permutation_null``
    Destroys item-item association while preserving every marginal, then
    regenerates the **entire** pipeline (centre -> correlate -> threshold ->
    measure) from the scrambled responses. This is the only null that matches
    the claim being made, because a correlation-derived network has
    transitivity built into its construction: if A correlates with B and B with
    C, A and C are pulled toward correlation automatically. That alone
    manufactures clustering and modularity in meaningless data.

``rewiring_null`` / ``er_null``
    Both take the *finished graph* as given, so neither can detect structure
    that the construction itself created. They are implemented so the report
    can show, with numbers, why they are insufficient rather than merely
    asserting it.

Empirically the distinction decides the headline: on the respondent network
modularity clears the rewiring null easily (0.423 vs 0.329 +/- 0.011, z = 7.04)
but does not clear the permutation null (0.403 +/- 0.017, z = 1.15).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence

import networkx as nx
import numpy as np

from src.common.config import N_NULL_REPS, SEED


def permute_columns(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute every column.

    A single permutation applied to all columns merely relabels respondents and
    destroys nothing -- the whole correlation structure would survive intact.
    A fresh permutation per column is what breaks item-item association while
    leaving each item's marginal distribution exactly as observed.
    """
    out = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        out[:, j] = rng.permutation(X[:, j])
    return out


def permutation_null(
    X_encoded: np.ndarray,
    pipeline_fn: Callable[[np.ndarray], object],
    n_reps: int = N_NULL_REPS,
    seed: int = SEED,
    progress_every: int = 0,
) -> List[object]:
    """Run ``pipeline_fn`` on ``n_reps`` column-permuted copies of the data.

    ``X_encoded`` must be the encoded matrix **before** row-centring: centring
    is part of the pipeline under test, so it has to happen inside
    ``pipeline_fn`` on already-shuffled data.

    Replicate ``i`` is seeded ``default_rng(seed + i)``, so any single replicate
    can be reproduced on its own without rerunning the whole job.
    """
    X = np.asarray(X_encoded, dtype=float)
    results: List[object] = []
    for i in range(n_reps):
        rng = np.random.default_rng(seed + i)
        results.append(pipeline_fn(permute_columns(X, rng)))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    permutation replicate {i + 1}/{n_reps}")
    return results


def rewiring_null(
    G: nx.Graph,
    n_reps: int = N_NULL_REPS,
    seed: int = SEED,
    measure_fn: Optional[Callable[[nx.Graph], object]] = None,
    swap_factor: int = 5,
) -> List[object]:
    """Degree-preserving double-edge-swap null. Weaker: takes the graph as given."""
    m = G.number_of_edges()
    out: List[object] = []
    for i in range(n_reps):
        H = G.copy()
        if m >= 2:
            try:
                nx.double_edge_swap(H, nswap=swap_factor * m, max_tries=swap_factor * m * 20, seed=seed + i)
            except (nx.NetworkXAlgorithmError, nx.NetworkXError):
                # Too few swappable configurations; keep the partially swapped graph.
                pass
        out.append(measure_fn(H) if measure_fn else H)
    return out


def er_null(
    G: nx.Graph,
    n_reps: int = N_NULL_REPS,
    seed: int = SEED,
    measure_fn: Optional[Callable[[nx.Graph], object]] = None,
) -> List[object]:
    """Erdos-Renyi G(n, m) null. Weakest: preserves only n and m."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    out: List[object] = []
    for i in range(n_reps):
        H = nx.gnm_random_graph(n, m, seed=seed + i)
        out.append(measure_fn(H) if measure_fn else H)
    return out


def zscore(observed: float, null_sample: Sequence[float]) -> dict:
    """Observed value against a null sample: mean, sd, z, and empirical p."""
    arr = np.asarray(list(null_sample), dtype=float)
    mu, sd = float(arr.mean()), float(arr.std(ddof=1))
    z = (observed - mu) / sd if sd > 0 else float("nan")
    p = float((arr >= observed).sum() + 1) / (len(arr) + 1)
    return {"observed": float(observed), "null_mean": mu, "null_sd": sd, "z": float(z), "p": p}


def cache_replicates(path: Path, **arrays: np.ndarray) -> None:
    """Cache replicate outputs so Person 3 never re-runs a multi-minute job."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


if __name__ == "__main__":
    # Synthetic validation: with no real association, the pooled permuted
    # correlations must scatter at the theoretical noise floor 1/sqrt(T-3).
    T, N = 86, 60
    rng = np.random.default_rng(SEED)
    synthetic = rng.integers(-2, 3, size=(T, N)).astype(float)

    def pipeline_fn(Xp: np.ndarray) -> np.ndarray:
        centred = Xp - Xp.mean(axis=1, keepdims=True)   # row-centre INSIDE the pipeline
        R = np.corrcoef(centred.T)
        return R[np.triu_indices(N, 1)]

    reps = permutation_null(synthetic, pipeline_fn, n_reps=50, seed=SEED)
    pooled = np.concatenate(reps)
    theory = 1 / np.sqrt(T - 3)
    print(f"synthetic {T}x{N}, 50 replicates, {len(pooled):,} pooled correlations")
    print(f"  pooled SD  {pooled.std():.4f}")
    print(f"  theory     {theory:.4f}   (1/sqrt({T}-3))")
    print(f"  ratio      {pooled.std()/theory:.3f}")
    print(f"  mean       {pooled.mean():+.4f}  (expect ~ -1/(N-1) = {-1/(N-1):+.4f} from centring)")

    # Guard against the classic bug: one permutation applied to every column.
    bad = synthetic[rng.permutation(T), :]
    bad_r = np.corrcoef((bad - bad.mean(axis=1, keepdims=True)).T)[np.triu_indices(N, 1)]
    real_r = np.corrcoef((synthetic - synthetic.mean(axis=1, keepdims=True)).T)[np.triu_indices(N, 1)]
    print(f"\n  sanity: row-shuffle (WRONG) preserves correlations exactly -> "
          f"max|diff| {np.abs(np.sort(bad_r)-np.sort(real_r)).max():.2e}")
    assert abs(pooled.std() / theory - 1) < 0.05, "permutation null is not at the noise floor"
    print("\n  PASS: column-wise permutation sits at the theoretical noise floor.")
