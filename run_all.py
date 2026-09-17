"""Reproduce every figure and number in the report from a fresh clone.

    python run_all.py              # everything, in dependency order
    python run_all.py --only P2    # one owner's stages
    python run_all.py --list       # show the stage table and exit

Orchestration only: no analysis logic lives here. Each stage is a callable in
somebody's module, and the stage table below is the dependency order.

Stage 1 fits per-item ordinal regression models and is by far the slowest part
of the run; everything Person 2 owns completes in a few seconds.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"

# Which person owes which artefact, so a missing input names its owner.
ARTEFACT_OWNER = {
    "outputs/sanitised_data/dataset_imputed.csv": "P1 (src/pipeline.py, stage 1)",
    "outputs/sanitised_data/row_centred_data.csv": "P1 (src/pipeline.py, stage 1)",
    "outputs/block_connectivity/block_means_centred.csv": "P1 (src/block_connectivity.py)",
    "artifacts/chance_edges_by_threshold.csv": "P2 (src/person2/graph_4.py, stage 3)",
    "artifacts/null_replicates.npz": "P2 (src/person2/graph_4.py, stage 3)",
    "artifacts/corr_raw.npy": "P2 (src/person2/graph_5.py, stage 4)",
    "artifacts/corr_denoised.npy": "P2 (src/person2/graph_5.py, stage 4)",
    "artifacts/mp_dimensions.json": "P2 (src/person2/graph_5.py, stage 4)",
    "artifacts/threshold.json": "P2 (src/person2/graph_7.py, stage 5)",
    "artifacts/null_comparison.json": "P2 (src/person2/null_comparison.py, stage 6)",
}

THRESHOLD_RULE = (
    "highest cutoff at which the giant component still spans more than two-thirds of the "
    "60 items and mean degree remains above 4, so that community and path-based metrics "
    "stay meaningful; spectral denoising has already removed the sampling noise that the "
    "chance-edge column measures, so a higher cutoff would penalise the same noise twice"
)
CHOSEN_THRESHOLD = 0.22


@dataclass
class Stage:
    owner: str
    name: str
    run: Callable[[], object]
    requires: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)


# --- stage callables -------------------------------------------------------


def _p1_pipeline():
    from src.config import PipelineConfig
    from src.pipeline import run_pipeline

    return run_pipeline(PipelineConfig())


def _p1_diagnostics():
    from src.diagnostics import (
        plot_item_variance_ranking,
        plot_missingness_and_response_distribution,
        plot_row_centring_effect,
    )

    plot_missingness_and_response_distribution()
    plot_item_variance_ranking()
    plot_row_centring_effect()


def _p1_graph_9():
    from src.block_connectivity import plot_block_connectivity

    return plot_block_connectivity()


def _p1_graph_12():
    from src.dynamics import plot_opinion_dynamics

    return plot_opinion_dynamics()


def _p2_graph_4():
    from src.person2.graph_4 import main

    return main()


def _p2_graph_5():
    from src.person2.graph_5 import main

    return main()


def _p2_graph_7():
    from src.person2.graph_7 import finalise, print_table, run_sweep

    table, chance, _ = run_sweep()
    print_table(table, chance)
    return finalise(CHOSEN_THRESHOLD, THRESHOLD_RULE)


def _p2_null_comparison():
    from src.person2.null_comparison import main

    return main()


def _p2_sensitivity():
    from src.person2.sensitivity import main

    return main()


def _p3_graph_10():
    from src.person3.graph_10 import run
    return run()


def _p3_graph_11():
    from src.person3.graph_11 import run
    return run()


def _p3_graph_6():
    from src.person3.graph_6 import run
    return run()


STAGES: List[Stage] = [
    Stage("P1", "Sanitise, impute, row-centre, item network", _p1_pipeline,
          produces=["outputs/sanitised_data/dataset_imputed.csv",
                    "outputs/sanitised_data/row_centred_data.csv"]),
    Stage("P1", "Graphs 1-3 (missingness, item variance, centring effect)", _p1_diagnostics,
          requires=["outputs/sanitised_data/dataset_imputed.csv"]),
    Stage("P1", "Graph 9: topic-block connectivity (4x4)", _p1_graph_9,
          requires=["outputs/sanitised_data/dataset_imputed.csv"],
          produces=["outputs/block_connectivity/block_means_centred.csv"]),
    Stage("P1", "Graph 12: Deffuant-Weisbuch opinion dynamics", _p1_graph_12,
          requires=["outputs/sanitised_data/row_centred_data.csv",
                    "outputs/sanitised_data/dataset_imputed.csv"]),
    Stage("P2", "Graph 4: noise floor + permutation null framework", _p2_graph_4,
          produces=["artifacts/chance_edges_by_threshold.csv", "artifacts/null_replicates.npz"]),
    Stage("P2", "Graph 5: Marchenko-Pastur denoising", _p2_graph_5,
          produces=["artifacts/corr_raw.npy", "artifacts/corr_denoised.npy",
                    "artifacts/mp_dimensions.json"]),
    Stage("P2", "Graph 7: threshold sweep and selection", _p2_graph_7,
          requires=["artifacts/corr_denoised.npy", "artifacts/chance_edges_by_threshold.csv"],
          produces=["artifacts/threshold.json"]),
    Stage("P2", "Null comparison: permutation vs rewiring vs ER", _p2_null_comparison,
          produces=["artifacts/null_comparison.json",
                    "artifacts/null_modularity_replicates.npz"]),
    Stage("P2", "Sensitivity annex: methods tried and rejected", _p2_sensitivity,
          requires=["artifacts/corr_raw.npy", "artifacts/threshold.json"],
          produces=["artifacts/sensitivity.json"]),
    Stage("P3", "Graph 10: communities vs three null models", _p3_graph_10,
          requires=["artifacts/corr_denoised.npy", "artifacts/threshold.json",
                    "artifacts/null_replicates.npz"],
          produces=["artifacts/graph10_communities.json"]),
    Stage("P3", "Graph 11: structural balance vs threshold", _p3_graph_11,
          requires=["artifacts/corr_denoised.npy", "artifacts/threshold.json"],
          produces=["artifacts/graph11_balance_stats.json"]),
    Stage("P3", "Graph 6: reordered correlation heatmap", _p3_graph_6,
          requires=["artifacts/corr_denoised.npy", "artifacts/graph10_communities.json"])
]


def check_requirements(stage: Stage) -> List[str]:
    """Missing inputs, annotated with who owes them."""
    missing = []
    for rel in stage.requires:
        if not (ROOT / rel).exists():
            missing.append(f"{rel}  (owed by {ARTEFACT_OWNER.get(rel, 'unknown')})")
    return missing


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", metavar="OWNER", help="run only P1, P2 or P3 stages")
    parser.add_argument("--list", action="store_true", help="print the stage table and exit")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue after a failing stage instead of stopping")
    args = parser.parse_args(argv)

    stages = STAGES
    if args.only:
        stages = [s for s in STAGES if s.owner.upper() == args.only.upper()]
        if not stages:
            print(f"no stages for owner {args.only!r}")
            return 2

    if args.list:
        print(f"{'#':>2}  {'owner':<6} stage")
        for i, s in enumerate(STAGES, 1):
            print(f"{i:>2}  {s.owner:<6} {s.name}")
        return 0

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    total = time.perf_counter()
    failures = 0

    for i, stage in enumerate(stages, 1):
        header = f"[{i}/{len(stages)}] {stage.owner}  {stage.name}"
        print("\n" + "=" * 78)
        print(header)
        print("=" * 78)

        missing = check_requirements(stage)
        if missing:
            print("  SKIPPED -- missing inputs:")
            for m in missing:
                print(f"    - {m}")
            failures += 1
            if not args.keep_going:
                print("\n  stopping; rerun with --keep-going to continue past this.")
                return 1
            continue

        start = time.perf_counter()
        try:
            stage.run()
        except NotImplementedError as exc:
            print(f"  NOT IMPLEMENTED: {exc}")
            continue
        except Exception:
            print(f"  FAILED after {time.perf_counter() - start:.1f}s")
            traceback.print_exc()
            failures += 1
            if not args.keep_going:
                return 1
            continue
        print(f"\n  done in {time.perf_counter() - start:.1f}s")

        absent = [p for p in stage.produces if not (ROOT / p).exists()]
        if absent:
            print("  WARNING -- stage did not produce: " + ", ".join(absent))

    print("\n" + "=" * 78)
    print(f"total {time.perf_counter() - total:.1f}s, {failures} stage(s) failed or skipped")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
